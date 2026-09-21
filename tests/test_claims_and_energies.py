"""Smoke tests for the claim extraction + four energies pipeline.

These deliberately avoid any GPU/model dependency: they only exercise the
dependency-free fallback paths (hashed n-gram embeddings, Gaussian geometry
density, embedding-similarity verifier), so they run in a minimal install
(no torch/transformers/sentence-transformers/scikit-learn required) and are
safe to run in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from hallucination_energy.calibration.conformal import conformal_pvalue, conformal_threshold
from hallucination_energy.claims.extraction import extract_claims
from hallucination_energy.energies.codelength import calibrate_surprisal_stats, codelength_energy
from hallucination_energy.energies.embeddings import embed_texts, l2_normalize, pairwise_cost
from hallucination_energy.energies.evidence import evidence_energy
from hallucination_energy.energies.factorized_energy import compute_energy_parts, composite_energy
from hallucination_energy.energies.geometry import fit_geometry_density, geometry_energy
from hallucination_energy.energies.invariance import invariance_energy, sinkhorn_unbalanced
from hallucination_energy.energies.normalization import EnergyNormalizer
from hallucination_energy.evaluation.diagnostics import build_energy_matrix, energy_correlation_matrix
from hallucination_energy.evaluation.metrics import auroc
from hallucination_energy.retrieval.retriever import ContextRetriever
from hallucination_energy.retrieval.verifier import default_verifier
from hallucination_energy.training.negatives import corrupt_claim_set, jitter_numeric_text


@pytest.fixture(autouse=True)
def _force_embedding_fallback(monkeypatch):
    """Force the embedding-similarity fallback in retrieval.verifier instead of
    attempting to download the ~1.5GB DeBERTa entailment model. Since torch/
    transformers are core dependencies, `get_entailment_model()` would
    otherwise actually succeed here, making this "dependency-free smoke
    test" module slow and network-dependent."""
    monkeypatch.setattr("hallucination_energy.retrieval.verifier.get_entailment_model", lambda: None)


def test_extract_claims_splits_sentences():
    text = "Paris is the capital of France. It has about 2.1 million residents."
    claim_set = extract_claims(text)
    assert len(claim_set.claims) >= 2
    assert any("Paris" in c.text for c in claim_set.claims)


def test_extract_claims_empty_text():
    claim_set = extract_claims("")
    assert claim_set.claims == []


def test_embed_texts_fallback_shapes():
    embs, metric = embed_texts(["hello world", "goodbye world"], metric="cosine")
    assert embs.shape[0] == 2
    assert metric == "cosine"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_pairwise_cost_and_sinkhorn_zero_for_identical_sets():
    embs, _ = embed_texts(["a fact about paris", "a fact about berlin"], metric="cosine")
    C = pairwise_cost(embs, embs, metric="cosine")
    a = np.ones(2) / 2
    b = np.ones(2) / 2
    dist = sinkhorn_unbalanced(C, a, b, reg=0.05, tau_src=1.0, tau_tgt=1.0)
    assert dist >= 0.0
    assert dist < 0.5  # transporting a distribution to itself should be cheap


def test_invariance_energy_low_for_paraphrase_of_itself():
    claim_set = extract_claims("The Eiffel Tower is in Paris.")
    energy_same = invariance_energy(claim_set, ["The Eiffel Tower is in Paris."])
    energy_far = invariance_energy(claim_set, ["Quantum entanglement violates locality in Bell tests."])
    assert energy_same <= energy_far


def test_evidence_energy_lower_when_supported():
    claim_set = extract_claims("Insulin is secreted by pancreatic beta cells.")
    supporting_doc = "Pancreatic beta cells secrete insulin in response to blood glucose."
    unrelated_doc = "The stock market closed higher today amid tech earnings."

    supported = evidence_energy(claim_set, ContextRetriever([supporting_doc]), default_verifier)
    unsupported = evidence_energy(claim_set, ContextRetriever([unrelated_doc]), default_verifier)
    assert supported <= unsupported


def test_codelength_energy_and_calibration():
    stats = calibrate_surprisal_stats(
        {
            "ex1": {"most_likely_answer": {"accuracy": 1.0, "token_log_likelihoods": [-0.1, -0.2, -0.1]}},
            "ex2": {"most_likely_answer": {"accuracy": 0.0, "token_log_likelihoods": [-5.0, -6.0]}},
        }
    )
    assert stats["mean"] > 0  # only the accurate example contributes
    energy = codelength_energy([-0.1, -0.2, -0.1], stats=stats, use_zscore=True)
    assert isinstance(energy, float)


def test_geometry_energy_higher_for_outlier():
    corpus = [f"Water boils at 100 degrees Celsius at sea level, fact {i}." for i in range(10)]
    density_model, meta = fit_geometry_density(corpus, n_components=3)
    trusted = extract_claims("Water boils at 100 degrees Celsius at sea level, fact 1.")
    outlier = extract_claims("Zzyx flooble wobbulon nonsense text unrelated to anything.")
    e_trusted = geometry_energy(trusted, density_model, meta)
    e_outlier = geometry_energy(outlier, density_model, meta)
    assert e_outlier >= e_trusted


def test_energy_normalizer_zscores():
    calib = [{"inv": 0.0, "ev": 0.0, "cl": 0.0, "geo": 0.0}, {"inv": 2.0, "ev": 2.0, "cl": 2.0, "geo": 2.0}]
    normalizer = EnergyNormalizer.fit(calib)
    z = normalizer.transform({"inv": 1.0, "ev": 1.0, "cl": 1.0, "geo": 1.0})
    for v in z.values():
        assert abs(v) < 1e-6  # midpoint should normalize to ~0


def test_compute_energy_parts_end_to_end():
    corpus = ["Insulin lowers blood glucose.", "Photosynthesis occurs in chloroplasts."]
    density_model, meta = fit_geometry_density(corpus, n_components=2)
    stats = {"mean": 0.0, "std": 1.0}
    answer = {"response": "Insulin lowers blood glucose by promoting cellular uptake.", "token_log_likelihoods": [-0.2, -0.3, -0.1]}
    retriever = ContextRetriever(["Insulin promotes glucose uptake by cells, lowering blood sugar."])
    parts, claim_set = compute_energy_parts(
        "What does insulin do?", answer, [], retriever, default_verifier, density_model, meta, stats
    )
    assert set(parts.keys()) == {"inv", "ev", "cl", "geo"}
    assert len(claim_set.claims) >= 1
    energy = composite_energy({"inv": 1.0, "ev": 1.0, "cl": 1.0, "geo": 1.0}, parts)
    assert isinstance(energy, float)


def test_negatives_are_naive_but_functional():
    rng = np.random.default_rng(0)
    claim_set = extract_claims("Paris is in France. Einstein developed relativity.")
    corrupted = corrupt_claim_set(claim_set, rng=rng)
    assert corrupted != ""
    jittered = jitter_numeric_text("The year was 1990.", rng=rng)
    assert jittered != ""


def test_conformal_threshold_and_pvalue_consistent():
    calib_energies = [1.0, 2.0, 3.0, 4.0, 5.0]
    tau = conformal_threshold(calib_energies, alpha=0.2)
    assert tau in calib_energies
    p_low = conformal_pvalue(calib_energies, 0.5)
    p_high = conformal_pvalue(calib_energies, 10.0)
    assert p_low > p_high  # a more extreme (higher) energy should get a smaller p-value


def test_diagnostics_matrix_and_correlation():
    records = [
        ({"inv": 0.1, "ev": 0.2, "cl": 0.3, "geo": 0.1}, 0),
        ({"inv": 2.0, "ev": 2.1, "cl": 1.9, "geo": 2.2}, 1),
        ({"inv": 0.2, "ev": 0.1, "cl": 0.2, "geo": 0.3}, 0),
        ({"inv": 1.8, "ev": 2.0, "cl": 2.1, "geo": 1.9}, 1),
    ]
    X, y = build_energy_matrix(records)
    assert X.shape == (4, 4)
    corr = energy_correlation_matrix(X)
    assert corr.shape == (4, 4)
    score = auroc(y, X[:, 0])
    assert 0.0 <= score <= 1.0
