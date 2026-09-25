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
from hallucination_energy.energies.embeddings import embed_texts, fit_whitener, apply_whitener, l2_normalize, pairwise_cost
from hallucination_energy.energies.evidence import evidence_energy
from hallucination_energy.energies.factorized_energy import compute_energy_parts, composite_energy
from hallucination_energy.energies.geometry import fit_geometry_density, geometry_energy
from hallucination_energy.energies.invariance import invariance_energy, sinkhorn_unbalanced
from hallucination_energy.energies.normalization import EnergyNormalizer
from hallucination_energy.evaluation.diagnostics import build_energy_matrix, composite_auroc, energy_correlation_matrix
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


def test_embed_texts_falls_back_on_empty_tfidf_vocabulary():
    """Regression test for a real crash on a 200-example run:
    TfidfVectorizer.fit_transform raises ValueError('empty vocabulary; ...')
    when every text in the batch is too short (e.g. single digits/letters)
    to produce any token under its default 2+-character tokenizer. This
    should fall back to the hashed embedding instead of propagating."""
    embs, metric = embed_texts(["2", "a"])
    assert embs.shape[0] == 2
    assert np.all(np.isfinite(embs))


def test_embed_texts_pair_falls_back_on_empty_tfidf_vocabulary():
    from hallucination_energy.energies.embeddings import embed_texts_pair

    e1, e2, _ = embed_texts_pair(["2"], ["a"])
    assert e1.shape[0] == 1 and e2.shape[0] == 1
    assert np.all(np.isfinite(e1)) and np.all(np.isfinite(e2))


def test_embed_texts_with_vectorizer_falls_back_on_empty_tfidf_vocabulary():
    from hallucination_energy.energies.embeddings import embed_texts_with_vectorizer

    embs, _, vectorizer = embed_texts_with_vectorizer(["2", "a"])
    assert embs.shape[0] == 2
    assert vectorizer is None  # hashed fallback engaged, nothing to persist
    assert np.all(np.isfinite(embs))


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


def test_whitener_truncates_to_data_rank_with_few_samples_high_dimensional_features():
    """Regression test for a real failure observed end-to-end: with far
    fewer samples than feature dimensions (the common case for the
    "trusted claim" geometry corpus with TF-IDF fallback embeddings, up to
    4096-D), a full-ambient-dimensionality whitener feeds a GMM thousands
    of directions the data never actually supports. sklearn's per-dimension
    covariance regularization summed over ~4000 such directions produced a
    geometry energy of ~99,000-99,900 (confirmed via energy_matrix.npz on a
    real 20-example run) versus single-digit-to-tens values for the other
    three energies — a curse-of-dimensionality artifact, not signal.

    The fix: truncate whitening output to the data's actual rank
    (at most n_samples - 1 directions can carry any signal), so any
    downstream density model only ever sees informative dimensions."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 8, 200  # n << p, matching the real failure mode
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    whitener = fit_whitener(X)
    whitened = apply_whitener(X, whitener)

    assert whitened.shape == (n_samples, n_samples - 1)  # truncated, not full 200-D
    assert np.all(np.isfinite(whitened))
    assert np.abs(whitened).max() < 100 * np.abs(X).max()


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


def test_energy_normalizer_does_not_explode_on_near_constant_factor():
    """Regression test for a real finding on ada: when a factor (there,
    invariance energy) is ~constant on the reference split — e.g. because
    that split lacks the paraphrase samples needed to compute it
    meaningfully — its std must not fall back to the tiny `eps` floor.
    Dividing by ~1e-6 turned a real but small difference into a z-score
    in the thousands, making that factor numerically dominate a downstream
    learned fusion despite carrying no real signal. It should instead be
    left unscaled (std=1) when no reliable variance estimate exists."""
    calib = [{"inv": 0.001, "ev": 1.0, "cl": 1.0, "geo": 1.0}] * 49 + [{"inv": 0.001, "ev": 3.0, "cl": 3.0, "geo": 3.0}]
    normalizer = EnergyNormalizer.fit(calib)
    assert normalizer.stds["inv"] == 1.0
    assert "inv" in normalizer.degenerate_keys
    z = normalizer.transform({"inv": 0.5, "ev": 2.0, "cl": 2.0, "geo": 2.0})
    assert abs(z["inv"]) < 1.0  # centered but not exploded


def test_energy_normalizer_fit_with_fallback_patches_degenerate_factor():
    """A factor degenerate on the primary (e.g. reference) set should be
    refit from the fallback set instead of staying silenced at std=1 —
    left unscaled, it would count for ~nothing next to properly z-scored
    peers in an equal- or learned-weight linear combination."""
    # inv is constant (degenerate) on the primary set; ev/cl/geo have real variance.
    primary = [{"inv": 0.001, "ev": 1.0 + 0.01 * i, "cl": 2.0 + 0.02 * i, "geo": 3.0 + 0.03 * i} for i in range(20)]
    fallback = [{"inv": 0.1 * i, "ev": 0.0, "cl": 0.0, "geo": 0.0} for i in range(20)]
    normalizer = EnergyNormalizer.fit_with_fallback(primary, fallback)
    assert "inv" not in normalizer.degenerate_keys
    assert normalizer.stds["inv"] > 0.1  # refit from the fallback's real spread, not left at 1.0
    assert normalizer.stds["ev"] != 1.0  # untouched: ev wasn't degenerate on primary


def test_energy_normalizer_fit_with_fallback_stays_degenerate_if_fallback_also_degenerate():
    primary = [{"inv": 0.001, "ev": 1.0, "cl": 1.0, "geo": 1.0}] * 20
    fallback = [{"inv": 0.001, "ev": 1.0, "cl": 1.0, "geo": 1.0}] * 20
    normalizer = EnergyNormalizer.fit_with_fallback(primary, fallback)
    assert "inv" in normalizer.degenerate_keys
    assert normalizer.stds["inv"] == 1.0


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


def test_composite_auroc_matches_manual_linear_combination():
    records = [
        ({"inv": 0.1, "ev": 0.2, "cl": 0.3, "geo": 0.1}, 0),
        ({"inv": 2.0, "ev": 2.1, "cl": 1.9, "geo": 2.2}, 1),
        ({"inv": 0.2, "ev": 0.1, "cl": 0.2, "geo": 0.3}, 0),
        ({"inv": 1.8, "ev": 2.0, "cl": 2.1, "geo": 1.9}, 1),
    ]
    X, y = build_energy_matrix(records)
    weights = np.array([1.0, 0.0, 0.0, 0.0])  # equivalent to using "inv" alone
    fused_score = composite_auroc(X, y, weights, bias=0.0)
    assert fused_score == auroc(y, X[:, 0])
