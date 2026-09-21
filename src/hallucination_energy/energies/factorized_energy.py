"""Fusing the four per-claim energies into a single hallucination energy.

.. important::
    ``composite_energy``/``linear_energy`` implement the **v0 baseline**:
    a fixed (or NCE-learned) *linear* weighted sum ``E = w^T e``. Per the
    project plan, this is deliberately not the final method — a reviewer
    could describe it as "four existing uncertainty/factuality scores
    combined through a learned weighted sum." The planned upgrade is a
    conditional, nonlinear energy model with pairwise interactions,

        E_theta(c_i | x, y) = f_theta(e_i, h_i, q_i)

    that can learn signatures like "confident + evidence-contradicted =
    confident hallucination" that a linear fusion cannot represent. That
    model belongs in ``training/contrastive.py`` once implemented; this
    module remains the v0 baseline / ablation reference point.

``compute_energy_parts`` is the single entry point that computes all four
raw energies for one (question, answer) pair; everything else in this
module operates on the resulting 4-dim energy vector.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hallucination_energy.claims.extraction import ClaimSet, claim_spans, extract_claims
from hallucination_energy.energies.codelength import codelength_energy
from hallucination_energy.energies.evidence import evidence_energy
from hallucination_energy.energies.geometry import geometry_energy
from hallucination_energy.energies.invariance import invariance_energy

ENERGY_ORDER: Tuple[str, ...] = ("inv", "ev", "cl", "geo")


def composite_energy(weights: Dict[str, float], parts: Dict[str, float]) -> float:
    """v0 fixed/uniform linear fusion: E = sum_k w_k * e_k."""
    return float(sum(weights.get(k, 1.0) * parts.get(k, 0.0) for k in ENERGY_ORDER))


def energy_parts_to_vector(parts: Dict[str, float], order: Tuple[str, ...] = ENERGY_ORDER) -> np.ndarray:
    return np.array([parts.get(k, 0.0) for k in order], dtype=np.float32)


def linear_energy(weights: np.ndarray, bias: float, parts: Dict[str, float]) -> float:
    """v0 NCE-learned linear fusion (see ``training.contrastive.train_linear_nce``)."""
    vec = energy_parts_to_vector(parts)
    return float(np.dot(weights, vec) + bias)


def compute_energy_parts(
    question: str,
    answer: Dict[str, Any],
    paraphrases: List[str],
    retriever: Callable[[str, int], List[str]],
    verifier: Callable[[str, str], float],
    density_model: Any,
    density_meta: Optional[Dict[str, Any]],
    surprisal_stats: Optional[Dict[str, float]],
    sinkhorn_params: Optional[Dict[str, float]] = None,
    retriever_top_k: int = 5,
    evidence_tau: float = 0.2,
    use_claim_spans: bool = True,
) -> Tuple[Dict[str, float], ClaimSet]:
    """Compute {inv, ev, cl, geo} for one generated answer, given:

    - ``paraphrases``: alternate samples of the same prompt (invariance).
    - ``retriever``/``verifier``: external evidence (evidence).
    - ``surprisal_stats``: calibration stats from
      ``energies.codelength.calibrate_surprisal_stats`` (codelength).
    - ``density_model``/``density_meta``: from
      ``energies.geometry.fit_geometry_density`` (geometry).
    """
    sink_params = sinkhorn_params or {}
    claim_set = extract_claims(answer.get("response", ""))
    inv = invariance_energy(
        claim_set,
        paraphrases or [],
        whitener=(density_meta or {}).get("whitener"),
        reg=sink_params.get("reg", 0.1),
        tau=sink_params.get("tau", 0.5),
        quantile=sink_params.get("quantile", 1.0),
        metric=sink_params.get("metric", "cosine"),
        max_claims=sink_params.get("max_claims", 32),
    )
    ev = evidence_energy(
        claim_set,
        retriever,
        verifier,
        R=retriever_top_k,
        tau=evidence_tau,
    )
    logs = answer.get("token_log_likelihoods", [])
    spans = claim_spans(claim_set) if use_claim_spans else None
    cl = codelength_energy(logs, spans=spans, stats=surprisal_stats, use_zscore=True)
    geo = geometry_energy(claim_set, density_model, density_meta)
    parts = {"inv": inv, "ev": ev, "cl": cl, "geo": geo}
    return parts, claim_set
