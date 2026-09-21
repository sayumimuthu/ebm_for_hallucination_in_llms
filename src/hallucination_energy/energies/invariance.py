"""Invariance energy: does the claim set survive meaning-preserving
perturbations (paraphrase / resample)?

Implemented as an unbalanced-optimal-transport (Sinkhorn) distance between
the embedded claims of the original answer and the embedded claims of each
paraphrase/resample, so that added/dropped claims are handled gracefully
(unlike a naive one-to-one claim alignment). Ported from ``ebm.ipynb``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from hallucination_energy.claims.extraction import ClaimSet, claim_texts, extract_claims
from hallucination_energy.energies.embeddings import embed_texts_pair, pairwise_cost


def sinkhorn_unbalanced(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.1,
    tau_src: float = 0.5,
    tau_tgt: float = 0.5,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> float:
    """Unbalanced entropic-OT cost between two claim mass distributions.

    ``tau_src``/``tau_tgt`` relax the marginal constraints (KL penalty
    instead of hard equality), which is what lets the transport plan ignore
    claims that were added or dropped between the two claim sets instead of
    being forced to match them at high cost.
    """
    if C.size == 0:
        return 0.0
    reg = max(reg, 1e-3)
    K = np.exp(-C / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)
    pow_a = 1.0 if tau_src is None or math.isinf(tau_src) else tau_src / (tau_src + reg)
    pow_b = 1.0 if tau_tgt is None or math.isinf(tau_tgt) else tau_tgt / (tau_tgt + reg)
    for _ in range(num_iters):
        u_prev = u
        Kv = K @ v + 1e-12
        u = np.power(np.clip(a / Kv, 1e-12, None), pow_a)
        KTu = K.T @ u + 1e-12
        v = np.power(np.clip(b / KTu, 1e-12, None), pow_b)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    T_plan = np.outer(u, v) * K
    return float((T_plan * C).sum())


def invariance_energy(
    answer_claims: ClaimSet,
    paraphrase_texts: List[str],
    whitener: Optional[Dict[str, np.ndarray]] = None,
    reg: float = 0.1,
    tau: float = 0.5,
    quantile: float = 1.0,
    metric: str = "cosine",
    max_claims: int = 32,
) -> float:
    """Aggregate unbalanced-OT distance to each paraphrase's claim set.

    ``quantile`` controls the aggregation over paraphrases: 1.0 (default)
    takes the worst-case (max) distance, matching the intuition that a claim
    only needs to break under *one* meaning-preserving perturbation to be
    suspicious.
    """
    if answer_claims is None or not answer_claims.claims:
        return 0.0
    base_texts = claim_texts(answer_claims)[:max_claims]
    if not base_texts:
        return 0.0
    distances: List[float] = []
    for paraphrase in paraphrase_texts:
        paraphrase_claims = extract_claims(paraphrase)
        if not paraphrase_claims.claims:
            continue
        para_texts = claim_texts(paraphrase_claims)[:max_claims]
        # Embed the base and paraphrase claims *jointly* (embed_texts_pair),
        # not via two independent embed_texts() calls: under the TF-IDF
        # fallback (no sentence-transformers installed), two independent
        # calls fit two different vocabularies and land in incompatible
        # feature spaces, silently corrupting (or crashing) the pairwise
        # cost matrix below. See embeddings.embed_texts_with_vectorizer for
        # the same fix applied to the geometry energy.
        base_embs, para_embs, metric_used = embed_texts_pair(
            base_texts, para_texts, metric=metric, whitener=whitener
        )
        if base_embs.size == 0 or para_embs.size == 0:
            continue
        a = np.ones(base_embs.shape[0], dtype=np.float32)
        a /= max(a.sum(), 1.0)
        b = np.ones(para_embs.shape[0], dtype=np.float32)
        b /= max(b.sum(), 1.0)
        C = pairwise_cost(base_embs, para_embs, metric=metric_used)
        dist = sinkhorn_unbalanced(C, a, b, reg=reg, tau_src=tau, tau_tgt=tau)
        distances.append(dist)
    if not distances:
        return 0.0
    distances_arr = np.asarray(distances, dtype=np.float32)
    if quantile >= 1.0:
        return float(distances_arr.max())
    q = np.clip(quantile, 0.0, 1.0)
    return float(np.quantile(distances_arr, q))
