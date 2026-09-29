"""Semantic Energy baseline (Ma et al., 2025).

.. important::
    This is a best-effort reimplementation of the project plan's
    specification (section 3, [1]): a logit-based Boltzmann-style energy,
    ``E(x) = -T * logsumexp(logits/T)``, combined with semantic clustering
    over resampled generations. The per-token raw-energy formula is taken
    directly from that specification and captured exactly (at ``T=1``, the
    model's natural scale — see ``generation.hf_model.predict``'s
    ``token_logsumexp`` field). The *cluster-aggregation* step (how
    per-response energies combine across a semantic cluster, and clusters
    across a question) is **this project's own design choice**, not
    independently verified against the original paper: the mean raw
    energy within each entailment cluster (reusing
    ``baselines.semantic_entropy``'s clustering), weighted by that
    cluster's probability mass. Treat this as a faithful-to-spec baseline
    for the comparison table, not a certified reproduction of the paper's
    exact numbers.

Requires ``token_logsumexp`` on every generation record — only present
for data generated *after* this baseline was added (see
``generation.hf_model.predict``); older ``*_generations.pkl`` files
predate this field and cannot be scored by this module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from hallucination_energy._optional import logsumexp_fallback
from hallucination_energy.baselines.semantic_entropy import cluster_by_entailment, cluster_logprobs


def raw_sequence_energy(token_logsumexp: Sequence[float]) -> float:
    """``E(x) = -mean_t logsumexp(logits_t)`` (T=1) for one generated
    sequence — the per-response raw Boltzmann energy before semantic
    clustering. Higher = the model's logits were collectively smaller in
    magnitude across the sequence (less confident everywhere), which the
    plan's specification treats as a lower-confidence signal."""
    if not token_logsumexp:
        return float("nan")
    return -float(sum(token_logsumexp) / len(token_logsumexp))


def semantic_energy(
    texts: Sequence[str],
    token_log_likelihoods: Sequence[Sequence[float]],
    token_logsumexps: Sequence[Sequence[float]],
    context_prefix: str = "",
) -> float:
    """Semantic Energy over ``texts`` (resampled answers to the same
    question): cluster by bidirectional entailment (same mechanism as
    ``baselines.semantic_entropy``), then take the probability-weighted
    average of each cluster's mean raw sequence energy. ``nan`` if given
    no samples or no logsumexp data.
    """
    if not texts or not all(token_logsumexps):
        return float("nan")
    cids = cluster_by_entailment(texts, context_prefix=context_prefix)
    unique_clusters, logprobs = cluster_logprobs(cids, token_log_likelihoods)
    total = logsumexp_fallback(logprobs) if logprobs.size > 1 else logprobs[0]
    weights = np.exp(logprobs - total)

    raw_energies = np.array([raw_sequence_energy(tls) for tls in token_logsumexps])
    cluster_mean_energy = np.array([
        raw_energies[[i for i, cid in enumerate(cids) if cid == c]].mean() for c in unique_clusters
    ])
    return float(np.sum(weights * cluster_mean_energy))


def semantic_energy_scores(generations: Dict[str, Any]) -> Dict[str, float]:
    """``semantic_energy`` for every example in a ``*_generations.pkl``-style
    dict that has ``token_logsumexp`` populated, keyed by example id.
    Examples generated before this field existed are silently skipped
    (not scored as ``nan`` — omitted from the returned dict entirely, so
    callers can distinguish "not available" from "computed as NaN")."""
    scores: Dict[str, float] = {}
    for tid, ex in generations.items():
        mla = ex.get("most_likely_answer", {})
        mla_text = mla.get("response")
        mla_lls = mla.get("token_log_likelihoods")
        mla_lse = mla.get("token_logsumexp")
        if not mla_text or not mla_lls or not mla_lse:
            continue
        texts = [mla_text]
        lls_list: List[List[float]] = [mla_lls]
        lse_list: List[List[float]] = [mla_lse]
        for r in ex.get("responses", []):
            if isinstance(r, (list, tuple)) and len(r) >= 5 and r[0] and r[1] and r[4]:
                texts.append(r[0])
                lls_list.append(r[1])
                lse_list.append(r[4])
        question = ex.get("question", "")
        prefix = f"{question} " if question else ""
        scores[tid] = semantic_energy(texts, lls_list, lse_list, context_prefix=prefix)
    return scores
