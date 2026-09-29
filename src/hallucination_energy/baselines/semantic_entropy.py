"""Semantic Entropy baseline (Kuhn et al., 2023; Farquhar et al., 2024).

Clusters the resampled answers to a question by bidirectional NLI
entailment (two answers are the same semantic cluster iff each entails
the other), then computes the Shannon entropy of the resulting
cluster-probability distribution — where each cluster's probability mass
is the (length-normalized) sequence likelihood of its members, summed via
logsumexp, not just a cluster-size count. Higher entropy = the model's
resampled answers disagree about *meaning*, not just surface wording =
more uncertain.

Reuses ``retrieval.entailment.EntailmentDeberta`` (already vendored for
``retrieval.verifier.default_verifier``) via the same optional-dependency
fallback pattern used throughout this project: with the NLI model,
cluster by mutual entailment; without it, fall back to an embedding
cosine-similarity threshold (a coarser proxy for semantic equivalence).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hallucination_energy._optional import get_entailment_model, logsumexp_fallback
from hallucination_energy.energies.embeddings import embed_texts

_EMBEDDING_FALLBACK_THRESHOLD = 0.9  # cosine similarity; coarse, only used without the NLI model


def _same_cluster(text_a: str, text_b: str, model: Optional[Any]) -> bool:
    if model is not None:
        try:
            imp_ab = model.check_implication(text_a, text_b)
            imp_ba = model.check_implication(text_b, text_a)
            return imp_ab == 2 and imp_ba == 2  # both directions entail -> semantically equivalent
        except Exception:
            pass
    embs, _ = embed_texts([text_a, text_b], metric="cosine")
    if embs.shape[0] < 2:
        return False
    return float(embs[0] @ embs[1]) >= _EMBEDDING_FALLBACK_THRESHOLD


def cluster_by_entailment(texts: Sequence[str], context_prefix: str = "") -> List[int]:
    """Greedy bidirectional-entailment clustering: each text joins the
    first existing cluster it mutually entails with, else starts a new
    one. Returns a list of cluster ids, same length/order as ``texts``.
    ``context_prefix`` (e.g. ``f"{question} "``) is prepended to both
    texts of every comparison, matching the upstream SEP convention of
    judging entailment in the context of the question rather than the
    bare answer strings alone.
    """
    if not texts:
        return []
    model = get_entailment_model()
    n = len(texts)
    cluster_ids = [-1] * n
    next_id = 0
    prefixed = [context_prefix + t for t in texts]
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, n):
            if cluster_ids[j] != -1:
                continue
            if _same_cluster(prefixed[i], prefixed[j], model):
                cluster_ids[j] = cluster_ids[i]
        next_id += 1
    return cluster_ids


def _sequence_logprob(token_log_likelihoods: Sequence[float]) -> float:
    """Length-normalized log-probability of one generated sequence."""
    if not token_log_likelihoods:
        return float("-inf")
    return float(sum(token_log_likelihoods) / len(token_log_likelihoods))


def cluster_logprobs(
    cluster_ids: Sequence[int], token_log_likelihoods: Sequence[Sequence[float]]
) -> Tuple[List[int], np.ndarray]:
    """Per-cluster log-probability mass: logsumexp of member sequences'
    length-normalized log-probabilities. Returns ``(unique_cluster_ids,
    cluster_logprob_array)``, aligned."""
    seq_logprobs = np.array([_sequence_logprob(lls) for lls in token_log_likelihoods])
    unique_clusters = sorted(set(cluster_ids))
    out = np.empty(len(unique_clusters), dtype=np.float64)
    for k, c in enumerate(unique_clusters):
        members = seq_logprobs[[i for i, cid in enumerate(cluster_ids) if cid == c]]
        out[k] = logsumexp_fallback(members) if members.size > 1 else members[0]
    return unique_clusters, out


def semantic_entropy(
    texts: Sequence[str],
    token_log_likelihoods: Sequence[Sequence[float]],
    context_prefix: str = "",
) -> float:
    """Shannon entropy (nats) of the cluster-probability distribution over
    ``texts`` (resampled answers to the same question). ``nan`` if given
    no samples."""
    if not texts:
        return float("nan")
    cids = cluster_by_entailment(texts, context_prefix=context_prefix)
    _, logprobs = cluster_logprobs(cids, token_log_likelihoods)
    total = logsumexp_fallback(logprobs) if logprobs.size > 1 else logprobs[0]
    log_probs_normalized = logprobs - total
    probs = np.exp(log_probs_normalized)
    return float(-np.sum(probs * log_probs_normalized))


def semantic_entropy_scores(generations: Dict[str, Any]) -> Dict[str, float]:
    """``semantic_entropy`` for every example in a ``*_generations.pkl``-style
    dict, keyed by example id. Uses the most-likely-answer plus every
    entry of ``responses`` (the ``num_generations`` high-temperature
    resamples already produced during generation — see
    ``generation.generate``) as the resample set.
    """
    scores: Dict[str, float] = {}
    for tid, ex in generations.items():
        mla = ex.get("most_likely_answer", {})
        mla_text = mla.get("response")
        mla_lls = mla.get("token_log_likelihoods")
        if not mla_text or not mla_lls:
            continue
        texts = [mla_text]
        lls_list: List[List[float]] = [mla_lls]
        for r in ex.get("responses", []):
            if isinstance(r, (list, tuple)) and len(r) >= 2 and r[0] and r[1]:
                texts.append(r[0])
                lls_list.append(r[1])
        question = ex.get("question", "")
        prefix = f"{question} " if question else ""
        scores[tid] = semantic_entropy(texts, lls_list, context_prefix=prefix)
    return scores
