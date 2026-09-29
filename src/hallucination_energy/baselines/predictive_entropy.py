"""Token-level predictive entropy / sequence log-likelihood baseline.

The simplest baseline in the project plan's comparison table (section 23):
a length-normalized negative log-likelihood of the most-likely-answer
generation, computed entirely from ``token_log_likelihoods`` already
captured during generation (see ``generation.hf_model.predict``) — no
additional sampling, clustering, or model calls needed.

Higher = more uncertain = more likely hallucinated, matching every other
score's convention in this project (``composite_energy``,
``evaluation.diagnostics.composite_auroc``, ...).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def predictive_entropy(token_log_likelihoods: List[float], length_normalize: bool = True) -> float:
    """Negative (mean, if ``length_normalize``, else summed) log-likelihood
    of a single generated sequence."""
    if not token_log_likelihoods:
        return float("nan")
    total = -float(sum(token_log_likelihoods))
    return total / len(token_log_likelihoods) if length_normalize else total


def predictive_entropy_scores(
    generations: Dict[str, Any], length_normalize: bool = True
) -> Dict[str, float]:
    """``predictive_entropy`` for the most-likely-answer of every example
    in a ``*_generations.pkl``-style dict, keyed by example id."""
    scores: Dict[str, float] = {}
    for tid, ex in generations.items():
        mla = ex.get("most_likely_answer", {})
        lls: Optional[List[float]] = mla.get("token_log_likelihoods")
        if not lls:
            continue
        scores[tid] = predictive_entropy(lls, length_normalize=length_normalize)
    return scores
