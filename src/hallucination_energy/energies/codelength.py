"""Codelength energy: how surprising was this claim to the generator itself?

Computed from the per-token log-likelihoods already captured during
generation (see ``generation.hf_model.HuggingfaceModel.predict``). Ported
from ``ebm.ipynb``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def codelength_energy(
    token_log_likelihoods: List[float],
    spans: Optional[List[Tuple[int, int]]] = None,
    stats: Optional[Dict[str, float]] = None,
    use_zscore: bool = False,
) -> float:
    """Sum of (optionally normalized) surprisal ``-log p`` over the claim's
    token span(s). ``stats`` (from ``calibrate_surprisal_stats``) lets the
    surprisal be normalized against a trusted reference distribution so
    absolute log-likelihoods are comparable across models/datasets."""
    if not token_log_likelihoods:
        return 0.0
    logs = np.asarray(token_log_likelihoods, dtype=np.float32)
    if spans:
        mask = np.zeros(len(logs), dtype=bool)
        for s, e in spans:
            s_idx = max(0, int(s))
            e_idx = min(len(logs), int(e))
            if s_idx < e_idx:
                mask[s_idx:e_idx] = True
        if mask.any():
            logs = logs[mask]
    surprisal = -logs
    mu = stats.get("mean", 0.0) if stats else 0.0
    centered = surprisal - mu
    if use_zscore:
        std = stats.get("std", 1.0) if stats else 1.0
        std = max(std, 1e-6)
        centered = centered / std
    return float(centered.sum())


def calibrate_surprisal_stats(
    validation_generations: Dict[str, Any],
    min_accuracy: float = 0.9,
) -> Dict[str, float]:
    """Fit a (mean, std) surprisal reference from high-accuracy validation
    generations, weighted by response length. Used to z-score codelength
    energy so it is comparable across examples/models."""
    means: List[float] = []
    variances: List[float] = []
    weights: List[float] = []
    for ex in validation_generations.values():
        mla = ex.get("most_likely_answer", {})
        acc = mla.get("accuracy", 0.0)
        logs = mla.get("token_log_likelihoods")
        if logs is None or acc < min_accuracy:
            continue
        arr = np.asarray(logs, dtype=np.float32)
        if arr.size == 0:
            continue
        surprisal = -arr
        means.append(float(surprisal.mean()))
        variances.append(float(np.var(surprisal)))
        weights.append(float(len(surprisal)))
    if not weights:
        return {"mean": 0.0, "std": 1.0}
    weights_arr = np.asarray(weights, dtype=np.float32)
    mean = float(np.average(np.asarray(means), weights=weights_arr))
    variance = float(np.average(np.asarray(variances), weights=weights_arr))
    return {"mean": mean, "std": float(np.sqrt(max(variance, 1e-6)))}
