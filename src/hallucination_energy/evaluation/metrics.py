"""Standard hallucination-detection metrics: AUROC, AUPRC, and ECE.

Uses scikit-learn when available; otherwise falls back to a dependency-free
rank-based implementation of AUROC/AUPRC so these metrics still work in a
minimal install (matching the rest of the package's optional-dependency
philosophy, see ``hallucination_energy._optional``).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from hallucination_energy._optional import HAS_SKLEARN


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve: does a higher score rank hallucinated
    examples (label=1) above factual ones (label=0)?"""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    if HAS_SKLEARN:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, s))
    # Mann-Whitney U / rank-sum formulation (no external dependency).
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (positive class = label 1)."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    if HAS_SKLEARN:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, s))
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum(np.diff(recall) * precision[1:]))


def expected_calibration_error(
    labels: Sequence[int],
    probs: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        bin_acc = y[mask].mean()
        bin_conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
