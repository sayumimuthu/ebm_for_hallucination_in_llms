"""Standard hallucination-detection metrics: AUROC, AUPRC, and ECE.

Uses scikit-learn when available; otherwise falls back to a dependency-free
rank-based implementation of AUROC/AUPRC so these metrics still work in a
minimal install (matching the rest of the package's optional-dependency
philosophy, see ``hallucination_energy._optional``).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

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


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = auroc,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for ``metric_fn(labels,
    scores)`` (default: AUROC). Motivated by a real finding: rerunning the
    same pipeline on a freshly-regenerated dataset flipped whether linear
    fusion beat the best single energy factor, with no way to tell from a
    single point estimate whether either result was more than noise at
    n~200. Resamples ``(label, score)`` pairs WITH replacement
    ``n_bootstrap`` times; resamples with only one class present (AUROC
    undefined) are skipped rather than counted.

    Returns ``{"point", "ci_low", "ci_high", "std", "n_bootstrap_valid"}``.
    """
    y = np.asarray(labels)
    s = np.asarray(scores)
    n = len(y)
    point = metric_fn(y, s)
    rng = np.random.default_rng(seed)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_scores.append(metric_fn(y_b, s[idx]))
    boot_arr = np.asarray(boot_scores, dtype=np.float64)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    if boot_arr.size == 0:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "std": float("nan"), "n_bootstrap_valid": 0}
    return {
        "point": float(point),
        "ci_low": float(np.percentile(boot_arr, lo_pct)),
        "ci_high": float(np.percentile(boot_arr, hi_pct)),
        "std": float(boot_arr.std()),
        "n_bootstrap_valid": int(boot_arr.size),
    }


def bootstrap_auroc_diff_ci(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Paired bootstrap CI for ``AUROC(scores_a) - AUROC(scores_b)`` on the
    SAME resampled examples each iteration — e.g. two fusion strategies
    scored against the same labels. This is more informative than
    comparing two separate marginal ``bootstrap_ci`` intervals, since it
    accounts for the correlation between ``scores_a`` and ``scores_b``
    (both computed on the same underlying examples): two marginal CIs can
    overlap even when the paired difference is consistently one-signed.

    ``significant`` is true iff the CI excludes 0 (whole interval above or
    below zero) at the given confidence level.

    Returns ``{"point", "ci_low", "ci_high", "significant", "n_bootstrap_valid"}``.
    """
    y = np.asarray(labels)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    n = len(y)
    point = auroc(y, a) - auroc(y, b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        diffs.append(auroc(y_b, a[idx]) - auroc(y_b, b[idx]))
    diff_arr = np.asarray(diffs, dtype=np.float64)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    if diff_arr.size == 0:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "significant": False, "n_bootstrap_valid": 0}
    ci_low = float(np.percentile(diff_arr, lo_pct))
    ci_high = float(np.percentile(diff_arr, hi_pct))
    return {
        "point": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(ci_low > 0 or ci_high < 0),
        "n_bootstrap_valid": int(diff_arr.size),
    }


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
