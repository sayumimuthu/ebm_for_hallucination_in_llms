"""Tests for evaluation.metrics, including the bootstrap CI utilities.

Motivated by a real finding: rerunning compute_energy_features.py on a
freshly-regenerated dataset flipped whether linear fusion beat the best
single energy factor (a positive gap on one dataset draw, a negative gap
on another), with no way to tell from a single AUROC point estimate
whether either result was more than noise at n~200. bootstrap_ci and
bootstrap_auroc_diff_ci exist to answer exactly that.
"""
from __future__ import annotations

import numpy as np

from hallucination_energy.evaluation.metrics import auroc, bootstrap_auroc_diff_ci, bootstrap_ci


def test_bootstrap_ci_point_matches_plain_auroc():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)])
    result = bootstrap_ci(labels, scores, n_bootstrap=200, seed=0)
    assert result["point"] == auroc(labels, scores)
    assert result["n_bootstrap_valid"] > 0
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_bootstrap_ci_wide_for_small_perfectly_separable_sample():
    """A tiny, perfectly-separable sample should still report a point
    estimate of 1.0, but the CI shouldn't be a degenerate point unless
    every bootstrap resample also stays perfectly separable."""
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    result = bootstrap_ci(labels, scores, n_bootstrap=500, seed=0)
    assert result["point"] == 1.0
    assert result["ci_low"] <= 1.0


def test_bootstrap_ci_handles_degenerate_resamples_gracefully():
    """With very few examples of the minority class, some bootstrap
    resamples will be single-class (AUROC undefined) -- these must be
    skipped, not crash or corrupt the CI."""
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.arange(10, dtype=np.float64)
    result = bootstrap_ci(labels, scores, n_bootstrap=300, seed=0)
    assert result["n_bootstrap_valid"] <= 300
    assert result["n_bootstrap_valid"] > 0
    assert np.isfinite(result["point"])


def test_bootstrap_auroc_diff_ci_detects_a_real_large_difference():
    """scores_a perfectly separates the classes; scores_b is pure noise.
    The true difference is large and should not include 0."""
    rng = np.random.default_rng(0)
    n = 100
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores_a = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])  # perfect separator
    scores_b = rng.normal(size=n)  # noise, ~0.5 AUROC

    result = bootstrap_auroc_diff_ci(labels, scores_a, scores_b, n_bootstrap=300, seed=0)
    assert result["point"] > 0.3
    assert result["significant"] is True
    assert result["ci_low"] > 0


def test_bootstrap_auroc_diff_ci_no_difference_when_scores_identical():
    rng = np.random.default_rng(0)
    n = 60
    labels = rng.integers(0, 2, size=n)
    scores = rng.normal(size=n)

    result = bootstrap_auroc_diff_ci(labels, scores, scores, n_bootstrap=200, seed=0)
    assert result["point"] == 0.0
    assert result["significant"] is False
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_bootstrap_auroc_diff_ci_matches_manual_point_difference():
    rng = np.random.default_rng(1)
    n = 80
    labels = rng.integers(0, 2, size=n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    result = bootstrap_auroc_diff_ci(labels, a, b, n_bootstrap=100, seed=0)
    assert result["point"] == auroc(labels, a) - auroc(labels, b)
