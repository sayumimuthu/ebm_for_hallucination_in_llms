"""Selective-prediction / risk-coverage evaluation (plan RQ1/RQ4), vendored
from the upstream SEP project's ``uncertainty/utils/eval_utils.py``.

``area_under_thresholded_accuracy`` is the AURAC metric referenced in the
project plan's minimum comparison set (section 23): abstain on the
highest-uncertainty (here: highest-energy) fraction of examples first, and
measure how accuracy on the remainder improves as you abstain more.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from hallucination_energy._optional import HAS_SCIPY


def accuracy_at_quantile(accuracies: np.ndarray, uncertainties: np.ndarray, quantile: float) -> float:
    """Mean accuracy after abstaining on the highest-uncertainty examples
    above ``quantile`` (e.g. quantile=0.5 keeps the 50% most confident)."""
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return float(np.mean(accuracies[select]))


def area_under_thresholded_accuracy(accuracies: np.ndarray, uncertainties: np.ndarray) -> float:
    """AURAC: area under the accuracy-vs-coverage curve as the abstention
    threshold sweeps from keeping the most confident 10% to keeping 100%."""
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array(
        [accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles]
    )
    dx = quantiles[1] - quantiles[0]
    return float((select_accuracies * dx).sum())


def bootstrap(function: Callable, rng, n_resamples: int = 1000) -> Callable:
    """Wrap a scalar statistic ``function(data) -> float`` into one that
    returns a bootstrap ``{std_err, low, high}`` confidence interval.
    Requires scipy."""
    if not HAS_SCIPY:
        raise ImportError("bootstrap() requires scipy (pip install hallucination-energy[full]).")
    import scipy.stats

    def inner(data):
        bs = scipy.stats.bootstrap(
            (data,), function, n_resamples=n_resamples, confidence_level=0.9, random_state=rng
        )
        return {"std_err": bs.standard_error, "low": bs.confidence_interval.low, "high": bs.confidence_interval.high}

    return inner


def compatible_bootstrap(func: Callable, rng) -> Callable:
    """Bootstrap wrapper for a 2-array metric ``func(y_true, y_score)``,
    since ``scipy.stats.bootstrap`` only resamples along a single axis."""

    def helper(y_true_y_score):
        y_true = np.array([i["y_true"] for i in y_true_y_score])
        y_score = np.array([i["y_score"] for i in y_true_y_score])
        return func(y_true, y_score)

    def wrap_inputs(y_true, y_score):
        return [{"y_true": i, "y_score": j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        return bootstrap(helper, rng=rng)(wrap_inputs(y_true, y_score))

    return converted_func
