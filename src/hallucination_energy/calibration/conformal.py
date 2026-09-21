"""Split-conformal calibration of the fused hallucination energy.

Given calibration-set energies ``E_1, ..., E_n`` (from held-out examples not
used to fit the energy model), ``conformal_threshold`` picks the
``1 - alpha`` quantile as a decision threshold, and ``conformal_pvalue``
gives an exact finite-sample conformal p-value for a new energy. Flag
``E_test > threshold`` (equivalently ``p_value < alpha``) as a suspected
hallucination.

.. note::
    **Deliberately dropped from the original prototype**: a
    ``convert_p_to_e(p) = 1/p`` function that treated the inverse of a
    conformal p-value as a valid e-value, plus an ``e_process_mixture``
    sequential e-process and an ``safe_merge_e_values`` combiner built on
    top of it. Treating ``1/p`` as an e-value is not generally justified —
    a conformal p-value is (super-)uniform under the null, but ``1/p`` is
    not guaranteed to have expectation <= 1 under the null in this setting,
    which is the defining property an e-value must have. Use the quantile
    threshold below as the v1 decision rule; only reintroduce a sequential
    e-process if you construct (or find) an e-calibrator with a real
    validity proof for this setting.
"""
from __future__ import annotations

from typing import List

import numpy as np


def conformal_threshold(energies: List[float], alpha: float = 0.05) -> float:
    """The ``1 - alpha`` quantile of calibration-set energies."""
    if not energies:
        return float("inf")
    arr = np.asarray(energies, dtype=np.float32)
    if hasattr(np, "quantile"):
        q = np.quantile(arr, 1.0 - alpha, method="higher")
    else:  # pragma: no cover - extremely old numpy
        q = np.sort(arr)[int(np.ceil((1 - alpha) * len(arr))) - 1]
    return float(q)


def conformal_pvalue(calib_energies: List[float], test_energy: float) -> float:
    """Exact finite-sample conformal p-value: the fraction of calibration
    energies at least as extreme as ``test_energy`` (Vovk-style, with the
    +1/+1 correction so the p-value is valid even for small calibration
    sets)."""
    arr = np.asarray(calib_energies, dtype=np.float64)
    n = len(arr)
    ge = int((arr >= test_energy).sum())
    return (1 + ge) / (n + 1)
