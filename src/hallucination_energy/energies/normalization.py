"""Cross-energy normalization.

The four raw energies (invariance, evidence, codelength, geometry) live on
different, model/dataset-dependent scales (e.g. codelength surprisal grows
with sequence length and vocabulary size; OT distances depend on the
embedding metric). Before fusing them — whether via the v0 linear weighted
sum in ``energies.factorized_energy`` or a future learned fusion — each
factor should be normalized against its own calibration-set distribution.

This was previously done ad hoc (only for codelength, inline) in the
prototype notebook; ``EnergyNormalizer`` generalizes it to all four energies
via a single fit/transform interface, matching the "normalize this by
model/layer/task" recommendation in the project plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping

import numpy as np

ENERGY_KEYS = ("inv", "ev", "cl", "geo")


@dataclass
class EnergyNormalizer:
    """Per-factor z-score normalization fit on a calibration set.

    Usage::

        normalizer = EnergyNormalizer.fit(calibration_parts)
        normalized_parts = normalizer.transform(parts)
    """

    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, calibration_parts: Iterable[Mapping[str, float]], eps: float = 1e-6) -> "EnergyNormalizer":
        values: Dict[str, list] = {k: [] for k in ENERGY_KEYS}
        for parts in calibration_parts:
            for k in ENERGY_KEYS:
                if k in parts:
                    values[k].append(float(parts[k]))
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for k, vs in values.items():
            if not vs:
                means[k], stds[k] = 0.0, 1.0
                continue
            arr = np.asarray(vs, dtype=np.float64)
            means[k] = float(arr.mean())
            stds[k] = float(max(arr.std(), eps))
        return cls(means=means, stds=stds)

    def transform(self, parts: Mapping[str, float]) -> Dict[str, float]:
        return {
            k: (float(parts.get(k, 0.0)) - self.means.get(k, 0.0)) / self.stds.get(k, 1.0)
            for k in ENERGY_KEYS
        }
