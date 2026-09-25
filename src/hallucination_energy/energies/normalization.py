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
from typing import Dict, Iterable, List, Mapping, Set

import numpy as np

ENERGY_KEYS = ("inv", "ev", "cl", "geo")


def _collect_values(calibration_parts: Iterable[Mapping[str, float]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {k: [] for k in ENERGY_KEYS}
    for parts in calibration_parts:
        for k in ENERGY_KEYS:
            if k in parts:
                values[k].append(float(parts[k]))
    return values


@dataclass
class EnergyNormalizer:
    """Per-factor z-score normalization fit on a calibration set.

    Usage::

        normalizer = EnergyNormalizer.fit(calibration_parts)
        normalized_parts = normalizer.transform(parts)

    ``degenerate_keys`` records which factors had a near-zero calibration
    std (see ``fit``) and were left unscaled rather than normalized —
    useful for callers that want to patch those factors from a fallback
    calibration set (see ``fit_with_fallback``).
    """

    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    degenerate_keys: Set[str] = field(default_factory=set)

    @classmethod
    def fit(cls, calibration_parts: Iterable[Mapping[str, float]], eps: float = 1e-6) -> "EnergyNormalizer":
        values = _collect_values(calibration_parts)
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        degenerate: Set[str] = set()
        for k, vs in values.items():
            if not vs:
                means[k], stds[k] = 0.0, 1.0
                degenerate.add(k)
                continue
            arr = np.asarray(vs, dtype=np.float64)
            means[k] = float(arr.mean())
            raw_std = float(arr.std())
            # A near-zero calibration-set std (e.g. a factor that happens to
            # be ~constant on the reference split — observed in practice for
            # invariance energy when the reference split lacks paraphrase
            # samples) must NOT fall back to dividing by `eps`: that turns
            # tiny, likely-noise differences into an enormous z-score,
            # making the factor look far more discriminative than it is and
            # dominating any downstream learned fusion by numerical artifact
            # rather than by genuine signal. Instead, treat it as "no
            # reliable variance estimate" and leave the factor unscaled
            # (std=1: still centered, just not blown up) — and record it as
            # degenerate so a caller with a fallback calibration set (e.g.
            # ``fit_with_fallback``) can patch it with a real estimate.
            if raw_std >= eps:
                stds[k] = raw_std
            else:
                stds[k] = 1.0
                degenerate.add(k)
        return cls(means=means, stds=stds, degenerate_keys=degenerate)

    @classmethod
    def fit_with_fallback(
        cls,
        calibration_parts: Iterable[Mapping[str, float]],
        fallback_parts: Iterable[Mapping[str, float]],
        eps: float = 1e-6,
    ) -> "EnergyNormalizer":
        """Fit on ``calibration_parts`` (the preferred, e.g. reference-split,
        calibration set); for any factor that comes out degenerate there
        (see ``fit``), refit just that factor's mean/std from
        ``fallback_parts`` instead of leaving it unscaled at std=1 — being
        left at std=1 doesn't just fail to help that factor, it silences it
        relative to every properly-scaled peer in any equal- or
        learned-weight linear combination, which is its own distortion.

        If a factor is *also* degenerate in ``fallback_parts``, it keeps
        the unscaled (std=1) fallback from ``fit``.
        """
        primary = cls.fit(calibration_parts, eps=eps)
        if not primary.degenerate_keys:
            return primary
        fallback_values = _collect_values(fallback_parts)
        still_degenerate = set(primary.degenerate_keys)
        for k in list(primary.degenerate_keys):
            vs = fallback_values.get(k, [])
            if not vs:
                continue
            arr = np.asarray(vs, dtype=np.float64)
            raw_std = float(arr.std())
            if raw_std >= eps:
                primary.means[k] = float(arr.mean())
                primary.stds[k] = raw_std
                still_degenerate.discard(k)
        primary.degenerate_keys = still_degenerate
        return primary

    def transform(self, parts: Mapping[str, float]) -> Dict[str, float]:
        return {
            k: (float(parts.get(k, 0.0)) - self.means.get(k, 0.0)) / self.stds.get(k, 1.0)
            for k in ENERGY_KEYS
        }
