"""The recommended first empirical milestone (see README roadmap): before
building a sophisticated fusion model, check whether the four energies
actually have *complementary* error patterns on labeled data.

If ``energy_correlation_matrix`` shows all four energies pairwise
correlated at r ~= 0.95, they are redundant and a factorized/nonlinear
fusion is unlikely to help much over a single score. If they disagree in
different regimes (e.g. per ``per_factor_auroc`` below), that is empirical
justification for the factorized approach.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from hallucination_energy.energies.factorized_energy import ENERGY_ORDER, energy_parts_to_vector


def build_energy_matrix(
    records: Iterable[Tuple[Mapping[str, float], int]],
    order: Sequence[str] = ENERGY_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """``records``: iterable of ``(energy_parts_dict, label)`` where label is
    1 for hallucinated / 0 for factual. Returns ``(N x len(order))`` energy
    matrix and ``(N,)`` label vector."""
    parts_list: List[np.ndarray] = []
    labels: List[int] = []
    for parts, label in records:
        parts_list.append(energy_parts_to_vector(dict(parts), order=tuple(order)))
        labels.append(int(label))
    if not parts_list:
        return np.zeros((0, len(order)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(parts_list), np.asarray(labels, dtype=np.int64)


def energy_correlation_matrix(energy_matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix between energy factors (columns)."""
    if energy_matrix.shape[0] < 2:
        return np.eye(energy_matrix.shape[1], dtype=np.float32)
    return np.corrcoef(energy_matrix, rowvar=False)


def per_factor_auroc(
    energy_matrix: np.ndarray,
    labels: np.ndarray,
    order: Sequence[str] = ENERGY_ORDER,
) -> Dict[str, float]:
    """AUROC of each individual energy factor as a standalone hallucination
    score, to see which failure modes each factor alone catches."""
    from hallucination_energy.evaluation.metrics import auroc

    return {name: auroc(labels, energy_matrix[:, i]) for i, name in enumerate(order)}
