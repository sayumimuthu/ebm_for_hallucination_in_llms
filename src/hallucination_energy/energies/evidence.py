"""Evidence energy: can independent evidence entail this claim?

For each claim, retrieves up to ``R`` candidate passages and scores support
via ``verifier``, then takes a soft-max (log-sum-exp) over passages so that
a single strongly-entailing passage is enough to lower the energy, while an
unsupported claim (no passage entails it) gets high energy. Ported from
``ebm.ipynb``.
"""
from __future__ import annotations

import math
from typing import Callable, List

import numpy as np

from hallucination_energy._optional import logsumexp_fallback
from hallucination_energy.claims.extraction import ClaimSet


def evidence_energy(
    claim_set: ClaimSet,
    retriever: Callable[[str, int], List[str]],
    verifier: Callable[[str, str], float],
    R: int = 5,
    tau: float = 0.2,
    verifier_temperature: float = 1.0,
) -> float:
    if claim_set is None or not claim_set.claims:
        return 0.0
    energies: List[float] = []
    temp = max(tau, 1e-6)
    calibration = max(verifier_temperature, 1e-6)
    for claim in claim_set.claims:
        docs = retriever(claim.text, top_k=R) if retriever else []
        if not docs:
            continue
        logits = np.array([verifier(claim.text, doc) for doc in docs], dtype=np.float32)
        if logits.size == 0:
            continue
        logits = logits / calibration
        lse = logsumexp_fallback(logits / temp)
        energy = -temp * (float(lse) - math.log(len(logits)))
        energies.append(energy)
    return float(np.mean(energies)) if energies else 0.0
