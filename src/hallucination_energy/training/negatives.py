"""v0 negative-generation strategies: numeric jitter and claim shuffling.

.. important::
    Per the project plan (section 14), these are **naive** negatives:
    shuffling claim order or jittering a number does not reliably produce a
    *factually false but fluent* statement, so a contrastive model trained
    only on these negatives risks learning to detect linguistic corruption
    rather than factual failure.

    Targeted, minimally-edited counterfactuals per claim (entity
    substitution, relation inversion, negation, numerical/date
    perturbation) are now implemented in
    ``training.counterfactual_negatives`` — that is the generator that
    should be used for real contrastive training. This module's functions
    are kept as the "no hard negatives" ablation baseline (see
    ``scripts/compute_energy_features.py``'s ``--negative_strategy``
    flag), not as the primary strategy.

Ported from ``ebm.ipynb``.
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from hallucination_energy.claims.extraction import ClaimSet, claim_texts

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def jitter_numeric_text(text: str, scale: float = 0.1, rng: np.random.Generator = None) -> str:
    rng = rng if rng is not None else np.random.default_rng()

    def repl(match: "re.Match[str]") -> str:
        token = match.group(0)
        try:
            if "." in token:
                base = float(token)
                jitter = base * rng.uniform(-scale, scale)
                return f"{base + jitter:.3f}"
            base_int = int(token)
            delta = rng.integers(-max(1, int(scale * 10)), max(2, int(scale * 10)) + 1)
            return str(max(0, base_int + int(delta)))
        except Exception:
            return token

    return NUMBER_RE.sub(repl, text)


def corrupt_claim_set(claim_set: ClaimSet, rng: np.random.Generator = None) -> str:
    """Shuffle claim order and jitter numbers: a naive "linguistic
    corruption" negative, not a targeted factual counterfactual."""
    rng = rng if rng is not None else np.random.default_rng()
    if not claim_set.claims:
        return ""
    texts = claim_texts(claim_set)
    if not texts:
        return ""
    order = rng.permutation(len(texts))
    shuffled = [texts[i] for i in order]
    return " ".join(jitter_numeric_text(t, rng=rng) for t in shuffled)


def mix_claim_sets(claim_a: ClaimSet, claim_b: ClaimSet, limit: int = 5, rng: np.random.Generator = None) -> str:
    """Interleave claims from two unrelated answers: another naive negative."""
    rng = rng if rng is not None else np.random.default_rng()
    texts = claim_texts(claim_a)[:limit] + claim_texts(claim_b)[:limit]
    if not texts:
        return ""
    order = rng.permutation(len(texts))
    shuffled = [texts[i] for i in order]
    return " ".join(shuffled)
