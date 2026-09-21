"""v0 contrastive (NCE) training of a linear energy fusion.

Trains ``w, b`` in ``E(parts) = w^T parts + b`` such that
``E(positive) < E(negative)`` for each negative in
``training.negatives``, via the standard noise-contrastive objective

    L = -log( exp(-E_pos) / (exp(-E_pos) + sum_j exp(-E_neg_j)) ).

Ported from ``ebm.ipynb``. This is the v0 fusion learner referenced in
``energies.factorized_energy``; the planned upgrade replaces the linear
model ``f(e) = w^T e`` with a small nonlinear network over
``(e_i, h_i, q_i)`` with explicit pairwise interaction terms (see project
plan sections 12-15), trained with the same contrastive objective but over
targeted counterfactual negatives (``training.negatives`` TODO) instead of
claim shuffling.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def train_linear_nce(
    features_pos: np.ndarray,
    features_neg: np.ndarray,
    steps: int = 200,
    lr: float = 0.05,
    l2: float = 0.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """``features_pos``: (N, D). ``features_neg``: (N, K, D) or (N, D)
    (auto-expanded to K=1). Returns learned ``(weights, bias)``."""
    if features_pos.size == 0 or features_neg.size == 0:
        return np.zeros(features_pos.shape[1:], dtype=np.float32), 0.0
    if features_neg.ndim == 2:
        features_neg = features_neg[:, None, :]
    w = np.zeros(features_pos.shape[1], dtype=np.float32)
    b = 0.0
    N = features_pos.shape[0]
    for step in range(steps):
        grad_w = np.zeros_like(w)
        grad_b = 0.0
        total_loss = 0.0
        for i in range(N):
            f_pos = features_pos[i]
            f_negs = features_neg[i]
            E_pos = float(np.dot(w, f_pos) + b)
            E_negs = np.dot(f_negs, w) + b
            max_term = max(-E_pos, float(np.max(-E_negs)))
            exp_pos = math.exp(-E_pos - max_term)
            exp_negs = np.exp(-E_negs - max_term)
            Z = exp_pos + exp_negs.sum()
            prob_pos = exp_pos / Z
            prob_negs = exp_negs / Z
            total_loss += math.log(Z) + E_pos + max_term
            grad_w += f_pos * (1.0 - prob_pos) - (prob_negs[:, None] * f_negs).sum(axis=0)
            grad_b += (1.0 - prob_pos) - prob_negs.sum()
        total_loss /= N
        grad_w = grad_w / N + l2 * w
        grad_b /= N
        w -= lr * grad_w
        b -= lr * grad_b
        if verbose and step % max(1, steps // 10) == 0:
            print(f"step={step} nce_loss={total_loss:.4f}")
    return w, b
