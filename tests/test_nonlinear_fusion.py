"""Tests for the v1 nonlinear (MLP) fusion model.

Mirrors ``training.contrastive``'s test coverage pattern: shape checks,
an empty-input guard, and a synthetic-separability check (the model
should learn to score positives lower than negatives on a trivially
separable toy problem), plus the reusable-scoring-function contract that
``evaluation.diagnostics.model_composite_auroc`` depends on.
"""
from __future__ import annotations

import numpy as np
import pytest

from hallucination_energy.evaluation.diagnostics import model_composite_auroc
from hallucination_energy.evaluation.metrics import auroc
from hallucination_energy.training.nonlinear_fusion import MLPEnergy, mlp_energy_score, train_mlp_nce


def test_mlp_energy_forward_shape():
    import torch

    model = MLPEnergy(input_dim=4, hidden_dim=8)
    x = torch.randn(10, 4)
    out = model(x)
    assert out.shape == (10,)


def test_train_mlp_nce_empty_input():
    model, loss = train_mlp_nce(np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32))
    assert loss == 0.0
    assert mlp_energy_score(model, np.zeros((3, 4), dtype=np.float32)).shape == (3,)


def test_train_mlp_nce_separates_synthetic_positives_and_negatives():
    """Positives cluster near the origin; negatives are shifted far away
    along one axis. A tiny MLP with enough steps should learn E(pos) <
    E(neg) on average, i.e. positive energy should rank lower."""
    rng = np.random.default_rng(0)
    n = 60
    pos = rng.normal(loc=0.0, scale=0.3, size=(n, 4)).astype(np.float32)
    neg = (rng.normal(loc=0.0, scale=0.3, size=(n, 1, 4)) + np.array([3.0, 3.0, 3.0, 3.0])).astype(np.float32)

    model, final_loss = train_mlp_nce(pos, neg, hidden_dim=8, steps=200, lr=0.05, l2=1e-3, seed=0)
    assert np.isfinite(final_loss)

    pos_scores = mlp_energy_score(model, pos)
    neg_scores = mlp_energy_score(model, neg.reshape(n, 4))
    assert pos_scores.mean() < neg_scores.mean()


def test_mlp_energy_score_no_grad_and_shape():
    model, _ = train_mlp_nce(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32)
    )
    scores = mlp_energy_score(model, np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32))
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_model_composite_auroc_matches_manual_score_fn():
    energy_matrix = np.array([
        [0.1, 0.2, 0.3, 0.1],
        [2.0, 2.1, 1.9, 2.2],
        [0.2, 0.1, 0.2, 0.3],
        [1.8, 2.0, 2.1, 1.9],
    ])
    labels = np.array([0, 1, 0, 1])
    score_fn = lambda x: x[:, 0]  # noqa: E731
    result = model_composite_auroc(score_fn, energy_matrix, labels)
    assert result == auroc(labels, energy_matrix[:, 0])
