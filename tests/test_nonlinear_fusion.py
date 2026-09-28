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
from hallucination_energy.training.nonlinear_fusion import (
    MLPEnergy,
    ResidualMLPEnergy,
    mlp_energy_score,
    train_mlp_nce,
    train_residual_mlp_nce,
)


def test_mlp_energy_forward_shape():
    import torch

    model = MLPEnergy(input_dim=4, hidden_dim=8)
    x = torch.randn(10, 4)
    out = model(x)
    assert out.shape == (10,)


def test_train_mlp_nce_empty_input():
    model, loss, best_score = train_mlp_nce(np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32))
    assert loss == 0.0
    assert best_score is None
    assert mlp_energy_score(model, np.zeros((3, 4), dtype=np.float32)).shape == (3,)


def test_train_mlp_nce_separates_synthetic_positives_and_negatives():
    """Positives cluster near the origin; negatives are shifted far away
    along one axis. A tiny MLP with enough steps should learn E(pos) <
    E(neg) on average, i.e. positive energy should rank lower."""
    rng = np.random.default_rng(0)
    n = 60
    pos = rng.normal(loc=0.0, scale=0.3, size=(n, 4)).astype(np.float32)
    neg = (rng.normal(loc=0.0, scale=0.3, size=(n, 1, 4)) + np.array([3.0, 3.0, 3.0, 3.0])).astype(np.float32)

    model, final_loss, best_score = train_mlp_nce(pos, neg, hidden_dim=8, steps=200, lr=0.05, l2=1e-3, seed=0)
    assert np.isfinite(final_loss)
    assert best_score is None  # no monitor_fn given

    pos_scores = mlp_energy_score(model, pos)
    neg_scores = mlp_energy_score(model, neg.reshape(n, 4))
    assert pos_scores.mean() < neg_scores.mean()


def test_mlp_energy_score_no_grad_and_shape():
    model, _, _ = train_mlp_nce(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32)
    )
    scores = mlp_energy_score(model, np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32))
    assert scores.shape == (5,)
    assert scores.dtype == np.float32


def test_residual_mlp_energy_starts_as_pure_linear():
    """At initialization (before any training), ResidualMLPEnergy must be
    EXACTLY the zero linear model: linear weights/bias zero and the
    residual branch's final layer zero -- the whole point of the
    architecture is starting at the already-validated linear solution."""
    import torch

    model = ResidualMLPEnergy(input_dim=4, hidden_dim=8)
    x = torch.randn(6, 4)
    out = model(x)
    assert torch.allclose(out, torch.zeros(6))


def test_residual_mlp_energy_forward_shape():
    import torch

    model = ResidualMLPEnergy(input_dim=4, hidden_dim=8)
    x = torch.randn(10, 4)
    assert model(x).shape == (10,)


def test_train_residual_mlp_nce_empty_input():
    model, loss, best_score = train_residual_mlp_nce(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32)
    )
    assert loss == 0.0
    assert best_score is None
    assert mlp_energy_score(model, np.zeros((3, 4), dtype=np.float32)).shape == (3,)


def test_train_residual_mlp_nce_separates_synthetic_positives_and_negatives():
    rng = np.random.default_rng(0)
    n = 60
    pos = rng.normal(loc=0.0, scale=0.3, size=(n, 4)).astype(np.float32)
    neg = (rng.normal(loc=0.0, scale=0.3, size=(n, 1, 4)) + np.array([3.0, 3.0, 3.0, 3.0])).astype(np.float32)

    model, final_loss, best_score = train_residual_mlp_nce(pos, neg, hidden_dim=8, steps=200, lr=0.05, l2=1e-3, seed=0)
    assert np.isfinite(final_loss)
    assert best_score is None

    pos_scores = mlp_energy_score(model, pos)
    neg_scores = mlp_energy_score(model, neg.reshape(n, 4))
    assert pos_scores.mean() < neg_scores.mean()


def test_residual_mlp_linear_component_shape():
    model, _, _ = train_residual_mlp_nce(
        np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2, 4), dtype=np.float32)
    )
    w, b = model.linear_component()
    assert w.shape == (4,)
    assert isinstance(b, float)


def test_train_residual_mlp_nce_monitor_fn_keeps_best_checkpoint():
    """The core new behavior: if training drifts to a worse point than an
    earlier checkpoint (as measured by monitor_fn), the returned model
    must be the best-scoring snapshot, not the last-step one."""
    rng = np.random.default_rng(0)
    n = 40
    pos = rng.normal(loc=0.0, scale=0.3, size=(n, 4)).astype(np.float32)
    neg = (rng.normal(loc=0.0, scale=0.3, size=(n, 1, 4)) + np.array([3.0, 3.0, 3.0, 3.0])).astype(np.float32)
    # A labeled evaluation set unrelated in size/order to the NCE pairs above --
    # monitor_fn only needs to accept a model and return a float score.
    eval_matrix = rng.normal(size=(30, 4)).astype(np.float32)
    eval_labels = rng.integers(0, 2, size=30)

    calls = []

    def monitor_fn(model):
        score = model_composite_auroc(lambda x: mlp_energy_score(model, x), eval_matrix, eval_labels)
        calls.append(score)
        return score

    model, final_loss, best_score = train_residual_mlp_nce(
        pos, neg, hidden_dim=8, steps=50, lr=0.05, l2=1e-3, seed=0,
        monitor_fn=monitor_fn, monitor_every=5,
    )
    assert len(calls) >= 2  # monitored more than once (steps=50, every=5, plus final step)
    assert best_score == max(calls)
    # The model actually returned must reproduce best_score, not whatever
    # the final training step happened to land on.
    final_score = model_composite_auroc(lambda x: mlp_energy_score(model, x), eval_matrix, eval_labels)
    assert final_score == best_score


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
