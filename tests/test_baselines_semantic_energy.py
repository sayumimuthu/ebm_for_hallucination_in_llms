"""Tests for the Semantic Energy baseline (also run against the embedding
fallback path, same as test_baselines_semantic_entropy.py)."""
from __future__ import annotations

import math

import pytest

from hallucination_energy.baselines.semantic_energy import (
    raw_sequence_energy,
    semantic_energy,
    semantic_energy_scores,
)


@pytest.fixture(autouse=True)
def _force_embedding_fallback(monkeypatch):
    monkeypatch.setattr("hallucination_energy.baselines.semantic_entropy.get_entailment_model", lambda: None)


def test_raw_sequence_energy():
    # E(x) = -mean(token_logsumexp)
    assert raw_sequence_energy([10.0, 20.0]) == -15.0


def test_raw_sequence_energy_empty_is_nan():
    assert math.isnan(raw_sequence_energy([]))


def test_semantic_energy_single_cluster_matches_mean_energy():
    """With one semantic cluster, the weighted average collapses to the
    plain mean of the cluster's raw sequence energies."""
    texts = ["Paris is the capital of France."] * 2
    lls = [[-0.1, -0.1], [-0.1, -0.1]]
    lse = [[10.0, 10.0], [20.0, 20.0]]
    result = semantic_energy(texts, lls, lse)
    assert result == pytest.approx(-15.0)  # mean(-10, -20)


def test_semantic_energy_empty_is_nan():
    assert math.isnan(semantic_energy([], [], []))


def test_semantic_energy_scores_skips_examples_without_logsumexp():
    generations = {
        "a": {
            "question": "Q?",
            "most_likely_answer": {
                "response": "A.", "token_log_likelihoods": [-0.1], "token_logsumexp": [10.0],
            },
            "responses": [],
        },
        "b": {
            # predates the token_logsumexp field -- must be skipped, not scored as NaN.
            "question": "Q2?",
            "most_likely_answer": {"response": "B.", "token_log_likelihoods": [-0.1]},
            "responses": [],
        },
    }
    scores = semantic_energy_scores(generations)
    assert "a" in scores
    assert "b" not in scores
