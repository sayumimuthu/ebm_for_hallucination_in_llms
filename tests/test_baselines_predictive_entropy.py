"""Tests for the predictive-entropy baseline."""
from __future__ import annotations

import math

from hallucination_energy.baselines.predictive_entropy import (
    predictive_entropy,
    predictive_entropy_scores,
)


def test_predictive_entropy_length_normalized():
    lls = [-1.0, -2.0, -3.0]
    assert predictive_entropy(lls, length_normalize=True) == 2.0


def test_predictive_entropy_sum():
    lls = [-1.0, -2.0, -3.0]
    assert predictive_entropy(lls, length_normalize=False) == 6.0


def test_predictive_entropy_empty_is_nan():
    assert math.isnan(predictive_entropy([]))


def test_predictive_entropy_scores_from_generations_dict():
    generations = {
        "a": {"most_likely_answer": {"token_log_likelihoods": [-1.0, -1.0]}},
        "b": {"most_likely_answer": {"token_log_likelihoods": [-4.0]}},
        "c": {"most_likely_answer": {}},  # missing token_log_likelihoods: skipped
    }
    scores = predictive_entropy_scores(generations)
    assert scores == {"a": 1.0, "b": 4.0}
