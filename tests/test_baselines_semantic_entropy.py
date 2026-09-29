"""Tests for the Semantic Entropy baseline.

Run against the embedding-cosine-similarity fallback path (spaCy-style
optional-dependency pattern): the DeBERTa entailment model is forced to
None, matching this dev environment (no NLI model download) and this
project's established dependency-free testing convention.
"""
from __future__ import annotations

import math

import pytest

from hallucination_energy.baselines.semantic_entropy import (
    cluster_by_entailment,
    cluster_logprobs,
    semantic_entropy,
    semantic_entropy_scores,
)


@pytest.fixture(autouse=True)
def _force_embedding_fallback(monkeypatch):
    monkeypatch.setattr("hallucination_energy.baselines.semantic_entropy.get_entailment_model", lambda: None)


def test_cluster_by_entailment_empty():
    assert cluster_by_entailment([]) == []


def test_cluster_by_entailment_groups_near_duplicate_texts():
    texts = ["Paris is the capital of France.", "Paris is the capital of France."]
    cids = cluster_by_entailment(texts)
    assert cids[0] == cids[1]


def test_cluster_by_entailment_separates_dissimilar_texts():
    texts = ["Paris is the capital of France.", "Quantum entanglement violates locality in Bell tests."]
    cids = cluster_by_entailment(texts)
    assert cids[0] != cids[1]


def test_cluster_logprobs_single_cluster():
    cids = [0, 0]
    lls = [[-0.1, -0.1], [-0.2, -0.2]]
    unique, logprobs = cluster_logprobs(cids, lls)
    assert unique == [0]
    assert logprobs.shape == (1,)


def test_semantic_entropy_zero_when_all_answers_agree():
    """All resampled answers land in one semantic cluster -> zero entropy
    (no disagreement about meaning)."""
    texts = ["Paris is the capital of France."] * 4
    lls = [[-0.1, -0.1]] * 4
    result = semantic_entropy(texts, lls)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_semantic_entropy_positive_when_answers_disagree():
    texts = [
        "Paris is the capital of France.",
        "Paris is the capital of France.",
        "Quantum entanglement violates locality in Bell tests.",
    ]
    lls = [[-0.1, -0.1], [-0.1, -0.1], [-0.1, -0.1]]
    result = semantic_entropy(texts, lls)
    assert result > 0.0


def test_semantic_entropy_empty_is_nan():
    assert math.isnan(semantic_entropy([], []))


def test_semantic_entropy_scores_from_generations_dict():
    generations = {
        "a": {
            "question": "What is the capital of France?",
            "most_likely_answer": {"response": "Paris.", "token_log_likelihoods": [-0.1]},
            "responses": [("Paris.", [-0.1], None, 1.0), ("Paris, France.", [-0.2], None, 1.0)],
        },
        "b": {"most_likely_answer": {}},  # missing response/lls: skipped
    }
    scores = semantic_entropy_scores(generations)
    assert "a" in scores
    assert "b" not in scores
    assert scores["a"] >= 0.0
