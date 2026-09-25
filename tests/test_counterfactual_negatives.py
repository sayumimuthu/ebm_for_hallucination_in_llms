"""Tests for the targeted counterfactual negative-generation module.

These deliberately run WITHOUT spaCy installed (matching the local dev
environment and CI), so every function here exercises the regex/heuristic
fallback path — see ``hallucination_energy._optional.get_spacy_model``. A
future environment with spaCy installed should still pass these (the
assertions only check semantic properties of the output, not which code
path produced it), but the fallback path is what's actually verified here.
"""
from __future__ import annotations

import numpy as np
import pytest

from hallucination_energy.claims.extraction import extract_claims
from hallucination_energy.training.counterfactual_negatives import (
    EntityPool,
    build_entity_pool,
    generate_counterfactual_negatives,
    invert_relation,
    negate_claim,
    perturb_date_or_temporal,
    substitute_entities,
)


@pytest.fixture(autouse=True)
def _force_no_spacy(monkeypatch):
    """Force the regex/heuristic fallback path, matching this dev
    environment (spaCy isn't installed) rather than depending on whether
    it happens to be present wherever tests run."""
    monkeypatch.setattr("hallucination_energy.training.counterfactual_negatives.get_spacy_model", lambda: None)


def test_build_entity_pool_buckets_capitalized_spans():
    pool = build_entity_pool(["Paris is the capital of France.", "Berlin is the capital of Germany."])
    assert "ENT" in pool.by_label
    surfaces = {s.lower() for s in pool.by_label["ENT"]}
    assert "france" in surfaces or "paris" in surfaces


def test_substitute_entities_swaps_for_a_different_entity():
    pool = EntityPool(by_label={"ENT": ["France", "Germany", "Italy"]})
    rng = np.random.default_rng(0)
    result = substitute_entities("Paris is the capital of France.", pool, rng)
    assert result is not None
    assert result != "Paris is the capital of France."
    assert any(alt in result for alt in ["Germany", "Italy"])


def test_substitute_entities_returns_none_without_candidates():
    pool = EntityPool(by_label={})
    rng = np.random.default_rng(0)
    assert substitute_entities("no capitalized spans here", pool, rng) is None


def test_substitute_entities_returns_none_when_only_self_match():
    pool = EntityPool(by_label={"ENT": ["France"]})
    rng = np.random.default_rng(0)
    assert substitute_entities("Paris is the capital of France.", pool, rng) is None


def test_negate_claim_removes_existing_negation():
    result = negate_claim("Water is not a compound.")
    assert result is not None
    assert "not" not in result.lower()


def test_negate_claim_inserts_negation_after_auxiliary():
    result = negate_claim("Insulin is secreted by pancreatic beta cells.")
    assert result is not None
    assert "not" in result.lower()
    assert result != "Insulin is secreted by pancreatic beta cells."


def test_negate_claim_falls_back_to_prefix_without_auxiliary():
    result = negate_claim("Cats purr.")
    assert result is not None
    assert result.lower().startswith("it is not true that")


def test_invert_relation_swaps_first_and_last_capitalized_spans():
    # Note: a capitalized span at position 0 is skipped (assumed to be
    # sentence-initial capitalization, not necessarily an entity), so this
    # needs two non-initial spans to swap.
    original = "The winner was France over Germany."
    result = invert_relation(original)
    assert result is not None
    assert result != original
    assert result.index("Germany") < result.index("France")


def test_invert_relation_returns_none_with_fewer_than_two_entities():
    assert invert_relation("cats purr softly") is None


def test_perturb_date_swaps_month():
    rng = np.random.default_rng(0)
    result = perturb_date_or_temporal("The event happened in March 1990.", rng)
    assert result is not None
    assert "March" not in result


def test_perturb_date_shifts_year_when_no_month():
    rng = np.random.default_rng(0)
    result = perturb_date_or_temporal("The treaty was signed in 1990.", rng)
    assert result is not None
    assert "1990" not in result


def test_perturb_date_returns_none_without_date():
    rng = np.random.default_rng(0)
    assert perturb_date_or_temporal("Cats purr softly.", rng) is None


def test_generate_counterfactual_negatives_end_to_end():
    claim_set = extract_claims("Insulin is secreted by pancreatic beta cells in the year 1990.")
    pool = build_entity_pool(["Glucagon is secreted by pancreatic alpha cells."])
    rng = np.random.default_rng(0)
    negatives = generate_counterfactual_negatives(claim_set, pool, rng, num_negatives=2)
    assert len(negatives) <= 2
    for category, text in negatives:
        assert isinstance(category, str)
        assert text.strip() != ""


def test_generate_counterfactual_negatives_empty_claim_set():
    claim_set = extract_claims("")
    pool = EntityPool()
    rng = np.random.default_rng(0)
    assert generate_counterfactual_negatives(claim_set, pool, rng) == []
