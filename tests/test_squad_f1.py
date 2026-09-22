"""Tests for the self-contained SQuAD F1 metric (no network access needed)."""
from __future__ import annotations

from hallucination_energy.evaluation.accuracy import squad_f1_metric
from hallucination_energy.evaluation.squad_f1 import compute_f1, normalize_answer, squad_f1


def test_normalize_answer_strips_articles_punctuation_case():
    assert normalize_answer("The Eiffel Tower!") == normalize_answer("eiffel tower")


def test_compute_f1_exact_match_is_one():
    assert compute_f1("Paris", "Paris") == 1.0


def test_compute_f1_partial_overlap():
    f1 = compute_f1("the quick brown fox", "quick brown fox jumps")
    assert 0.0 < f1 < 1.0


def test_compute_f1_no_overlap_is_zero():
    assert compute_f1("Paris", "Berlin") == 0.0


def test_compute_f1_both_empty_no_answer_case():
    assert compute_f1("", "") == 1.0


def test_compute_f1_one_empty_one_not():
    assert compute_f1("", "Paris") == 0.0
    assert compute_f1("Paris", "") == 0.0


def test_squad_f1_takes_max_over_multiple_gold_answers():
    score = squad_f1("Shakespeare", ["William Shakespeare", "Marlowe"])
    assert score == 100.0 * compute_f1("William Shakespeare", "Shakespeare")


def test_squad_f1_empty_gold_answers_no_answer_case():
    assert squad_f1("", []) == 100.0
    assert squad_f1("Paris", []) == 0.0


def test_squad_f1_metric_thresholds_at_50():
    example = {"id": "q1", "answers": {"text": ["Paris"], "answer_start": [0]}}
    assert squad_f1_metric("Paris", example) == 1.0
    assert squad_f1_metric("Berlin", example) == 0.0
