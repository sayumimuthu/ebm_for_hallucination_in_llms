"""Tests for the vendored, network-free dataset preprocessing utilities
(``data.preprocessing``). ``data.loaders.load_ds`` itself needs network
access (HuggingFace `datasets` downloads, or a local bioasq file) and is
intentionally not covered here — see README for how to smoke-test it.
"""
from __future__ import annotations

from hallucination_energy.data.preprocessing import (
    construct_fewshot_prompt_from_indices,
    get_make_prompt,
    get_reference,
    split_dataset,
)

TOY_DATASET = [
    {"id": "q1", "question": "What is the capital of France?", "context": "France is a country in Europe.", "answers": {"text": ["Paris"]}},
    {"id": "q2", "question": "What is the capital of nowhere?", "context": None, "answers": {"text": []}},
    {"id": "q3", "question": "Who wrote Hamlet?", "context": "Hamlet is a play.", "answers": {"text": ["Shakespeare"]}},
]


def test_split_dataset_separates_answerable_and_unanswerable():
    answerable, unanswerable = split_dataset(TOY_DATASET)
    assert answerable == [0, 2]
    assert unanswerable == [1]


def test_make_prompt_with_context():
    make_prompt = get_make_prompt(use_context=True)
    prompt = make_prompt("France is a country.", "What is the capital?", "Paris", "Brief.\n", False)
    assert "Context: France is a country." in prompt
    assert "Question: What is the capital?" in prompt
    assert "Answer: Paris" in prompt


def test_make_prompt_without_context_omits_it_even_if_use_context_true():
    make_prompt = get_make_prompt(use_context=True)
    prompt = make_prompt(None, "Who wrote Hamlet?", None, "Brief.\n", False)
    assert "Context:" not in prompt
    assert prompt.endswith("Answer:")


def test_make_prompt_brief_always():
    make_prompt = get_make_prompt(use_context=False)
    prompt = make_prompt(None, "Q?", None, "Be brief.\n", True)
    assert prompt.startswith("Be brief.\n")


def test_construct_fewshot_prompt_from_indices():
    make_prompt = get_make_prompt(use_context=True)
    prompt = construct_fewshot_prompt_from_indices(TOY_DATASET, [0, 2], "Brief.\n", False, make_prompt)
    assert "Paris" in prompt
    assert "Shakespeare" in prompt
    assert prompt.startswith("Brief.\n")


def test_get_reference_from_example():
    ref = get_reference(TOY_DATASET[0])
    assert ref == {"answers": {"answer_start": [], "text": ["Paris"]}, "id": "q1"}


def test_get_reference_from_generation_record():
    record = {"reference": TOY_DATASET[2]}
    ref = get_reference(record)
    assert ref["id"] == "q3"
    assert ref["answers"]["text"] == ["Shakespeare"]
