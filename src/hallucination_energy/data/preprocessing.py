"""Dataset-agnostic prompt construction and dataset splitting.

Vendored from the upstream SEP project's ``uncertainty/utils/utils.py``
(the prompt-construction and few-shot pieces only — the model
initialization, wandb-coupled ``save``, and accuracy-metric registry live
in ``generation.hf_model``/``generation.io_utils``/``evaluation.accuracy``
respectively, to keep this module's concerns to "turn dataset rows into
prompts").
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

BRIEF_PROMPTS = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}


def get_make_prompt(use_context: bool) -> Callable:
    """Return a ``make_prompt(context, question, answer, brief, brief_always)``
    function. ``answer=None`` renders a prompt awaiting completion;
    otherwise it renders a filled-in few-shot example."""

    def make_prompt(context, question, answer, brief, brief_always):
        prompt = ""
        if brief_always:
            prompt += brief
        if use_context and (context is not None):
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += "Answer:"
        return prompt

    return make_prompt


def construct_fewshot_prompt_from_indices(
    dataset, example_indices, brief: str, brief_always: bool, make_prompt: Callable
) -> str:
    """Given a dataset and indices, construct a fewshot prompt."""
    prompt = brief if not brief_always else ""
    for example_index in example_indices:
        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]
        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)
    return prompt


def split_dataset(dataset) -> Tuple[List[int], List[int]]:
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    assert set(answerable_indices) | set(unanswerable_indices) == set(range(len(dataset)))
    assert set(answerable_indices) - set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def get_reference(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an example (or a generation record wrapping one under
    ``example['reference']``) into the SQuAD-metric reference format."""
    if "answers" not in example:
        example = example["reference"]
    answers = example["answers"]
    answer_starts = answers.get("answer_start", [])
    return {"answers": {"answer_start": answer_starts, "text": answers["text"]}, "id": example["id"]}
