"""Ground-truth accuracy metrics for generated answers, vendored from the
upstream SEP project's ``uncertainty/utils/utils.py``.

These decide whether a generation counts as "correct" (used both as the
generation-time accuracy label and, downstream, as the hallucination label
for calibration/training — e.g. ``scripts/compute_energy_features.py``
treats ``accuracy < threshold`` as "hallucinated").

The ``openai``-backed judge (``llm_gpt-3.5``/``llm_gpt-4``) is imported
lazily so that the default ``squad`` metric (token-overlap F1 against
reference answers) works without an OpenAI API key or the ``openai``
package installed at all.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from hallucination_energy.data.preprocessing import get_reference
from hallucination_energy.evaluation.squad_f1 import squad_f1


def model_based_metric(predicted_answer: str, example: Dict[str, Any], model) -> float:
    """Ask the generator LLM itself (or a judge model) whether the
    predicted answer means the same thing as the reference answer(s)."""
    if "answers" in example:
        correct_answers = example["answers"]["text"]
    elif "reference" in example:
        correct_answers = example["reference"]["answers"]["text"]
    else:
        raise ValueError("Example has neither 'answers' nor 'reference'.")

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if "gpt" in model.model_name.lower():
        predicted = model.predict(prompt, 0.01)
    else:
        predicted, _, _ = model.predict(prompt, 0.01)

    if "yes" in predicted.lower():
        return 1.0
    elif "no" in predicted.lower():
        return 0.0
    else:
        logging.warning("Redo llm check.")
        predicted = model.predict(prompt, 1)
        if "yes" in predicted.lower():
            return 1.0
        elif "no" in predicted.lower():
            return 0.0
        logging.warning("Answer neither no nor yes. Defaulting to no!")
        return 0.0


def llm_metric(predicted_answer: str, example: Dict[str, Any], model) -> float:
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name: str) -> Callable:
    """``metric_name`` like 'llm_gpt-4' or 'llm_gpt-3.5'."""
    model_name = "_".join(metric_name.split("_")[1:])

    class EntailmentGPT:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def predict(self, prompt: str, temperature: float) -> str:
            from hallucination_energy.data import openai_judge  # local import: optional dependency

            return openai_judge.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def squad_f1_metric(response: str, example: Dict[str, Any], *args, **kwargs) -> float:
    """1.0 if the token-overlap F1 (``evaluation.squad_f1``) between
    ``response`` and the best-matching gold answer is >= 50, else 0.0.
    Needs no network access, LLM judge, or API key — see
    ``evaluation.squad_f1`` module docstring for why this replaced an
    ``evaluate.load("squad_v2")`` call."""
    del args, kwargs
    reference = get_reference(example)
    f1 = squad_f1(response, reference["answers"]["text"])
    return 1.0 if f1 >= 50.0 else 0.0


def get_metric(metric: str) -> Callable:
    """Return an ``metric(response, example, model) -> float in {0.0, 1.0}``
    accuracy function. ``'squad'`` (default) needs no LLM judge or API key."""
    if metric == "squad":
        return squad_f1_metric
    elif metric == "llm":
        return llm_metric
    elif metric in ("llm_gpt-3.5", "llm_gpt-4"):
        return get_gpt_metric(metric)
    else:
        raise ValueError(f"Unknown metric={metric!r}. Supported: squad, llm, llm_gpt-3.5, llm_gpt-4.")
