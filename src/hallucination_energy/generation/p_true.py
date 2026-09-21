"""The p(True) self-evaluation baseline (Kadavath et al., 2022), vendored
from the upstream SEP project's ``uncertainty/uncertainty_measures/p_true.py``.

Used by ``generation.generate`` when ``--compute_p_true`` is set: build a
few-shot prompt where the model brainstorms answers and judges whether its
own most-likely answer is "True" or "False", then read off the model's
log-probability of "True" as a confidence score.

Dropped from upstream: an unused, eagerly-loaded ``squad_metric = load
("squad_v2")`` at module import time — dead code (the function always uses
the ``metric`` callable passed in by the caller instead), and it forced a
metric download on every import of this module regardless of whether squad
was even the dataset in use.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple


def construct_few_shot_prompt(
    *, model, dataset, indices, prompt, brief, brief_always, make_prompt, num_generations, metric
) -> Tuple[str, Dict[int, Dict[str, Any]], int]:
    """Construct the few-shot prompt for the p(True) metric by having the
    model brainstorm answers to ``indices`` questions and self-labeling
    each as True/False against the accuracy ``metric``."""
    few_shot_prompt: List[str] = []
    all_responses: Dict[int, Dict[str, Any]] = {}
    it = 0
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ["\n"]
        prompt_candidate += ["Question: " + question]
        prompt_candidate += ["\nBrainstormed Answers: "]
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info("P_TRUE >> Current Question: ".ljust(25) + current_question)

        responses = []
        most_likely_response = None
        is_correct = 0.0
        for j in range(num_generations + 1):
            temperature = 0.1 if j == 0 else 1.0
            response, _, _ = model.predict(local_prompt, temperature)
            logging.info("P_TRUE >> Current Response: ".ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f"{response.strip()} \n"]
            if j == 0:
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = list(example["answers"]["text"])
                logging.info("P_TRUE >> LOW-T >> true answer: ".ljust(35) + str(answers))
                logging.info("P_TRUE >> LOW-T >> acc: ".ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response, is_correct=is_correct
        )

        prompt_candidate += ["Possible answer: " + most_likely_response + "\n"]
        prompt_candidate += ["Is the possible answer:\n"]
        prompt_candidate += ["A) True\n"]
        prompt_candidate += ["B) False\n"]
        prompt_candidate += ["The possible answer is:"]
        prompt_candidate += [" A" if is_correct else " B"]

        prompt_len = len(model.tokenizer.encode("".join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.max_new_tokens`
        # extra tokens, plus a 200-token buffer for question + 'Possible answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning("Cutting off p_true prompt at index %d.", it)
            break

    return "".join(few_shot_prompt), all_responses, it


def calculate_p_true(
    model, question: str, most_probable_answer: str, brainstormed_answers: List[str],
    few_shot_prompt: str, hint: bool = False,
) -> float:
    """Return the model's log-probability that its most-likely answer is
    True, given a few-shot prompt of prior brainstorm-and-judge examples."""
    prompt = few_shot_prompt + "\n" if few_shot_prompt else ""

    prompt += "Question: " + question
    prompt += "\nBrainstormed Answers: "
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + "\n"
    prompt += "Possible answer: " + most_probable_answer + "\n"
    if not hint:
        prompt += "Is the possible answer:\n"
        prompt += "A) True\n"
        prompt += "B) False\n"
        prompt += "The possible answer is:"
    else:
        prompt += (
            "Do the brainstormed answers match the possible answer? "
            "Respond with A if they do, if they do not respond with B. Answer:"
        )

    return model.get_p_true(prompt)
