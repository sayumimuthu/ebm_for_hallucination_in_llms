"""Self-contained SQuAD-style token-overlap F1, replacing a dependency on
``evaluate.load("squad_v2")``.

The upstream SEP project's accuracy metric downloaded the ``squad_v2``
*metric script* via the ``evaluate`` library at generation time. Two
problems with that, discovered while getting this pipeline running:

1. It routes through the same HF Hub dataset/metric-script resolution
   mechanism that broke ``datasets.load_dataset("squad_v2")`` (Hub
   deprecated resolving bare, un-namespaced repo ids — see
   ``data.loaders``'s fix to ``rajpurkar/squad_v2``). The metric-script
   equivalent is liable to break the same way, and/or require a
   ``trust_remote_code=True`` flag, since ``evaluate`` metric scripts are
   the same deprecated script-loading mechanism as old-style datasets.
2. It's an extra network round-trip and dependency (``evaluate``, which
   pulls in a large chunk of the ``transformers``/``tensorflow`` optional
   import graph) purely to compute a formula that fits in ~30 lines.

This is a direct reimplementation of the official SQuAD 2.0 evaluation
script's ``normalize_answer``/``compute_f1`` (the same logic
``evaluate``'s own ``squad_v2`` metric implements), so scores are
identical — just computed locally, with no network access needed at
metric-evaluation time.
"""
from __future__ import annotations

import collections
import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(gold_answer: str, prediction: str) -> float:
    """Token-overlap F1 between one gold answer and the prediction. If
    either is empty (a "no answer" case), F1 is 1.0 iff both are empty."""
    gold_toks = get_tokens(gold_answer)
    pred_toks = get_tokens(prediction)
    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def squad_f1(prediction: str, gold_answers: List[str]) -> float:
    """Max F1 (as a 0-100 percentage, matching ``evaluate``'s scale) over
    every acceptable gold answer. An empty ``gold_answers`` list (SQuAD 2.0
    "no answer") scores 100 iff ``prediction`` is also empty."""
    if not gold_answers:
        return 100.0 * float(not get_tokens(prediction))
    return 100.0 * max(compute_f1(gold, prediction) for gold in gold_answers)
