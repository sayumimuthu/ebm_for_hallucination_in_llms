"""DeBERTa-based NLI entailment checker, vendored from the upstream SEP
project's ``uncertainty/uncertainty_measures/semantic_entropy.py``.

Only ``BaseEntailment``/``EntailmentDeberta`` are kept: the GPT- and
LLaMA-based entailment judges (``EntailmentGPT4``, ``EntailmentGPT35``,
``EntailmentLLM``) are coupled to a wandb-run prediction cache and aren't
needed for this project's evidence energy, which only needs a local,
training-free entailment score (see ``retrieval.verifier.default_verifier``
and the planned Semantic Entropy baseline, ``baselines.semantic_entropy``,
which will also reuse this class for resample clustering).
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """Downloads ``microsoft/deberta-v2-xlarge-mnli`` on first use."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(DEVICE)

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        """Returns 0 (contradiction), 1 (neutral), or 2 (entailment) for
        whether ``text2`` follows from ``text1``."""
        del args, kwargs
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)
        return prediction
