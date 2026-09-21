"""Optional GPT-based judge for the ``llm_gpt-3.5``/``llm_gpt-4`` accuracy
metrics (see ``evaluation.accuracy.get_gpt_metric``).

Vendored from the upstream SEP project's ``uncertainty/utils/openai.py``,
with one fix: the original constructed the client with
``OpenAI(api_key='OPENAI_API_KEY')`` — the literal string, not the
environment variable's value, which would always fail authentication. The
client is also now constructed lazily inside ``predict()`` instead of at
import time, so importing this module doesn't require ``OPENAI_API_KEY``
to be set or the ``openai`` package's client to initialize eagerly.
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential  # noqa: F401
    HAS_TENACITY = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TENACITY = False

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI  # local import: optional dependency

        _CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _CLIENT


def _predict_once(prompt, temperature: float = 1.0, model: str = "gpt-4") -> str:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    if model == "gpt-4":
        model = "gpt-4-turbo"
    elif model == "gpt-3.5":
        model = "gpt-3.5-turbo"

    output = _get_client().chat.completions.create(
        model=model, messages=messages, max_tokens=200, temperature=temperature
    )
    return output.choices[0].message.content


if HAS_TENACITY:
    predict = retry(wait=wait_random_exponential(min=1, max=10))(_predict_once)
else:  # pragma: no cover - optional dependency
    predict = _predict_once


def md5hash(string: str) -> int:
    return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16)
