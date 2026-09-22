"""Regression test for a silent-failure bug found running on a real cluster:
under ``HF_HUB_OFFLINE=1`` with the model not locally cached,
``sentence_transformers.SentenceTransformer(...)`` does not raise — it logs
"No modules.json found for <name>, initializing a new SentenceTransformer
model." and proceeds to build a fresh, effectively randomly-initialized
model. An ordinary ``try/except Exception`` around the constructor can't
catch this, so embeddings silently became noise, which in turn made the
geometry energy's AUROC land at exactly chance (0.5) with near-zero NCE
weight on a real 20-example run — a textbook silent-failure symptom, not an
exception anyone could have caught downstream.

This test doesn't require sentence-transformers to be installed: it
monkeypatches ``embeddings.HAS_SENTENCE_TRANSFORMERS`` and
``embeddings.SentenceTransformer`` directly, so it exercises
``_load_sentence_transformer``'s detection logic in isolation.
"""
from __future__ import annotations

import logging

import pytest

import hallucination_energy.energies.embeddings as embeddings


@pytest.fixture(autouse=True)
def _clear_sentence_transformer_cache():
    """``_load_sentence_transformer`` now caches its outcome per
    (model_name, device) — necessary in production (see its docstring: a
    real run re-attempted and re-logged the same failed load hundreds of
    times), but it means the module-level cache would otherwise leak
    between these tests, since they all use the same model name with
    different monkeypatched behavior."""
    embeddings._SENTENCE_TRANSFORMER_CACHE.clear()
    yield
    embeddings._SENTENCE_TRANSFORMER_CACHE.clear()


class _FakeSilentFallbackModel:
    """Stands in for what sentence-transformers actually does: logs the
    fallback message and returns a (bogus) model object without raising."""

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        raise AssertionError("embed_texts should never call .encode() on a detected fallback model")


def _fake_sentence_transformer_silent_fallback(model_name, device=None):
    logging.getLogger("sentence_transformers").info(
        "No modules.json found for %s, initializing a new SentenceTransformer model.", model_name
    )
    return _FakeSilentFallbackModel()


class _FakeGoodModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        import numpy as np

        return np.ones((len(texts), 4), dtype=np.float32)


def _fake_sentence_transformer_success(model_name, device=None):
    return _FakeGoodModel()


def test_silent_fallback_is_detected_and_treated_as_a_load_failure(monkeypatch):
    monkeypatch.setattr(embeddings, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(embeddings, "SentenceTransformer", _fake_sentence_transformer_silent_fallback)

    model = embeddings._load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

    assert model is None  # detected as a failure despite no exception being raised


def test_real_load_still_returns_the_model(monkeypatch):
    monkeypatch.setattr(embeddings, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(embeddings, "SentenceTransformer", _fake_sentence_transformer_success)

    model = embeddings._load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

    assert isinstance(model, _FakeGoodModel)


def test_embed_texts_falls_back_to_tfidf_on_silent_fallback(monkeypatch):
    """End-to-end: embed_texts() should produce real (TF-IDF/hashed)
    embeddings, not silently-random ones, when the ST load is a detected
    fallback."""
    monkeypatch.setattr(embeddings, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(embeddings, "SentenceTransformer", _fake_sentence_transformer_silent_fallback)

    embs, metric = embeddings.embed_texts(["Paris is the capital of France.", "Unrelated sentence about cats."])

    assert embs.shape[0] == 2
    # The two unrelated sentences shouldn't be embedded as identical/constant
    # vectors — that would indicate the (rejected) fallback model's garbage
    # output leaked through instead of a real fallback embedding.
    assert not (embs[0] == embs[1]).all()
