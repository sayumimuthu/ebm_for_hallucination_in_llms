"""Text embedding, whitening, and pairwise-cost utilities shared by the
invariance and geometry energies.

Ported from the ``ebm.ipynb`` prototype's "Embeddings and Distances"
section. Falls back gracefully when optional dependencies
(sentence-transformers, scikit-learn) are unavailable: TF-IDF, then a hashed
character n-gram bag as a last resort, so the pipeline is still runnable
offline / in CI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from hallucination_energy._optional import (
    HAS_SENTENCE_TRANSFORMERS,
    HAS_SKLEARN,
    SentenceTransformer,
    TfidfVectorizer,
)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def fit_whitener(X: np.ndarray, eps: float = 1e-5) -> Optional[Dict[str, np.ndarray]]:
    if X.size == 0:
        return None
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    cov += eps * np.eye(cov.shape[0])
    U, s, _ = np.linalg.svd(cov, full_matrices=False)
    W = U @ np.diag(1.0 / np.sqrt(s + eps)) @ U.T
    return {"mu": mu, "W": W}


def apply_whitener(X: np.ndarray, whitener: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if not whitener:
        return X
    if not isinstance(whitener, dict):
        return X
    mu = whitener.get("mu")
    W = whitener.get("W")
    if mu is None or W is None:
        return X
    if mu.shape[1] != X.shape[1] or W.shape[0] != X.shape[1]:
        return X
    return (X - mu) @ W


def _hashed_char_ngram_embed(texts: List[str], char_dim: int = 256) -> np.ndarray:
    embs = np.zeros((len(texts), char_dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t):
            idx = (ord(ch) * 131 + j * 17) % char_dim
            embs[i, idx] += 1.0
    return embs


def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    metric: str = "cosine",
    whitener: Optional[Dict[str, np.ndarray]] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), metric
    embs = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer(model_name, device=device)
            embs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            embs = None
    if embs is None:
        if HAS_SKLEARN:
            vec = TfidfVectorizer(max_features=4096)
            X = vec.fit_transform(texts)
            embs = X.toarray().astype(np.float32)
        else:
            embs = _hashed_char_ngram_embed(texts)
    embs = embs.astype(np.float32, copy=False)
    if whitener:
        embs = apply_whitener(embs, whitener)
    if metric == "cosine":
        embs = l2_normalize(embs)
    return embs, metric


def embed_texts_pair(
    texts1: List[str],
    texts2: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    metric: str = "cosine",
    whitener: Optional[Dict[str, np.ndarray]] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if not texts1 and not texts2:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            metric,
        )
    embs1 = embs2 = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer(model_name, device=device)
            embs1 = model.encode(list(texts1), convert_to_numpy=True, normalize_embeddings=False)
            embs2 = model.encode(list(texts2), convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            embs1 = embs2 = None
    if embs1 is None or embs2 is None:
        if HAS_SKLEARN:
            vec = TfidfVectorizer(max_features=4096)
            X = vec.fit_transform(list(texts1) + list(texts2))
            embs1 = X[: len(texts1)].toarray().astype(np.float32)
            embs2 = X[len(texts1) :].toarray().astype(np.float32)
        else:
            embs1 = _hashed_char_ngram_embed(texts1)
            embs2 = _hashed_char_ngram_embed(texts2)
    embs1 = embs1.astype(np.float32, copy=False)
    embs2 = embs2.astype(np.float32, copy=False)
    if whitener:
        embs1 = apply_whitener(embs1, whitener)
        embs2 = apply_whitener(embs2, whitener)
    if metric == "cosine":
        embs1 = l2_normalize(embs1)
        embs2 = l2_normalize(embs2)
    return embs1, embs2, metric


def embed_texts_with_vectorizer(
    texts: List[str],
    vectorizer: Optional["TfidfVectorizer"] = None,
    metric: str = "cosine",
    whitener: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, str, Optional["TfidfVectorizer"]]:
    """Like ``embed_texts``, but for the TF-IDF fallback path, reuses a
    caller-supplied, already-fitted ``vectorizer`` instead of fitting a new
    one from scratch.

    This matters whenever embeddings computed in two separate calls need to
    land in the *same* feature space to be comparable (e.g. scoring a new
    claim's embedding against a density model fit on a reference corpus, or
    computing a pairwise cost matrix between two claim sets embedded
    separately). A fresh ``TfidfVectorizer().fit_transform(texts)`` per call
    builds a vocabulary from only that call's texts, so two such calls
    generally produce vectors of different, incompatible dimensionality —
    this function exists to avoid that trap. The sentence-transformers path
    needs no such care (it is a fixed, pretrained embedding space), and the
    hashed character n-gram fallback is already deterministic per-text
    regardless of what else is in the batch.

    Returns ``(embeddings, metric, fitted_vectorizer_or_None)`` so the
    caller can persist the vectorizer (``vectorizer_or_None`` is ``None``
    whenever the sentence-transformers or hashed-fallback path was used, in
    which case there is nothing that needs to be persisted).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), metric, vectorizer
    embs = None
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
            vectorizer = None
        except Exception:
            embs = None
    if embs is None:
        if HAS_SKLEARN:
            if vectorizer is None:
                vectorizer = TfidfVectorizer(max_features=4096)
                X = vectorizer.fit_transform(texts)
            else:
                X = vectorizer.transform(texts)
            embs = X.toarray().astype(np.float32)
        else:
            embs = _hashed_char_ngram_embed(texts)
            vectorizer = None
    embs = embs.astype(np.float32, copy=False)
    if whitener:
        embs = apply_whitener(embs, whitener)
    if metric == "cosine":
        embs = l2_normalize(embs)
    return embs, metric, vectorizer


def pairwise_cost(E1: np.ndarray, E2: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if E1.size == 0 or E2.size == 0:
        return np.zeros((len(E1), len(E2)), dtype=np.float32)
    if metric == "cosine":
        sims = E1 @ E2.T
        return 1.0 - sims
    a = (E1**2).sum(axis=1, keepdims=True)
    b = (E2**2).sum(axis=1, keepdims=True).T
    return np.sqrt(np.maximum(a + b - 2.0 * (E1 @ E2.T), 0.0))
