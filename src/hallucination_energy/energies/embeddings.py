"""Text embedding, whitening, and pairwise-cost utilities shared by the
invariance and geometry energies.

Ported from the ``ebm.ipynb`` prototype's "Embeddings and Distances"
section. Falls back gracefully when optional dependencies
(sentence-transformers, scikit-learn) are unavailable: TF-IDF, then a hashed
character n-gram bag as a last resort, so the pipeline is still runnable
offline / in CI.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hallucination_energy._optional import (
    HAS_SENTENCE_TRANSFORMERS,
    HAS_SKLEARN,
    SentenceTransformer,
    TfidfVectorizer,
)


class _SilentFallbackDetector(logging.Handler):
    """Watches for sentence-transformers' own silent fallback: when a model
    name isn't fully available locally (observed under ``HF_HUB_OFFLINE=1``
    with an uncached model), it does NOT raise — it logs
    "No modules.json found for <name>, initializing a new SentenceTransformer
    model." and proceeds to build a fresh, effectively randomly-initialized
    model instead. An ordinary ``try/except Exception`` around the
    constructor can't catch this, since nothing raises; the resulting
    "embeddings" are numerically valid but meaningless, and silently poison
    every energy that consumes them. This handler lets ``_load_sentence_transformer``
    detect that log message and treat the load as a failure.
    """

    def __init__(self) -> None:
        super().__init__()
        self.triggered = False

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "No modules.json found" in msg or "initializing a new SentenceTransformer model" in msg:
            self.triggered = True


_SENTENCE_TRANSFORMER_CACHE: Dict[Tuple[str, Optional[str]], Optional[Any]] = {}


def _load_sentence_transformer(model_name: str, device: Optional[str] = None) -> Optional[Any]:
    """Load a SentenceTransformer, or ``None`` on any failure — including
    the silent random-model fallback described in ``_SilentFallbackDetector``,
    which does not raise and so needs its own detection, not just
    ``try/except``.

    Explicitly forces the ``sentence_transformers`` logger's level to
    ``INFO`` for the duration of the load: Python's logging filters a
    record at the *logger's effective level* before it ever reaches a
    handler, so without this, whether the fallback message reaches our
    handler at all would silently depend on whatever the calling script's
    ambient root logging level happens to be (e.g. this only worked in one
    real run because that particular script's ``main()`` happened to call
    ``logging.basicConfig(level=INFO)`` first) — this function must not
    depend on that.

    Caches the outcome (success or failure) per ``(model_name, device)``:
    ``embed_texts``/``embed_texts_pair``/``embed_texts_with_vectorizer`` each
    call this fresh with no memoization of their own, so on a real
    multi-hundred-example run this was being re-attempted (and re-logged in
    full, including sentence-transformers' own internal log line) many
    hundreds of times over — flooding logs to the point a genuine error
    would be hard to spot, and repeatedly paying the failed-construction
    cost for no reason. A negative result (unavailable/rejected) is just as
    cacheable as a positive one, since neither changes mid-process.
    """
    cache_key = (model_name, device)
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[cache_key]
    if not HAS_SENTENCE_TRANSFORMERS:
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = None
        return None
    detector = _SilentFallbackDetector()
    st_logger = logging.getLogger("sentence_transformers")
    original_level = st_logger.level
    st_logger.setLevel(logging.INFO)
    st_logger.addHandler(detector)
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        _SENTENCE_TRANSFORMER_CACHE[cache_key] = None
        return None
    finally:
        st_logger.removeHandler(detector)
        st_logger.setLevel(original_level)
    result = None if detector.triggered else model
    _SENTENCE_TRANSFORMER_CACHE[cache_key] = result
    return result


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def fit_whitener(
    X: np.ndarray, abs_eps: float = 1e-5, rel_eps: float = 1e-3, max_components: Optional[int] = None
) -> Optional[Dict[str, np.ndarray]]:
    """Fit a PCA-whitening transform ``W`` such that ``(X - mu) @ W`` has
    (approximately) identity covariance, **truncated to the data's actual
    rank**: ``min(n_samples - 1, n_features[, max_components])`` output
    dimensions.

    This matters whenever there are far fewer samples than embedding
    dimensions — the common case for the small "trusted claim" geometry
    corpus, especially with the hashed-ngram/TF-IDF fallback embeddings
    (up to 4096-D). An earlier version of this function kept the full
    ambient dimensionality and only regularized the near-zero eigenvalues
    with an epsilon floor; that stopped literal division blowups but still
    fed a density model (GMM/KDE) thousands of near-singular, effectively
    noise-only dimensions. In practice this surfaced as the *geometry*
    energy sitting at a huge, nearly-constant ~99,000-99,900 across 20 real
    examples (confirmed via ``energy_matrix.npz``), dominated by sklearn's
    per-dimension covariance regularization (``reg_covar``) summed over
    ~4000 directions the data never actually supports — a classic
    curse-of-dimensionality failure of covariance-based density estimation,
    not a per-example signal. Truncating the output to the data's real rank
    (at most ``n_samples - 1`` directions can carry any signal at all)
    fixes this at the source: the density model then only ever sees
    dimensions the data can actually inform.
    """
    if X.size == 0:
        return None
    n_samples, n_features = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.atleast_2d(np.cov(Xc, rowvar=False))
    U, s, _ = np.linalg.svd(cov, full_matrices=False)

    rank = max(1, min(n_samples - 1, n_features))
    if max_components is not None:
        rank = min(rank, max_components)
    s_top, U_top = s[:rank], U[:, :rank]

    floor = max(abs_eps, rel_eps * float(s_top[0])) if s_top.size else abs_eps
    W = U_top @ np.diag(1.0 / np.sqrt(s_top + floor))  # (n_features, rank) — NOT re-expanded to n_features
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


def _tfidf_or_hashed_embed(texts: List[str]) -> np.ndarray:
    """TF-IDF embedding of ``texts``, falling back to the hashed character
    n-gram embedding if the TF-IDF fit itself fails.

    Observed on a real 200-example run: ``TfidfVectorizer.fit_transform``
    raises ``ValueError: empty vocabulary; perhaps the documents only
    contain stop words`` whenever every text in a batch is too short (e.g.
    a claim that's just a single digit or short word) to produce any
    2+-character token under the vectorizer's default tokenizer. That's a
    real, if rare, batch composition — not something worth raising on.
    """
    if HAS_SKLEARN:
        try:
            vec = TfidfVectorizer(max_features=4096)
            X = vec.fit_transform(texts)
            return X.toarray().astype(np.float32)
        except ValueError:
            pass
    return _hashed_char_ngram_embed(texts)


def _tfidf_or_hashed_embed_pair(texts1: List[str], texts2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Like ``_tfidf_or_hashed_embed``, but fits one TF-IDF vocabulary
    jointly over both text lists (so the two outputs are comparable), with
    the same empty-vocabulary fallback."""
    if HAS_SKLEARN:
        try:
            vec = TfidfVectorizer(max_features=4096)
            X = vec.fit_transform(list(texts1) + list(texts2))
            embs1 = X[: len(texts1)].toarray().astype(np.float32)
            embs2 = X[len(texts1) :].toarray().astype(np.float32)
            return embs1, embs2
        except ValueError:
            pass
    return _hashed_char_ngram_embed(texts1), _hashed_char_ngram_embed(texts2)


def _tfidf_or_hashed_embed_with_vectorizer(
    texts: List[str], vectorizer: Optional["TfidfVectorizer"]
) -> Tuple[np.ndarray, Optional["TfidfVectorizer"]]:
    """Like ``_tfidf_or_hashed_embed``, but reuses a caller-supplied fitted
    ``vectorizer`` (see ``embed_texts_with_vectorizer``) instead of fitting
    a new one, when one is given. Returns ``(embeddings, vectorizer_used)``
    — ``None`` for the vectorizer whenever the hashed fallback engaged, so
    the caller knows there's nothing to persist/reuse.

    Note: if ``vectorizer.transform(texts)`` itself somehow raises (not the
    empty-vocabulary case this guards against, which only affects fitting),
    the hashed fallback here would land in a different feature space than
    whatever this vectorizer's earlier outputs used — an accepted, unlikely
    edge case, not the one actually observed and fixed here.
    """
    if HAS_SKLEARN:
        try:
            if vectorizer is None:
                vectorizer = TfidfVectorizer(max_features=4096)
                X = vectorizer.fit_transform(texts)
            else:
                X = vectorizer.transform(texts)
            return X.toarray().astype(np.float32), vectorizer
        except ValueError:
            pass
    return _hashed_char_ngram_embed(texts), None


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
    model = _load_sentence_transformer(model_name, device)
    if model is not None:
        try:
            embs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            embs = None
    if embs is None:
        embs = _tfidf_or_hashed_embed(texts)
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
    model = _load_sentence_transformer(model_name, device)
    if model is not None:
        try:
            embs1 = model.encode(list(texts1), convert_to_numpy=True, normalize_embeddings=False)
            embs2 = model.encode(list(texts2), convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            embs1 = embs2 = None
    if embs1 is None or embs2 is None:
        embs1, embs2 = _tfidf_or_hashed_embed_pair(texts1, texts2)
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
    model = _load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    if model is not None:
        try:
            embs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
            vectorizer = None
        except Exception:
            embs = None
    if embs is None:
        embs, vectorizer = _tfidf_or_hashed_embed_with_vectorizer(texts, vectorizer)
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
