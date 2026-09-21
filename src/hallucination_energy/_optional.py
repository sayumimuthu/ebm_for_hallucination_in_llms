"""Centralized optional-dependency detection with graceful fallbacks.

Several third-party libraries (sentence-transformers, scikit-learn, scipy,
an NLI entailment model from the upstream Semantic-Entropy-Probes / SEP
project) are "nice to have" but not strictly required to exercise the
energies on toy data. Every optional import used across the package is
resolved here once, instead of being scattered across modules as ad hoc
try/except blocks (as in the original prototype notebook).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
    from sklearn.neighbors import KernelDensity  # noqa: F401
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]
    KernelDensity = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]
    HAS_SKLEARN = False

try:
    from sklearn.mixture import GaussianMixture  # noqa: F401
    HAS_GMM = True
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None  # type: ignore[assignment]
    HAS_GMM = False

try:
    from scipy.special import logsumexp  # noqa: F401
    HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    logsumexp = None  # type: ignore[assignment]
    HAS_SCIPY = False


def logsumexp_fallback(x, axis=None):
    """logsumexp that works whether or not scipy is installed."""
    import numpy as np

    if HAS_SCIPY:
        return logsumexp(x, axis=axis)
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


_DEBERTA_MODEL: Optional[Any] = None


def get_entailment_model() -> Optional[Any]:
    """Lazily load ``retrieval.entailment.EntailmentDeberta`` (downloads
    ``microsoft/deberta-v2-xlarge-mnli`` on first call). Returns ``None`` if
    that fails (e.g. no internet access, transformers/torch unavailable),
    in which case callers should fall back to an embedding-similarity
    proxy.
    """
    global _DEBERTA_MODEL
    if _DEBERTA_MODEL is not None:
        return _DEBERTA_MODEL
    try:
        from hallucination_energy.retrieval.entailment import EntailmentDeberta

        _DEBERTA_MODEL = EntailmentDeberta()
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("EntailmentDeberta unavailable; falling back to embedding similarity.")
        _DEBERTA_MODEL = None
    return _DEBERTA_MODEL
