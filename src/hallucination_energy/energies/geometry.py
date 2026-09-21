"""Geometry energy: does the claim's representation lie in a trusted region?

Fits a density model (GMM, falling back to KDE, falling back to a single
Gaussian/Mahalanobis) over embeddings of "trusted" (high-accuracy) claims,
then scores new claims by negative log-density. Ported from ``ebm.ipynb``.

Per the project plan, this Mahalanobis/GMM/KDE density is a v0 stand-in for
a learned density/energy function over generator hidden states (e.g. an
SAE-based potential-energy landscape, as in the HalluSAE baseline) — see
README roadmap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hallucination_energy._optional import HAS_GMM, HAS_SKLEARN, GaussianMixture, KernelDensity
from hallucination_energy.claims.extraction import ClaimSet, claim_texts
from hallucination_energy.energies.embeddings import (
    apply_whitener,
    embed_texts_with_vectorizer,
    fit_whitener,
    l2_normalize,
)


def fit_geometry_density(
    claim_corpus: List[str],
    metric: str = "cosine",
    n_components: int = 8,
    whiten: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    if not claim_corpus:
        return None, {"metric": metric, "whitener": None, "vectorizer": None}
    embeddings, _, vectorizer = embed_texts_with_vectorizer(claim_corpus, metric="euclidean")
    whitener = fit_whitener(embeddings) if whiten else None
    Z = apply_whitener(embeddings, whitener) if whitener else embeddings
    if metric == "cosine":
        Z = l2_normalize(Z)
    density: Any
    if HAS_GMM and len(claim_corpus) >= min(n_components, 2):
        K = min(n_components, len(claim_corpus))
        gmm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=1e-5, random_state=42)
        gmm.fit(Z)
        density = ("gmm", gmm)
    elif HAS_SKLEARN:
        kde = KernelDensity(kernel="gaussian", bandwidth=1.0)
        kde.fit(Z)
        density = ("kde", kde)
    else:
        mu = Z.mean(axis=0, keepdims=True)
        cov = np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1])
        density = ("gaussian", mu, np.linalg.inv(cov))
    # `vectorizer` is only non-None on the TF-IDF fallback path; it must be
    # reused (not re-fit) when embedding future claims so they land in the
    # same feature space as this density model (see embed_texts_with_vectorizer).
    return density, {"metric": metric, "whitener": whitener, "vectorizer": vectorizer}


def geometry_energy(
    claim_set: ClaimSet,
    density_model: Any,
    embed_meta: Optional[Dict[str, Any]] = None,
) -> float:
    if claim_set is None or not claim_set.claims or density_model is None:
        return 0.0
    texts = claim_texts(claim_set)
    meta = embed_meta or {}
    metric = meta.get("metric", "cosine")
    whitener = meta.get("whitener")
    embeddings, _, _ = embed_texts_with_vectorizer(
        texts, vectorizer=meta.get("vectorizer"), metric="euclidean", whitener=whitener
    )
    if metric == "cosine":
        embeddings = l2_normalize(embeddings)
    tag = density_model[0]
    if tag == "gmm":
        gmm = density_model[1]
        ll = gmm.score_samples(embeddings)
        return float(-np.mean(ll))
    if tag == "kde":
        kde = density_model[1]
        ll = kde.score_samples(embeddings)
        return float(-np.mean(ll))
    _, mu, inv = density_model
    dif = embeddings - mu
    quad = np.einsum("nd,dd,nd->n", dif, inv, dif)
    return float(0.5 * np.mean(quad))
