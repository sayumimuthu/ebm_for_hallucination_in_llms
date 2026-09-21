"""A minimal, embedding-based context retriever.

Ported from ``ebm.ipynb``. This is intentionally simple: it retrieves top-k
documents from whatever context/evidence fields happen to be present on a
generation record. It is a placeholder for a real retrieval stack (e.g. a
dense retriever over a fixed corpus + reranker, see
``retrieval.reranker``) and is sufficient for the v0 evidence energy on
datasets that already ship gold/retrieved context (e.g. SQuAD, BioASQ).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from hallucination_energy._optional import HAS_SENTENCE_TRANSFORMERS, HAS_SKLEARN
from hallucination_energy.energies.embeddings import embed_texts


def collect_docs_from_example(example: Dict[str, Any], max_docs: int = 8) -> List[str]:
    """Pull candidate evidence passages out of a generation record's
    ``context``/``retrieved_docs``/``passages``/``evidence`` fields."""
    docs: List[str] = []
    ctx = example.get("context")
    if isinstance(ctx, str):
        docs.append(ctx)
    elif isinstance(ctx, list):
        docs.extend([c for c in ctx if isinstance(c, str)])
    for key in ("retrieved_docs", "retrieved_passages", "passages", "evidence", "docs"):
        value = example.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    docs.append(item)
                elif isinstance(item, (list, tuple)) and item:
                    if isinstance(item[0], str):
                        docs.append(item[0])
                elif isinstance(item, dict):
                    for field in ("text", "passage", "content", "evidence"):
                        if field in item and isinstance(item[field], str):
                            docs.append(item[field])
                            break
    seen = set()
    result: List[str] = []
    for doc in docs:
        if not doc or doc in seen:
            continue
        result.append(doc)
        seen.add(doc)
        if len(result) >= max_docs:
            break
    return result


class ContextRetriever:
    """Retrieve the top-k most similar documents to a query claim from a
    small, fixed pool of candidate documents (not a full corpus index)."""

    def __init__(self, docs: List[str], metric: str = "cosine") -> None:
        self.docs = [d for d in docs if isinstance(d, str) and d.strip()]
        self.metric = metric
        self._embeddings: Optional[np.ndarray] = None

    def __call__(self, query: str, top_k: int = 5) -> List[str]:
        if not self.docs:
            return []
        if len(self.docs) <= top_k:
            return self.docs[:top_k]
        if self.metric == "cosine" and (HAS_SENTENCE_TRANSFORMERS or HAS_SKLEARN):
            if self._embeddings is None:
                self._embeddings, _ = embed_texts(self.docs, metric=self.metric)
            q_emb, _ = embed_texts([query], metric=self.metric)
            if q_emb.size == 0:
                return self.docs[:top_k]
            sims = self._embeddings @ q_emb.T
            order = np.argsort(sims[:, 0])[::-1][:top_k]
            return [self.docs[i] for i in order]
        return self.docs[:top_k]
