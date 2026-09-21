"""NLI-style claim verification against a retrieved evidence passage.

``default_verifier`` uses an entailment model (DeBERTa-based, from the
upstream SEP package) when available, and falls back to an embedding
cosine-similarity proxy otherwise. Used by ``energies.evidence.evidence_energy``.
"""
from __future__ import annotations

from hallucination_energy._optional import get_entailment_model
from hallucination_energy.energies.embeddings import embed_texts


def default_verifier(claim: str, doc: str) -> float:
    """Return a scalar "support" score for ``claim`` given evidence ``doc``.

    With the DeBERTa entailment model: +2 (entailment), 0 (neutral),
    -2 (contradiction). Without it: cosine similarity between claim and
    document embeddings, as a weak proxy for support.
    """
    model = get_entailment_model()
    if model is not None:
        try:
            pred = model.check_implication(doc, claim)
            mapping = {0: -2.0, 1: 0.0, 2: 2.0}
            return float(mapping.get(pred, 0.0))
        except Exception:
            pass
    embs, _ = embed_texts([claim, doc], metric="cosine")
    if embs.shape[0] < 2:
        return 0.0
    return float(embs[0] @ embs[1])
