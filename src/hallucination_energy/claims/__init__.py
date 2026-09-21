"""Atomic claim extraction and simple accessors over claim sets."""
from hallucination_energy.claims.extraction import (
    Claim,
    ClaimSet,
    batch_extract_claims,
    claim_spans,
    claim_texts,
    extract_claims,
)

__all__ = [
    "Claim",
    "ClaimSet",
    "extract_claims",
    "batch_extract_claims",
    "claim_texts",
    "claim_spans",
]
