"""Rule-based atomic claim extraction.

Ported from the ``ebm.ipynb`` prototype's "Claim Extraction" section.
This is a lightweight sentence/numeric-span segmenter used as a v0 claim
extractor: fast, dependency-free, and good enough to bootstrap the energy
pipeline. It does not perform semantic decomposition of compound sentences
into fully atomic propositions (e.g. via an LLM-based claim splitter) — that
upgrade is left as future work (see README roadmap).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Claim:
    text: str
    span: Tuple[int, int]  # token indices [start, end)
    kind: str = "sentence"


@dataclass
class ClaimSet:
    claims: List[Claim]
    tokens: List[str]


_TOKEN_RE = re.compile(r"\d{1,4}(?:[\-/]\d{1,2}){1,2}|\d{1,3}(?:,\d{3})*(?:\.\d+)?|\w+|[^\w\s]")
_PUNCT_NO_SPACE = {".", ",", ";", ":", "?", "!", ")", "]", "}", "'s", "'re", "'ve", "'ll", "'t"}
_PUNCT_LEFT_ATTACH = {"(", "[", "{", "$", "#"}
_SENTENCE_END = {".", "?", "!"}


def _untokenize(tokens: List[str]) -> str:
    pieces: List[str] = []
    for i, tok in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            if tok not in _PUNCT_NO_SPACE and prev not in _PUNCT_LEFT_ATTACH:
                pieces.append(" ")
        pieces.append(tok)
    return "".join(pieces).strip()


def _simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _ensure_numeric_claims(tokens: List[str], claims: List[Claim]) -> None:
    """Add small windowed claims centered on numeric tokens.

    Numbers (dates, quantities, statistics) are disproportionately likely to
    be hallucinated, so we make sure every numeric token is covered by at
    least one claim span, even if sentence segmentation missed it.
    """
    spans = {claim.span for claim in claims}
    for idx, tok in enumerate(tokens):
        if any(ch.isdigit() for ch in tok):
            start = max(0, idx - 3)
            end = min(len(tokens), idx + 4)
            span = (start, end)
            if span in spans:
                continue
            snippet = _untokenize(tokens[start:end])
            if snippet:
                claims.append(Claim(text=snippet, span=span, kind="numeric"))
                spans.add(span)


def extract_claims(text: str, min_tokens: int = 2) -> ClaimSet:
    """Extract claims with token spans for downstream scoring."""
    tokens = _simple_tokenize(text)
    if not tokens:
        stripped = text.strip()
        claims = [Claim(text=stripped, span=(0, 0))] if stripped else []
        return ClaimSet(claims=claims, tokens=tokens)

    spans: List[Tuple[int, int]] = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok in _SENTENCE_END:
            if i - start >= min_tokens:
                spans.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        spans.append((start, len(tokens)))

    claims: List[Claim] = []
    for s, e in spans:
        claim_tokens = tokens[s:e]
        text_span = _untokenize(claim_tokens)
        if text_span:
            claims.append(Claim(text=text_span, span=(s, e), kind="sentence"))

    if not claims:
        claims = [Claim(text=text.strip(), span=(0, len(tokens)))]

    _ensure_numeric_claims(tokens, claims)
    claims.sort(key=lambda c: (c.span[0], c.span[1]))
    return ClaimSet(claims=claims, tokens=tokens)


def batch_extract_claims(texts: List[str]) -> List[ClaimSet]:
    return [extract_claims(t) for t in texts]


def claim_texts(claim_set: ClaimSet) -> List[str]:
    return [claim.text for claim in claim_set.claims]


def claim_spans(claim_set: ClaimSet) -> List[Tuple[int, int]]:
    return [claim.span for claim in claim_set.claims]
