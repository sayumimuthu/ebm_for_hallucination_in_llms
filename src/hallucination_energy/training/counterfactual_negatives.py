"""Targeted counterfactual negatives for contrastive fusion training.

Replaces the naive claim-shuffling/numeric-jitter negatives in
``training.negatives`` (kept there, unchanged, as the "no hard negatives"
ablation baseline — see the project plan's ablation table) with
minimally-edited, plausible-but-**false** counterfactuals per claim,
covering the categories from the project plan (section 14):

===================================  ==========================================
Category                             Implemented as
===================================  ==========================================
entity substitution                  :func:`substitute_entities`
relation/entity swap                 (a special case of entity substitution)
relation inversion                   :func:`invert_relation`
numerical perturbation               reused from ``training.negatives.jitter_numeric_text``
date/temporal change                 :func:`perturb_date_or_temporal`
negation                             :func:`negate_claim`
causal reversal                      **deferred** — see module note below
attribution swap                     **deferred** — see module note below
fabricated supporting detail         **deferred** — see module note below
===================================  ==========================================

The three deferred categories need real semantic-role/discourse
understanding (which clause is the cause vs. effect; who is the reported
speaker/agent; what invented detail would be *plausible*) that a
regex/dependency-parse heuristic can't reliably provide. Per this
project's own design discussion, they're left for a follow-up LLM-prompted
pass rather than shipped as a weak rule-based approximation.

Every strategy degrades gracefully: with spaCy (``_optional.get_spacy_model``)
available, entity substitution and relation inversion use real NER /
dependency parses; without it, both fall back to a capitalized-word-span
heuristic, matching the rest of this package's optional-dependency
philosophy (see ``_optional.py``, ``energies/embeddings.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from hallucination_energy._optional import get_spacy_model
from hallucination_energy.claims.extraction import ClaimSet, claim_texts
from hallucination_energy.training.negatives import jitter_numeric_text, NUMBER_RE

# --------------------------------------------------------------------------
# Entity pool: a background corpus's entities, bucketed by type, so
# substitution swaps in something of the *same kind* (a person for a
# person, a place for a place) rather than a random token.
# --------------------------------------------------------------------------

_CAP_SPAN_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}\b")


@dataclass
class EntityPool:
    """``label -> [surface strings]``. With spaCy, ``label`` is a real NER
    tag (PERSON, GPE, ORG, DATE, ...); without it, everything falls under
    the single generic label ``"ENT"`` (capitalized-word-span heuristic)."""

    by_label: Dict[str, List[str]] = field(default_factory=dict)

    def sample_alternative(self, label: str, exclude: str, rng: np.random.Generator) -> Optional[str]:
        candidates = [e for e in self.by_label.get(label, []) if e.lower() != exclude.lower()]
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]


def _extract_entities_spacy(text: str, nlp) -> List[Tuple[str, str, int, int]]:
    doc = nlp(text)
    return [(ent.label_, ent.text, ent.start_char, ent.end_char) for ent in doc.ents]


def _extract_entities_regex(text: str) -> List[Tuple[str, str, int, int]]:
    """Capitalized multi-word spans, skipping one at position 0 (very
    likely just sentence-initial capitalization, not a real entity)."""
    return [
        ("ENT", m.group(0), m.start(), m.end())
        for m in _CAP_SPAN_RE.finditer(text)
        if m.start() != 0
    ]


def _extract_entities(text: str) -> List[Tuple[str, str, int, int]]:
    nlp = get_spacy_model()
    if nlp is not None:
        try:
            ents = _extract_entities_spacy(text, nlp)
            if ents:
                return ents
        except Exception:
            pass
    return _extract_entities_regex(text)


def build_entity_pool(texts: List[str]) -> EntityPool:
    """Build a substitution pool from a background corpus (e.g. the same
    reference/train split used to fit the geometry energy's trusted-claim
    density — see ``scripts/compute_energy_features.py``)."""
    pool = EntityPool()
    for text in texts:
        for label, surface, _start, _end in _extract_entities(text):
            bucket = pool.by_label.setdefault(label, [])
            if surface not in bucket:
                bucket.append(surface)
    return pool


# --------------------------------------------------------------------------
# Per-category perturbations. Each takes (text, rng, entity_pool) for a
# uniform call signature (see CATEGORIES below) and returns a perturbed
# string, or None if this category found nothing to perturb in this text.
# --------------------------------------------------------------------------


def substitute_entities(
    text: str, pool: EntityPool, rng: np.random.Generator, max_substitutions: int = 1
) -> Optional[str]:
    """Swap up to ``max_substitutions`` entities for a different one of the
    same type from ``pool`` (e.g. a different PERSON, a different GPE)."""
    ents = _extract_entities(text)
    if not ents:
        return None
    order = rng.permutation(len(ents))
    made = 0
    result = text
    # Apply substitutions right-to-left so earlier character offsets stay valid.
    for idx in sorted(order.tolist(), key=lambda i: -ents[i][2]):
        if made >= max_substitutions:
            break
        label, surface, start, end = ents[idx]
        replacement = pool.sample_alternative(label, surface, rng)
        if replacement:
            result = result[:start] + replacement + result[end:]
            made += 1
    return result if made > 0 else None


_NEGATION_CUES = [" not ", "n't ", " never ", " no longer ", " cannot "]
_AUX_RE = re.compile(r"\b(is|are|was|were|has|have|had|does|did|can|could|will|would|should|must)\b", re.IGNORECASE)


def negate_claim(text: str, rng: np.random.Generator = None, pool: Optional[EntityPool] = None) -> Optional[str]:
    """Flip polarity: remove an existing negation cue, or insert one after
    the first auxiliary/modal/copula verb, or (last resort) prefix
    "It is not true that ..."."""
    del rng, pool
    lowered = text.lower()
    for cue in _NEGATION_CUES:
        idx = lowered.find(cue)
        if idx != -1:
            return (text[:idx] + " " + text[idx + len(cue) :]).replace("  ", " ").strip()
    m = _AUX_RE.search(text)
    if m:
        insert_at = m.end()
        return text[:insert_at] + " not" + text[insert_at:]
    stripped = text.strip()
    if not stripped:
        return None
    return "It is not true that " + stripped[0].lower() + stripped[1:]


def _invert_relation_spacy(text: str, nlp) -> Optional[str]:
    doc = nlp(text)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    obj = next((c for c in root.children if c.dep_ in ("dobj", "attr", "oprd")), None)
    if obj is None:
        for c in root.children:
            if c.dep_ == "prep":
                obj = next((gc for gc in c.children if gc.dep_ == "pobj"), None)
                if obj is not None:
                    break
    if subj is None or obj is None:
        return None
    subj_span = doc[subj.left_edge.i : subj.right_edge.i + 1]
    obj_span = doc[obj.left_edge.i : obj.right_edge.i + 1]
    if subj_span.start >= obj_span.start:
        return None  # unexpected order; skip rather than risk a garbled swap
    return (
        text[: subj_span.start_char]
        + obj_span.text
        + text[subj_span.end_char : obj_span.start_char]
        + subj_span.text
        + text[obj_span.end_char :]
    )


def _invert_relation_regex(text: str) -> Optional[str]:
    spans = [(m.start(), m.end(), m.group(0)) for m in _CAP_SPAN_RE.finditer(text) if m.start() != 0]
    if len(spans) < 2:
        return None
    first, last = spans[0], spans[-1]
    if first[2].lower() == last[2].lower() or first[1] > last[0]:
        return None
    return text[: first[0]] + last[2] + text[first[1] : last[0]] + first[2] + text[last[1] :]


def invert_relation(text: str, rng: np.random.Generator = None, pool: Optional[EntityPool] = None) -> Optional[str]:
    """Swap the subject and object around the main verb (spaCy dependency
    parse), or the first/last capitalized spans as a fallback."""
    del rng, pool
    nlp = get_spacy_model()
    if nlp is not None:
        try:
            result = _invert_relation_spacy(text, nlp)
            if result:
                return result
        except Exception:
            pass
    return _invert_relation_regex(text)


_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b")


def perturb_date_or_temporal(
    text: str, rng: np.random.Generator, pool: Optional[EntityPool] = None
) -> Optional[str]:
    """Swap a month name for a different one, or shift a year by 1-50
    (distinct from ``jitter_numeric_text``'s small-percentage jitter — this
    is deliberately a large, unambiguous temporal change)."""
    del pool
    m = _MONTH_RE.search(text)
    if m:
        choices = [mo for mo in _MONTHS if mo != m.group(0)]
        return text[: m.start()] + choices[int(rng.integers(0, len(choices)))] + text[m.end() :]
    m = _YEAR_RE.search(text)
    if m:
        year = int(m.group(0))
        sign = 1 if rng.random() < 0.5 else -1
        new_year = max(1000, year + sign * int(rng.integers(1, 51)))
        return text[: m.start()] + str(new_year) + text[m.end() :]
    return None


def _numerical_perturbation(text: str, rng: np.random.Generator, pool: Optional[EntityPool] = None) -> Optional[str]:
    del pool
    if not NUMBER_RE.search(text):
        return None
    return jitter_numeric_text(text, scale=0.3, rng=rng)


def _entity_substitution(text: str, rng: np.random.Generator, pool: Optional[EntityPool] = None) -> Optional[str]:
    if pool is None:
        return None
    return substitute_entities(text, pool, rng)


CounterfactualFn = Callable[[str, np.random.Generator, Optional[EntityPool]], Optional[str]]

CATEGORIES: List[Tuple[str, CounterfactualFn]] = [
    ("entity_substitution", _entity_substitution),
    ("negation", negate_claim),
    ("date_temporal", perturb_date_or_temporal),
    ("relation_inversion", invert_relation),
    ("numerical_perturbation", _numerical_perturbation),
]


def _apply_to_one_claim(
    texts: List[str], fn: CounterfactualFn, pool: Optional[EntityPool], rng: np.random.Generator
) -> Optional[str]:
    """Try ``fn`` against each claim (in random order), substituting the
    first successful perturbation back into the full (unshuffled) claim
    sequence — the negative is a normal-looking answer except for one
    deliberately-wrong claim, not shuffled/interleaved text."""
    order = rng.permutation(len(texts))
    for idx in order:
        idx = int(idx)
        perturbed = fn(texts[idx], rng, pool)
        if perturbed and perturbed.strip() and perturbed != texts[idx]:
            new_texts = list(texts)
            new_texts[idx] = perturbed
            return " ".join(new_texts)
    return None


def generate_counterfactual_negatives(
    claim_set: ClaimSet,
    entity_pool: EntityPool,
    rng: np.random.Generator,
    num_negatives: int = 2,
) -> List[Tuple[str, str]]:
    """Return up to ``num_negatives`` ``(category_name, negative_text)``
    pairs: minimally-edited, plausibly-false counterfactuals of
    ``claim_set``'s joined text. Categories are tried in random order per
    call; a category that finds nothing to perturb in any claim is simply
    skipped (returns fewer than ``num_negatives`` rather than a low-quality
    negative). Callers that need an exact count (e.g. NCE training, which
    expects a fixed number of negatives per example) should pad any
    shortfall with ``training.negatives.corrupt_claim_set`` — see
    ``scripts/compute_energy_features.py``.
    """
    texts = claim_texts(claim_set)
    if not texts:
        return []
    results: List[Tuple[str, str]] = []
    order = rng.permutation(len(CATEGORIES))
    for i in order:
        if len(results) >= num_negatives:
            break
        name, fn = CATEGORIES[int(i)]
        made = _apply_to_one_claim(texts, fn, entity_pool, rng)
        if made is not None:
            results.append((name, made))
    return results
