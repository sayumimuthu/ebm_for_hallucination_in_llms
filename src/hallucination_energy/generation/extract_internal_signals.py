"""Package a single generation's internal signals into the record format
consumed downstream by the energies (``energies.factorized_energy.
compute_energy_parts`` reads ``response`` and ``token_log_likelihoods``;
the embeddings are reserved for the geometry energy's planned learned-density
upgrade, see ``energies.geometry`` module docstring).

Factored out of the inline dict-building duplicated in the original
``generate_answers.py`` / ``generate_answers_nowb.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def build_answer_record(
    predicted_answer: str,
    token_log_likelihoods: List[float],
    hidden_states: Tuple[Optional[Any], Optional[Any], Optional[Any]],
    accuracy: float,
) -> Dict[str, Any]:
    embedding, emb_last_before_gen, emb_before_eos = hidden_states
    return {
        "response": predicted_answer,
        "token_log_likelihoods": token_log_likelihoods,
        "embedding": embedding.cpu() if embedding is not None else None,
        "accuracy": accuracy,
        "emb_last_tok_before_gen": emb_last_before_gen.cpu() if emb_last_before_gen is not None else None,
        "emb_tok_before_eos": emb_before_eos.cpu() if emb_before_eos is not None else None,
    }
