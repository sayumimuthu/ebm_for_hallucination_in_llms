"""Model interface, vendored from the upstream SEP project's
``uncertainty/models/base_model.py``.

``huggingface_models.py`` originally imported ``BaseModel`` and
``STOP_SEQUENCES`` from ``uncertainty.models.base_model``. This module
vendors that interface directly (rather than the earlier placeholder
guess this file held before the upstream source was available), so
``generation.hf_model`` has no hidden dependency on an external install.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

STOP_SEQUENCES: List[str] = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]


class BaseModel(ABC):
    """Interface expected by ``generation.generate``."""

    stop_sequences: Optional[List[str]]
    token_limit: int

    @abstractmethod
    def predict(
        self,
        input_data: str,
        temperature: float,
        return_full: bool = False,
        return_latent: bool = False,
    ) -> Tuple[str, List[float], Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
        """Generate a continuation for ``input_data``.

        Returns ``(answer_text, token_log_likelihoods, hidden_states)``
        where ``hidden_states`` is a 3-tuple of
        ``(last_token_embedding, second_last_token_embedding,
        last_input_token_embedding)``, the latter two only populated when
        ``return_latent=True``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_p_true(self, input_data: str) -> float:
        """Log-probability the model answers "A" (True) for a p_true-style
        prompt ending in a binary True/False question."""
        raise NotImplementedError

    def get_character_start_stop_indices(self, input_data_offset: int, answer: str) -> Tuple[int, int]:
        """Remove any output following (and including) a stop_sequence.

        Some outputs start with newlines (unfortunately). We strip these, in
        order to ensure generations with greater-than-zero length.
        """
        start_index = input_data_offset

        newline = "\n"
        while answer[start_index:].startswith(newline):
            start_index += len(newline)

        stop_index = len(answer)
        for word in self.stop_sequences:
            index = answer[start_index:].find(word)
            if index != -1 and index + start_index < stop_index:
                stop_index = index + start_index

        return start_index, stop_index
