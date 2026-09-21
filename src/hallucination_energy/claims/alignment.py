"""Claim-to-evidence alignment (planned).

TODO (see README roadmap / project plan): map each extracted claim to the
retrieved evidence passage(s) that most directly speak to it, beyond the
top-k retrieval already performed in ``retrieval.retriever``. Candidate
approaches: cross-encoder re-ranking per claim, or entity/relation overlap
scoring. Not needed for the v0 energy pipeline, where
``energies.evidence.evidence_energy`` retrieves per-claim directly.
"""
