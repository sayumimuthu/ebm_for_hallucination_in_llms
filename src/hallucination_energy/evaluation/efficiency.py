"""Compute-cost accounting for the cost-aware adaptive inference cascade
(planned, RQ4; see project plan section 18).

TODO: instrument each energy's cost (extra generations for invariance,
retrieval + NLI calls for evidence, which are free/cheap for codelength and
geometry since they reuse signals already captured during generation) and
implement the cheap-first escalation cascade: codelength + geometry -> if
ambiguous, invariance -> if still ambiguous, evidence retrieval.
"""
