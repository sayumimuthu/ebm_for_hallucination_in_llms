"""Semantic Entropy baseline (planned).

TODO: cluster the ``num_generations`` resampled answers by bidirectional
entailment (reusing the upstream SEP package's clustering/NLI utilities,
see ``retrieval.verifier.default_verifier`` for the same entailment-model
fallback pattern) and compute entropy over cluster assignments. Needed for
the plan's "confidently wrong" / "unfamiliar but correct" case studies.
"""
