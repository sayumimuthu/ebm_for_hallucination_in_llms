"""Cross-encoder reranking of retrieved passages (planned).

``retrieval.retriever.ContextRetriever`` currently ranks by embedding cosine
similarity only. TODO: add an optional cross-encoder reranking stage here
for higher-precision evidence selection before the evidence energy scores
claim/passage pairs.
"""
