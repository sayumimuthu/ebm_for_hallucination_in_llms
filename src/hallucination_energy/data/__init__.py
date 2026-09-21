"""Dataset loading (``loaders.py``) and prompt construction
(``preprocessing.py``), vendored from the upstream Semantic-Entropy-Probes
(SEP) project's ``uncertainty/data`` and ``uncertainty/utils`` modules.
``generation.generate`` no longer depends on the external, unvendored
``uncertainty``/``semantic_uncertainty`` package for these pieces.

``claim_dataset.py`` (building the N x 4 energy-parts dataset for training/
diagnostics) is still planned — see that module's docstring.
"""
