"""Claim-level dataset construction (planned).

TODO: build the ``N x 4`` energy-parts + label dataset used by
``evaluation.diagnostics`` and ``training.contrastive`` directly from a
directory of generation pickles (the output of
``scripts/generate_dataset.py``) plus ground-truth accuracy labels, instead
of the ad hoc loop currently in ``scripts/compute_energy_features.py``.
"""
