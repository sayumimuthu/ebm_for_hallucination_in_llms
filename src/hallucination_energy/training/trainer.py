"""Training loop orchestration (planned).

TODO: once ``training.negatives`` grows real counterfactual negative
generation and the fusion model in ``energies.factorized_energy`` grows a
nonlinear/conditional form, this module should own the end-to-end loop:
sample calibration examples -> generate negatives -> compute energy
features -> contrastive update -> periodic conformal re-calibration
(``calibration.conformal``) -> checkpointing to ``outputs/checkpoints``.

For now, ``scripts/train_energy_model.py`` is a thin CLI stub pointing here.
"""
