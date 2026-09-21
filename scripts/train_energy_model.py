#!/usr/bin/env python3
"""Train the fused energy model (planned).

v0 (linear NCE fusion over naive negatives) is already runnable via
``scripts/compute_energy_features.py``. This script is reserved for the
planned conditional/nonlinear fusion model with targeted counterfactual
negatives, once ``training.trainer`` and ``training.negatives`` grow real
implementations (see project plan sections 11-15 and those modules' TODOs).
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Use scripts/compute_energy_features.py for the v0 linear-fusion baseline. "
        "This script is a placeholder for the nonlinear/conditional energy model "
        "(see hallucination_energy/training/trainer.py)."
    )
