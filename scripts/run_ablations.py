#!/usr/bin/env python3
"""Ablation runner: drop each energy factor / interaction / hard-negatives
in turn and re-evaluate (planned; see project plan section 24's ablation
table: -invariance, -evidence, -codelength, -geometry, no interactions,
fixed weights, no hard negatives, no conditional gating).

TODO: implement once ``scripts/train_energy_model.py`` / ``scripts/run_evaluation.py``
are functional; this script should just loop over ablation configs and call
those two.
"""

if __name__ == "__main__":
    raise NotImplementedError("Depends on scripts/train_energy_model.py and scripts/run_evaluation.py.")
