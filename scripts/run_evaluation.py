"""Full evaluation against the baseline comparison set (planned).

TODO: load an ``energy_matrix.npz`` (from ``compute_energy_features.py``)
plus baseline scores (``hallucination_energy.baselines``), and report
AUROC/AUPRC/ECE (``evaluation.metrics``) and selective-risk curves
(``evaluation.selective_prediction``, not yet implemented) per the project
plan's minimum comparison set (section 23) and ablation table (section 24).

Named ``run_evaluation.py``, not ``evaluate.py``: a script literally named
``evaluate.py`` shadows the third-party ``evaluate`` package (used by
``evaluation.accuracy`` for the squad-F1 metric) for the whole process,
because ``python scripts/whatever.py`` puts ``scripts/`` at the front of
``sys.path``. That collision broke ``scripts/generate_dataset.py`` with
``ImportError: cannot import name 'load' from 'evaluate'`` when this file
was still called ``evaluate.py`` — see README/commit notes.
"""

if __name__ == "__main__":
    raise NotImplementedError("Baselines (hallucination_energy/baselines/*) are not implemented yet.")
