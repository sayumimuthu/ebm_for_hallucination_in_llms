"""Baseline uncertainty/energy detectors for comparison.

None of these existed in the original prototype. They implement the
minimum comparison set in the project plan (section 23): predictive
entropy (``predictive_entropy.py``), Semantic Entropy / Semantic Entropy
Probes (``semantic_entropy.py``), and Semantic Energy
(``semantic_energy.py``, a best-effort reimplementation — see that
module's docstring for exactly which parts follow the plan's spec versus
this project's own design choice). Spilled Energy, HalluField, DiffuTruth,
and HalluSAE remain external methods to benchmark against via their own
repos/reimplementations rather than being vendored here — see
``scripts/run_evaluation.py``.
"""
