"""The four claim-level energies and their v0 linear fusion.

- ``invariance``: perturbation/paraphrase stability (unbalanced OT).
- ``evidence``: external retrieval + NLI support.
- ``codelength``: generator token-surprisal.
- ``geometry``: off-manifold representation distance.
- ``normalization``: per-factor calibration-set z-scoring before fusion.
- ``factorized_energy``: v0 linear fusion + the single ``compute_energy_parts``
  entry point (see module docstring for the planned nonlinear upgrade).
"""
