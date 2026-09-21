"""hallucination_energy: a factorized, claim-level energy-based hallucination detector.

The package decomposes hallucination risk into four complementary energies —
invariance, evidence, codelength, and geometry — computed per atomic claim,
and fuses them into a single hallucination energy (see
``energies.factorized_energy``). See the repository README for the full
project plan and roadmap from the current linear-fusion baseline to a
conditional, nonlinear factorized energy model (FHEM).
"""

__version__ = "0.1.0"
