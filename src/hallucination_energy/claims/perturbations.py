"""Meaning-preserving claim perturbations for the invariance energy (planned).

The v0 invariance energy (``energies.invariance.invariance_energy``) consumes
paraphrases that already exist as a side effect of multi-sample generation
(high-temperature resamples of the same prompt, see
``generation.generate``). It does not yet generate *targeted* semantic
perturbations of a single claim (paraphrase, back-translation, or
diffusion-style corrupt-and-reconstruct as in the "DiffuTruth" baseline
referenced in the project plan).

TODO: implement dedicated perturbation strategies here (e.g. paraphrase
generation, back-translation, controlled corruption + reconstruction) so
invariance energy can be computed without relying on incidental resamples.
"""
