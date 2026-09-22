#!/usr/bin/env python3
"""Compute the four energies for a set of generations, calibrate a v0
linear fusion via contrastive (NCE) training over naive negatives, and
write out the resulting energy matrix + conformal threshold.

This generalizes the ad hoc, hardcoded example at the end of ``ebm.ipynb``
(cells with a fixed ``run_id``/``wandb_run_ids``/``run_subdir``) into a
reusable CLI: point it at a ``*_generations.pkl`` file produced by
``scripts/generate_dataset.py`` (or ``generation.generate`` directly), no
hardcoded paths or run IDs.

Usage:
    python scripts/compute_energy_features.py \\
        --generations_path outputs/validation_generations.pkl \\
        --reference_generations_path outputs/train_generations.pkl \\
        --out_dir outputs/metrics/run1 \\
        --num_calib 80 --num_negatives 2 --alpha 0.1

``--reference_generations_path`` should be a DIFFERENT split than
``--generations_path`` (e.g. the ``train_generations.pkl`` produced by the
same ``generate.py`` run) — it's used only to fit the geometry energy's
"trusted claim" density and the codelength surprisal stats. Omitting it
falls back to reusing ``--generations_path`` for both, which is circular
(see the runtime warning this prints) and should only be used for a quick
smoke test, not a real diagnostic.

Output (in ``--out_dir``):
    - energy_matrix.npz: {"X": N x 4 energy matrix, "y": accuracy labels}
    - linear_weights.npz: {"w": learned NCE weights, "b": bias} (if trained)
    - summary.json: conformal threshold + per-factor correlations/AUROC
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from hallucination_energy.claims.extraction import extract_claims
from hallucination_energy.energies.codelength import calibrate_surprisal_stats
from hallucination_energy.energies.factorized_energy import (
    compute_energy_parts,
    composite_energy,
    energy_parts_to_vector,
)
from hallucination_energy.energies.geometry import fit_geometry_density
from hallucination_energy.calibration.conformal import conformal_threshold
from hallucination_energy.evaluation.diagnostics import energy_correlation_matrix, per_factor_auroc
from hallucination_energy.retrieval.retriever import ContextRetriever, collect_docs_from_example
from hallucination_energy.retrieval.verifier import default_verifier
from hallucination_energy.training.contrastive import train_linear_nce
from hallucination_energy.training.negatives import corrupt_claim_set, mix_claim_sets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generations_path", required=True, help="Path to a *_generations.pkl file (scored/calibrated on).")
    p.add_argument(
        "--reference_generations_path", default=None,
        help=(
            "Path to a SEPARATE *_generations.pkl (e.g. train_generations.pkl from the same "
            "generate.py run) used only to fit the 'trusted claim' geometry density and the "
            "codelength surprisal stats. If omitted, falls back to reusing --generations_path "
            "for both, which is fitting the reference distributions on the same examples you "
            "then score/evaluate — this is circular for the geometry energy in particular "
            "(its AUROC will look artificially perfect) and is only intended as a quick "
            "smoke-test fallback, not a real diagnostic."
        ),
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_calib", type=int, default=80, help="Max number of examples to use for calibration/training.")
    p.add_argument("--num_negatives", type=int, default=2, help="Naive negatives per example (see training.negatives).")
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
    p.add_argument("--accuracy_threshold", type=float, default=1.0, help="Accuracy >= this counts as 'trusted' for geometry/codelength fitting.")
    p.add_argument("--nce_steps", type=int, default=150)
    p.add_argument("--nce_lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.generations_path, "rb") as fh:
        validation_generations: Dict[str, Any] = pickle.load(fh)

    if args.reference_generations_path:
        with open(args.reference_generations_path, "rb") as fh:
            reference_generations: Dict[str, Any] = pickle.load(fh)
    else:
        logging.warning(
            "--reference_generations_path not given: fitting the geometry density and "
            "surprisal stats on the SAME examples being scored below (%s). This is circular "
            "for the geometry energy in particular — its per-factor AUROC will look "
            "artificially close to 1.0 regardless of any real signal. Pass "
            "--reference_generations_path pointing at a separate split (e.g. "
            "train_generations.pkl from the same generate.py run) for a trustworthy result.",
            args.generations_path,
        )
        reference_generations = validation_generations

    surprisal_stats = calibrate_surprisal_stats(reference_generations, min_accuracy=args.accuracy_threshold)

    trusted_claim_texts: List[str] = []
    context_corpus: List[str] = []
    for ex in reference_generations.values():
        mla = ex.get("most_likely_answer", {})
        if mla.get("accuracy", 0.0) >= args.accuracy_threshold and isinstance(mla.get("response"), str):
            trusted_claim_texts.extend(c.text for c in extract_claims(mla["response"]).claims)
        context_corpus.extend(collect_docs_from_example(ex, max_docs=3))

    density_model, density_meta = fit_geometry_density(trusted_claim_texts, n_components=12)
    context_retriever = ContextRetriever(context_corpus)

    sinkhorn_params = {"reg": 0.1, "tau": 0.7, "quantile": 0.9, "metric": "cosine", "max_claims": 32}
    default_weights = {"inv": 1.0, "ev": 1.0, "cl": 1.0, "geo": 1.0}

    keys = list(validation_generations.keys())
    max_calib = min(len(keys), args.num_calib)

    energy_rows: List[np.ndarray] = []
    label_rows: List[int] = []
    feature_pos: List[np.ndarray] = []
    feature_negs: List[List[np.ndarray]] = []
    calib_energies: List[float] = []

    for idx in range(max_calib):
        tid = keys[idx]
        ex = validation_generations[tid]
        mla = ex.get("most_likely_answer", {})
        if not mla.get("token_log_likelihoods"):
            continue
        paraphrases = [r[0] for r in ex.get("responses", [])[:5] if isinstance(r, (list, tuple)) and r]
        local_docs = collect_docs_from_example(ex, max_docs=8) or context_corpus[:8]
        retriever = ContextRetriever(local_docs)

        parts, claim_set = compute_energy_parts(
            ex.get("question", ""), mla, paraphrases, retriever, default_verifier,
            density_model, density_meta, surprisal_stats,
            sinkhorn_params=sinkhorn_params, retriever_top_k=5, evidence_tau=0.2,
        )
        energy = composite_energy(default_weights, parts)
        calib_energies.append(energy)

        # Label: 1 = hallucinated (incorrect), 0 = factual (correct); adapt
        # to your dataset's accuracy metric semantics.
        label_rows.append(int(mla.get("accuracy", 0.0) < args.accuracy_threshold))
        energy_rows.append(energy_parts_to_vector(parts))

        # v0 naive negatives (see training.negatives docstring for caveats).
        pos_vec = energy_parts_to_vector(parts)
        neg_vectors: List[np.ndarray] = []
        corrupt_text = corrupt_claim_set(claim_set, rng=rng)
        if corrupt_text:
            corrupt_answer = {"response": corrupt_text, "token_log_likelihoods": mla.get("token_log_likelihoods", [])}
            corrupt_parts, _ = compute_energy_parts(
                ex.get("question", ""), corrupt_answer, paraphrases, retriever, default_verifier,
                density_model, density_meta, surprisal_stats,
                sinkhorn_params=sinkhorn_params, retriever_top_k=5, evidence_tau=0.2, use_claim_spans=False,
            )
            neg_vectors.append(energy_parts_to_vector(corrupt_parts))
        if len(keys) > 1:
            partner_ex = validation_generations[keys[int(rng.integers(0, len(keys)))]]
            partner_claims = extract_claims(partner_ex.get("most_likely_answer", {}).get("response", ""))
            mix_text = mix_claim_sets(claim_set, partner_claims, rng=rng)
            if mix_text:
                mix_answer = {"response": mix_text, "token_log_likelihoods": mla.get("token_log_likelihoods", [])}
                mix_parts, _ = compute_energy_parts(
                    ex.get("question", ""), mix_answer, paraphrases, retriever, default_verifier,
                    density_model, density_meta, surprisal_stats,
                    sinkhorn_params=sinkhorn_params, retriever_top_k=5, evidence_tau=0.2, use_claim_spans=False,
                )
                neg_vectors.append(energy_parts_to_vector(mix_parts))
        if len(neg_vectors) == args.num_negatives:
            feature_pos.append(pos_vec)
            feature_negs.append(neg_vectors)

    energy_matrix = np.stack(energy_rows) if energy_rows else np.zeros((0, 4), dtype=np.float32)
    labels = np.asarray(label_rows, dtype=np.int64)
    np.savez(os.path.join(args.out_dir, "energy_matrix.npz"), X=energy_matrix, y=labels)

    tau = conformal_threshold(calib_energies, alpha=args.alpha) if calib_energies else float("inf")

    summary = {"conformal_threshold": tau, "alpha": args.alpha, "n_calib": len(calib_energies)}
    if energy_matrix.shape[0] >= 2:
        summary["correlation_matrix"] = energy_correlation_matrix(energy_matrix).tolist()
    if energy_matrix.shape[0] > 0 and len(np.unique(labels)) > 1:
        summary["per_factor_auroc"] = per_factor_auroc(energy_matrix, labels)

    if feature_pos and feature_negs:
        pos_array = np.stack(feature_pos)
        neg_array = np.stack(feature_negs)
        w, b = train_linear_nce(pos_array, neg_array, steps=args.nce_steps, lr=args.nce_lr, l2=1e-3)
        np.savez(os.path.join(args.out_dir, "linear_weights.npz"), w=w, b=b)
        summary["learned_weights"] = w.tolist()
        summary["learned_bias"] = float(b)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "correlation_matrix"}, indent=2))


if __name__ == "__main__":
    main()
