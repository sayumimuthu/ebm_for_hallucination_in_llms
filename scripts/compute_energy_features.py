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
from hallucination_energy.energies.normalization import EnergyNormalizer
from hallucination_energy.calibration.conformal import conformal_threshold
from hallucination_energy.evaluation.diagnostics import (
    composite_auroc,
    energy_correlation_matrix,
    model_composite_auroc,
    per_factor_auroc,
)
from hallucination_energy.retrieval.retriever import ContextRetriever, collect_docs_from_example
from hallucination_energy.retrieval.verifier import default_verifier
from hallucination_energy.training.contrastive import train_linear_nce
from hallucination_energy.training.counterfactual_negatives import (
    build_entity_pool,
    generate_counterfactual_negatives,
)
from hallucination_energy.training.negatives import corrupt_claim_set, mix_claim_sets
from hallucination_energy.training.nonlinear_fusion import mlp_energy_score, train_mlp_nce


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
    p.add_argument("--num_negatives", type=int, default=2, help="Negatives per example for NCE training.")
    p.add_argument(
        "--negative_strategy", choices=["counterfactual", "naive"], default="counterfactual",
        help=(
            "'counterfactual' (default): targeted, minimally-edited negatives (entity "
            "substitution, negation, relation inversion, date/numerical perturbation — see "
            "training.counterfactual_negatives), padded with the naive strategy if a claim set "
            "doesn't yield enough. 'naive': the original claim-shuffling/numeric-jitter "
            "negatives only (training.negatives) — kept for the 'no hard negatives' ablation."
        ),
    )
    p.add_argument(
        "--normalizer_ref_size", type=int, default=100,
        help=(
            "Number of reference-split examples used to fit the per-factor z-score "
            "EnergyNormalizer applied before NCE training and composite scoring. Fit on the "
            "reference split (never the validation split being scored/trained on below) for the "
            "same reason --reference_generations_path is separate: fitting it on the split "
            "being evaluated would leak that split's energy distribution into the weights "
            "later scored against it."
        ),
    )
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level.")
    p.add_argument("--accuracy_threshold", type=float, default=1.0, help="Accuracy >= this counts as 'trusted' for geometry/codelength fitting.")
    p.add_argument(
        "--fusion_model", choices=["linear", "mlp"], default="linear",
        help=(
            "'linear' (default): v0 E(e) = w^T e + b (training.contrastive.train_linear_nce). "
            "'mlp': v1 nonlinear E(e) = MLP(e) (training.nonlinear_fusion.train_mlp_nce), same "
            "NCE objective and 4 normalized energies — tests whether representing interactions "
            "among the four factors beats their best linear combination."
        ),
    )
    p.add_argument("--mlp_hidden_dim", type=int, default=16, help="Hidden width for --fusion_model mlp.")
    p.add_argument("--nce_steps", type=int, default=150)
    p.add_argument("--nce_lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _collect_negative_texts(
    claim_set, entity_pool, args: argparse.Namespace, rng: np.random.Generator,
    validation_generations: Dict[str, Any], keys: List[str],
) -> List[str]:
    """Up to ``args.num_negatives`` negative response texts for one example.

    'counterfactual' strategy tries targeted, minimally-edited negatives
    first (see training.counterfactual_negatives); any shortfall (a
    category found nothing to perturb) is padded with the naive
    claim-shuffling/mixing strategy so NCE training always gets a fixed
    negative count per example. 'naive' strategy skips straight to that
    padding step, reproducing the original v0 behavior for the "no hard
    negatives" ablation.
    """
    neg_texts: List[str] = []
    if args.negative_strategy == "counterfactual" and entity_pool is not None:
        neg_texts.extend(
            text for _category, text in
            generate_counterfactual_negatives(claim_set, entity_pool, rng, num_negatives=args.num_negatives)
        )

    attempts = 0
    max_attempts = args.num_negatives * 4 + 4
    while len(neg_texts) < args.num_negatives and attempts < max_attempts:
        attempts += 1
        corrupt_text = corrupt_claim_set(claim_set, rng=rng)
        if corrupt_text and corrupt_text not in neg_texts:
            neg_texts.append(corrupt_text)
            continue
        if len(keys) > 1:
            partner_ex = validation_generations[keys[int(rng.integers(0, len(keys)))]]
            partner_claims = extract_claims(partner_ex.get("most_likely_answer", {}).get("response", ""))
            mix_text = mix_claim_sets(claim_set, partner_claims, rng=rng)
            if mix_text and mix_text not in neg_texts:
                neg_texts.append(mix_text)
                continue
        break  # no more distinct negatives producible; accept a shortfall

    return neg_texts[: args.num_negatives]


def _collect_calibration_parts(
    generations: Dict[str, Any],
    density_model: Any,
    density_meta: Dict[str, Any],
    surprisal_stats: Dict[str, float],
    context_corpus: List[str],
    sinkhorn_params: Dict[str, float],
    max_examples: int,
) -> List[Dict[str, float]]:
    """Raw {inv, ev, cl, geo} energy dicts for up to ``max_examples``
    examples of ``generations`` — the building block for fitting
    ``EnergyNormalizer`` in ``main`` (on the reference split, and,
    conditionally, on a validation-split fallback — see there)."""
    calib_keys = list(generations.keys())[:max_examples]
    calib_parts: List[Dict[str, float]] = []
    for tid in calib_keys:
        ex = generations[tid]
        mla = ex.get("most_likely_answer", {})
        if not mla.get("token_log_likelihoods"):
            continue
        paraphrases = [r[0] for r in ex.get("responses", [])[:5] if isinstance(r, (list, tuple)) and r]
        local_docs = collect_docs_from_example(ex, max_docs=8) or context_corpus[:8]
        retriever = ContextRetriever(local_docs)
        parts, _ = compute_energy_parts(
            ex.get("question", ""), mla, paraphrases, retriever, default_verifier,
            density_model, density_meta, surprisal_stats,
            sinkhorn_params=sinkhorn_params, retriever_top_k=5, evidence_tau=0.2,
        )
        calib_parts.append(parts)
    return calib_parts


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

    # Entity pool for counterfactual entity substitution, built from the
    # same "trusted" reference corpus used to fit the geometry density —
    # substituting in an entity that actually appears in verified-true
    # answers keeps the negative plausible rather than absurd.
    entity_pool = build_entity_pool(trusted_claim_texts) if args.negative_strategy == "counterfactual" else None

    sinkhorn_params = {"reg": 0.1, "tau": 0.7, "quantile": 0.9, "metric": "cosine", "max_claims": 32}
    default_weights = {"inv": 1.0, "ev": 1.0, "cl": 1.0, "geo": 1.0}

    reference_calib_parts = _collect_calibration_parts(
        reference_generations, density_model, density_meta, surprisal_stats,
        context_corpus, sinkhorn_params, args.normalizer_ref_size,
    )
    normalizer = EnergyNormalizer.fit(reference_calib_parts)
    if normalizer.degenerate_keys:
        # A factor with ~zero variance on the reference split (observed in
        # practice: invariance energy, when the reference split lacks
        # paraphrase samples) can't be scaled fairly from that split alone —
        # leaving it at std=1 avoids exploding it (the original bug here),
        # but silences it relative to every properly-scaled peer instead,
        # which is its own distortion. Patch just those factors from a
        # sample of the validation split (only computed when actually
        # needed, to avoid doubling runtime in the common case).
        logging.warning(
            "Factors %s had ~zero variance across %d reference-split examples; refitting "
            "their scale from a %d-example sample of the validation split instead.",
            sorted(normalizer.degenerate_keys), len(reference_calib_parts), args.normalizer_ref_size,
        )
        fallback_calib_parts = _collect_calibration_parts(
            validation_generations, density_model, density_meta, surprisal_stats,
            context_corpus, sinkhorn_params, args.normalizer_ref_size,
        )
        normalizer = EnergyNormalizer.fit_with_fallback(reference_calib_parts, fallback_calib_parts)

    keys = list(validation_generations.keys())
    max_calib = min(len(keys), args.num_calib)

    energy_rows: List[np.ndarray] = []
    normalized_energy_rows: List[np.ndarray] = []
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
        normalized_energy_rows.append(energy_parts_to_vector(normalizer.transform(parts)))

        # NCE training and composite scoring operate on normalized (z-scored)
        # energies, not raw ones — see _fit_energy_normalizer. Diagnostics
        # above (energy_matrix/per_factor_auroc/conformal_threshold) stay on
        # the raw scale, which is what they're meant to describe.
        pos_vec = energy_parts_to_vector(normalizer.transform(parts))
        neg_texts = _collect_negative_texts(
            claim_set, entity_pool, args, rng, validation_generations, keys,
        )
        neg_vectors: List[np.ndarray] = []
        for neg_text in neg_texts:
            neg_answer = {"response": neg_text, "token_log_likelihoods": mla.get("token_log_likelihoods", [])}
            neg_parts, _ = compute_energy_parts(
                ex.get("question", ""), neg_answer, paraphrases, retriever, default_verifier,
                density_model, density_meta, surprisal_stats,
                sinkhorn_params=sinkhorn_params, retriever_top_k=5, evidence_tau=0.2, use_claim_spans=False,
            )
            neg_vectors.append(energy_parts_to_vector(normalizer.transform(neg_parts)))
        if len(neg_vectors) == args.num_negatives:
            feature_pos.append(pos_vec)
            feature_negs.append(neg_vectors)

    energy_matrix = np.stack(energy_rows) if energy_rows else np.zeros((0, 4), dtype=np.float32)
    normalized_energy_matrix = (
        np.stack(normalized_energy_rows) if normalized_energy_rows else np.zeros((0, 4), dtype=np.float32)
    )
    labels = np.asarray(label_rows, dtype=np.int64)
    np.savez(os.path.join(args.out_dir, "energy_matrix.npz"), X=energy_matrix, y=labels)

    tau = conformal_threshold(calib_energies, alpha=args.alpha) if calib_energies else float("inf")

    summary = {
        "conformal_threshold": tau, "alpha": args.alpha, "n_calib": len(calib_energies),
        "energy_normalizer": {
            "means": normalizer.means, "stds": normalizer.stds,
            "degenerate_keys": sorted(normalizer.degenerate_keys),
        },
    }
    if energy_matrix.shape[0] >= 2:
        summary["correlation_matrix"] = energy_correlation_matrix(energy_matrix).tolist()
    if energy_matrix.shape[0] > 0 and len(np.unique(labels)) > 1:
        summary["per_factor_auroc"] = per_factor_auroc(energy_matrix, labels)

    if feature_pos and feature_negs:
        pos_array = np.stack(feature_pos)
        neg_array = np.stack(feature_negs)

        if args.fusion_model == "mlp":
            model, final_loss = train_mlp_nce(
                pos_array, neg_array, hidden_dim=args.mlp_hidden_dim,
                steps=args.nce_steps, lr=args.nce_lr, l2=1e-3, seed=args.seed,
            )
            import torch  # local import: only needed for this branch

            torch.save(model.state_dict(), os.path.join(args.out_dir, "mlp_weights.pt"))
            summary["fusion_model"] = "mlp"
            summary["mlp_hidden_dim"] = args.mlp_hidden_dim
            summary["mlp_num_params"] = sum(p.numel() for p in model.parameters())
            summary["final_nce_loss"] = final_loss
            score_fn = lambda x: mlp_energy_score(model, x)  # noqa: E731
        else:
            w, b = train_linear_nce(pos_array, neg_array, steps=args.nce_steps, lr=args.nce_lr, l2=1e-3)
            np.savez(os.path.join(args.out_dir, "linear_weights.npz"), w=w, b=b)
            summary["fusion_model"] = "linear"
            summary["learned_weights"] = w.tolist()
            summary["learned_bias"] = float(b)
            score_fn = lambda x: x @ w + b  # noqa: E731

        # Does the NCE-learned fusion actually improve real hallucination
        # detection, or did the contrastive objective (trained on synthetic
        # negatives) just latch onto a factor that doesn't generalize? Score
        # the learned fusion — and an equal-weight linear reference — against
        # the real validation labels, not the synthetic NCE task.
        if normalized_energy_matrix.shape[0] > 0 and len(np.unique(labels)) > 1:
            summary["composite_auroc_learned"] = model_composite_auroc(score_fn, normalized_energy_matrix, labels)
            summary["composite_auroc_equal_weights"] = composite_auroc(
                normalized_energy_matrix, labels, np.ones(normalized_energy_matrix.shape[1])
            )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "correlation_matrix"}, indent=2))


if __name__ == "__main__":
    main()
