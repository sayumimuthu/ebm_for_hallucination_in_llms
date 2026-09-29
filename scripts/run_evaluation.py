#!/usr/bin/env python3
"""Compare FHEM's composite score against the baseline set (project plan
section 23): predictive entropy, Semantic Entropy, and Semantic Energy
(``hallucination_energy.baselines``).

Reuses a prior ``scripts/compute_energy_features.py`` run's saved
artifacts (``energy_matrix.npz``'s raw ``X``/``y``/``ids``, plus
``summary.json``'s normalizer stats and learned fusion weights) to
reconstruct FHEM's exact per-example composite score, rather than
recomputing the four energies from scratch here. Baseline scores are
computed directly from the same ``*_generations.pkl`` used for that run.

For each method: AUROC with a bootstrap 95% CI (``evaluation.metrics``,
same machinery as the fusion-model addenda) and AURAC (area under the
thresholded-accuracy / risk-coverage curve, ``evaluation.selective_prediction``
— assumes ``--accuracy_threshold 1.0`` was used for the energy-features
run, so the binary hallucination label doubles as a binary accuracy value;
see the ``accuracies`` construction below).

Usage:
    python scripts/run_evaluation.py \\
        --generations_path outputs/squad_run500/validation_generations.pkl \\
        --energy_features_dir outputs/squad_run500/energy_features_linear \\
        --out_dir outputs/squad_run500/baseline_comparison

Named ``run_evaluation.py``, not ``evaluate.py``: a script literally named
``evaluate.py`` shadows the third-party ``evaluate`` package for the whole
process (``scripts/`` is prepended to ``sys.path``), which broke
``scripts/generate_dataset.py`` with ``ImportError: cannot import name
'load' from 'evaluate'`` when this file was still called ``evaluate.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

from hallucination_energy.baselines.predictive_entropy import predictive_entropy_scores
from hallucination_energy.baselines.semantic_energy import semantic_energy_scores
from hallucination_energy.baselines.semantic_entropy import semantic_entropy_scores
from hallucination_energy.energies.factorized_energy import ENERGY_ORDER
from hallucination_energy.evaluation.metrics import bootstrap_auroc_diff_ci, bootstrap_ci
from hallucination_energy.evaluation.selective_prediction import area_under_thresholded_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generations_path", required=True, help="Same *_generations.pkl used for --energy_features_dir.")
    p.add_argument(
        "--energy_features_dir", required=True,
        help="Output dir of a prior compute_energy_features.py run (needs energy_matrix.npz + summary.json, "
             "plus linear_weights.npz or mlp_weights.pt depending on --fusion_model).",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_fhem_scores(energy_features_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Reconstruct FHEM's per-example composite score, exactly matching
    what compute_energy_features.py trained, from its saved artifacts.
    Returns (scores, labels, ids), aligned."""
    with open(os.path.join(energy_features_dir, "summary.json")) as fh:
        summary = json.load(fh)

    npz = np.load(os.path.join(energy_features_dir, "energy_matrix.npz"), allow_pickle=True)
    X, y, ids = npz["X"], npz["y"], [str(i) for i in npz["ids"]]

    normalizer = summary["energy_normalizer"]
    means = np.array([normalizer["means"][k] for k in ENERGY_ORDER])
    stds = np.array([normalizer["stds"][k] for k in ENERGY_ORDER])
    X_norm = (X - means) / stds

    fusion_model = summary.get("fusion_model", "linear")
    if fusion_model == "linear":
        w = np.array(summary["learned_weights"])
        b = summary["learned_bias"]
        scores = X_norm @ w + b
    else:
        import torch

        from hallucination_energy.training.nonlinear_fusion import MLPEnergy, ResidualMLPEnergy, mlp_energy_score

        hidden_dim = summary["mlp_hidden_dim"]
        model_cls = ResidualMLPEnergy if fusion_model == "residual_mlp" else MLPEnergy
        model = model_cls(input_dim=X.shape[1], hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(os.path.join(energy_features_dir, "mlp_weights.pt")))
        model.eval()
        scores = mlp_energy_score(model, X_norm.astype(np.float32))

    return np.asarray(scores, dtype=np.float64), y, ids


def _evaluate_one(name: str, scores_by_id: Dict[str, float], ids: List[str], labels: np.ndarray, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    """AUROC + bootstrap CI + AURAC for one method, restricted to the ids
    it actually has a score for (baselines can have narrower coverage than
    FHEM, e.g. semantic_energy needs token_logsumexp)."""
    mask = np.array([tid in scores_by_id for tid in ids])
    n_covered = int(mask.sum())
    if n_covered < 2 or len(np.unique(labels[mask])) < 2:
        return {"n_covered": n_covered, "note": "too few covered/labeled examples to score"}

    y = labels[mask]
    s = np.array([scores_by_id[tid] for tid, m in zip(ids, mask) if m])
    accuracies = 1 - y  # valid when --accuracy_threshold 1.0 was used for the energy-features run (binary correct/incorrect)

    return {
        "n_covered": n_covered,
        "auroc": bootstrap_ci(y, s, n_bootstrap=n_bootstrap, seed=seed),
        "aurac": area_under_thresholded_accuracy(accuracies, s),
    }


def _fhem_vs_baseline(
    fhem_by_id: Dict[str, float], baseline_by_id: Dict[str, float],
    ids: List[str], labels: np.ndarray, n_bootstrap: int, seed: int,
) -> Dict[str, Any]:
    """Paired bootstrap AUROC difference (FHEM minus baseline), restricted
    to ids both methods actually cover -- the correct tool for "is FHEM
    significantly better," which overlapping marginal CIs cannot answer
    (see the nonlinear-fusion addendum's Section 4 for why: two marginal
    intervals can overlap even when the paired, same-examples difference
    is consistently one-signed)."""
    id_to_index = {tid: i for i, tid in enumerate(ids)}
    common = [tid for tid in ids if tid in fhem_by_id and tid in baseline_by_id]
    idx = [id_to_index[t] for t in common]
    if len(common) < 2 or len(np.unique(labels[idx])) < 2:
        return {"n_common": len(common), "note": "too few common covered/labeled examples to compare"}
    y = labels[idx]
    a = np.array([fhem_by_id[t] for t in common])
    b = np.array([baseline_by_id[t] for t in common])
    return {"n_common": len(common), **bootstrap_auroc_diff_ci(y, a, b, n_bootstrap=n_bootstrap, seed=seed)}


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.generations_path, "rb") as fh:
        generations: Dict[str, Any] = pickle.load(fh)

    fhem_scores, labels, ids = _load_fhem_scores(args.energy_features_dir)
    fhem_scores_by_id = dict(zip(ids, fhem_scores))

    baseline_scores_by_id = {
        "predictive_entropy": predictive_entropy_scores(generations),
        "semantic_entropy": semantic_entropy_scores(generations),
        "semantic_energy": semantic_energy_scores(generations),
    }

    results = {
        "fhem": _evaluate_one("fhem", fhem_scores_by_id, ids, labels, args.n_bootstrap, args.seed),
    }
    for name, scores_by_id in baseline_scores_by_id.items():
        results[name] = _evaluate_one(name, scores_by_id, ids, labels, args.n_bootstrap, args.seed)
        results[name]["fhem_vs_this"] = _fhem_vs_baseline(
            fhem_scores_by_id, scores_by_id, ids, labels, args.n_bootstrap, args.seed
        )

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
