"""Generate LLM answers (+ per-token log-likelihoods + hidden states) for a
dataset, optionally with Weights & Biases logging.

Merges the repository root's original ``generate_answers.py`` (wandb) and
``generate_answers_nowb.py`` (no-wandb) into one script driven by
``--use_wandb``. All dataset loading, prompt construction, the accuracy
metric registry, model init, p(True), and artifact saving are now handled
by local, vendored modules (``hallucination_energy.data``,
``hallucination_energy.evaluation.accuracy``, ``generation.hf_model``,
``generation.p_true``, ``generation.io_utils``) — this script no longer
imports the external, unvendored ``uncertainty``/``compute_uncertainty_measures``
package.

Fixes carried over from the earlier version of this file, still in effect:

- **No hardcoded HuggingFace token.** ``huggingface_hub.login()`` is only
  called if ``HF_TOKEN`` is set in the environment.
- **No hardcoded** ``CUDA_VISIBLE_DEVICES``; pass ``--cuda_visible_devices``
  or set it in your shell/job script.

Additional fixes made while vendoring the upstream argument parser:

- **``--use_mc_options`` was declared** ``type=bool`` **upstream**, which is
  an argparse trap: any non-empty string (including ``"False"``) parses to
  ``True``. Switched to ``action=argparse.BooleanOptionalAction`` like its
  sibling flags, so ``--no-use_mc_options`` actually works.
- Dropped the upstream ``compute`` argument-parser stage (~20 flags such as
  ``--eval_wandb_runid``, ``--assign_new_wandb_id``) entirely: those only
  configure ``compute_uncertainty_measures.py``, which this script no
  longer calls (see ``--compute_uncertainties`` below).

.. note::
    ``--compute_uncertainties`` (Semantic Entropy, p_ik, etc. via the
    upstream ``compute_uncertainty_measures.py``) now defaults to
    ``False`` and, if set, only logs a warning: that script is
    wandb-run-restore-coupled and is deferred until the Semantic Entropy /
    Semantic Entropy Probes baselines are implemented (see
    ``hallucination_energy/baselines/``). This script still produces
    everything ``scripts/compute_energy_features.py`` needs
    (``*_generations.pkl`` with token log-likelihoods, hidden states, and
    accuracy) independent of that flag.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random

import numpy as np
import torch
from huggingface_hub import login
from tqdm import tqdm

from hallucination_energy.data.loaders import load_ds
from hallucination_energy.data.preprocessing import (
    BRIEF_PROMPTS,
    construct_fewshot_prompt_from_indices,
    get_make_prompt,
    get_reference,
    split_dataset,
)
from hallucination_energy.evaluation.accuracy import get_metric
from hallucination_energy.generation import p_true as p_true_utils
from hallucination_energy.generation.extract_internal_signals import build_answer_record
from hallucination_energy.generation.hf_model import HuggingfaceModel
from hallucination_energy.generation.io_utils import save


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="squad", choices=["squad", "llm", "llm_gpt-3.5", "llm_gpt-4"],
        help="Metric used to assign accuracy to generations.",
    )
    parser.add_argument(
        "--compute_accuracy_at_all_temps", action=argparse.BooleanOptionalAction, default=True,
        help="Compute accuracy at all temperatures or only for the low-temperature (most-likely) generation.",
    )
    parser.add_argument("--experiment_lot", type=str, default="Unnamed Experiment")

    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat")
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--dataset", type=str, default="trivia_qa", choices=["trivia_qa", "squad", "bioasq", "nq", "svamp"],
    )
    parser.add_argument(
        "--ood_train_dataset", type=str, default=None, choices=["trivia_qa", "squad", "bioasq", "nq", "svamp"],
        help="Dataset used only to assemble the few-shot / p_true prompts (out-of-distribution).",
    )
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--num_few_shot", type=int, default=5)
    parser.add_argument("--p_true_num_fewshot", type=int, default=20)
    parser.add_argument("--p_true_hint", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--use_mc_options", default=True, action=argparse.BooleanOptionalAction,
        help="Include multiple-choice options in the question (fixed from upstream's `type=bool`; see module docstring).",
    )
    parser.add_argument("--get_training_set_generations", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--get_training_set_generations_most_likely_only", default=True, action=argparse.BooleanOptionalAction,
        help="Only get the most-likely-answer embedding for the training set (all p_true needs).",
    )
    parser.add_argument("--compute_p_true", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--brief_always", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable_brief", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--brief_prompt", default="default", type=str, choices=list(BRIEF_PROMPTS))
    parser.add_argument("--answerable_only", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--compute_uncertainties", default=False, action=argparse.BooleanOptionalAction,
        help="Deferred: see module docstring. Currently just logs a warning if set.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getenv("SCRATCH_DIR", "."),
        help="Where *_generations.pkl / experiment_details.pkl are written.",
    )

    parser.add_argument(
        "--use_wandb", action="store_true", default=False,
        help="Log this run to Weights & Biases (replaces the separate *_nowb.py script).",
    )
    parser.add_argument(
        "--cuda_visible_devices", default=None,
        help="If set, exported as CUDA_VISIBLE_DEVICES before any CUDA/model import side effects.",
    )
    return parser


def bootstrap(args) -> None:
    """Logging setup, CUDA device pinning, and HF Hub login — shared by
    this module's own ``__main__`` and by ``scripts/generate_dataset.py``,
    so both entry points get the same setup rather than the script-level
    logic being skipped when called as a library function."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # Only attempts a HF Hub login if a token is actually configured (HF_TOKEN
    # env var or prior `huggingface-cli login`); never hardcode a token here.
    # `huggingface_hub.login()` always makes an online call to validate the
    # token (it does not consult HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE), so on
    # a no-internet compute node it's a hard crash rather than a harmless
    # no-op — even though it's unnecessary there: gated models already
    # downloaded to the local cache load fine without re-authenticating.
    offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    if os.environ.get("HF_TOKEN") and not offline:
        login(token=os.environ["HF_TOKEN"])
    elif os.environ.get("HF_TOKEN") and offline:
        logging.info("HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is set: skipping huggingface_hub.login() (it always makes an online call). Relying on the local cache.")


def init_model(args) -> HuggingfaceModel:
    mn = args.model_name
    if any(family in mn.lower() for family in ("llama", "falcon", "mistral", "phi", "gemma")):
        return HuggingfaceModel(
            mn, stop_sequences="default", max_new_tokens=args.model_max_new_tokens,
            cache_dir=os.getenv("HF_HOME"),
        )
    raise ValueError(f"Unknown model_name `{mn}`.")


def main(args):
    if args.use_wandb:
        import wandb  # local import: wandb is an optional dependency of this script only.

    if args.dataset == "svamp":
        if not args.use_context:
            logging.info("Forcing `use_context=True` for svamp dataset.")
            args.use_context = True
    elif args.dataset == "squad":
        if not args.answerable_only:
            logging.info("Forcing `answerable_only=True` for squad dataset.")
            args.answerable_only = True

    if args.compute_uncertainties:
        logging.warning(
            "`--compute_uncertainties` is set, but computing Semantic Entropy / p_ik via the "
            "upstream compute_uncertainty_measures.py has not been vendored yet (see module "
            "docstring). Generations will still be produced and saved."
        )

    experiment_details = {"args": args}
    random.seed(args.random_seed)

    if args.use_wandb:
        wandb.login()
        wandb.init(
            project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
            dir=args.output_dir,
            config=args,
            notes=f"experiment_lot: {args.experiment_lot}",
        )
        logging.info("Finished wandb init.")

    metric = get_metric(args.metric)

    train_dataset, validation_dataset = load_ds(args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            "Using OOD dataset %s to construct few-shot prompts.", args.ood_train_dataset
        )
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if not isinstance(train_dataset, list):
        logging.info("Train dataset: %s", train_dataset)

    answerable_indices, unanswerable_indices = split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, _ = split_dataset(validation_dataset)
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details["prompt_indices"] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = get_make_prompt(args.use_context)
    BRIEF = BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = construct_fewshot_prompt_from_indices(train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details["prompt"] = prompt
    experiment_details["BRIEF"] = BRIEF
    logging.info("Prompt is: %s", prompt)

    # Initialize model.
    model = init_model(args)

    # Initialize prompt for p_true baseline.
    p_true_few_shot_prompt = None
    if args.compute_p_true:
        logging.info(80 * "#")
        logging.info("Constructing few-shot prompt for p_true.")

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model,
            dataset=train_dataset,
            indices=p_true_indices,
            prompt=prompt,
            brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt,
            num_generations=args.num_generations,
            metric=metric,
        )
        if args.use_wandb:
            wandb.config.update({"p_true_num_fewshot": len_p_true}, allow_val_change=True)
            wandb.log(dict(len_p_true=len_p_true))
        experiment_details["p_true_indices"] = p_true_indices
        experiment_details["p_true_responses"] = p_true_responses
        experiment_details["p_true_few_shot_prompt"] = p_true_few_shot_prompt
        logging.info("Finished constructing few-shot prompt for p_true.")
        logging.info(80 * "#")
        logging.info("p_true_few_shot_prompt: %s", p_true_few_shot_prompt)
        logging.info(80 * "#")

    # Start answer generation.
    logging.info(80 * "=")
    logging.info("Generating answers: ")
    logging.info(80 * "=")
    for dataset_split in ["train", "validation"]:
        logging.info(80 * "x")
        logging.info("Starting with dataset_split %s.", dataset_split)
        logging.info(80 * "x")

        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        generations_for_clustering = {}

        if dataset_split == "train":
            if not args.get_training_set_generations:
                logging.info("Skip training data.")
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {"indices": indices}

        if args.num_samples > len(dataset):
            logging.warning("Not enough samples in dataset. Using all %d samples.", len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            example = dataset[index]
            question, context = example["question"], example["context"]
            generations[example["id"]] = {"question": question, "context": context}
            correct_answer = example["answers"]["text"]

            current_input = make_prompt(context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info("Current input: ".ljust(15) + current_input)

            full_responses = []

            # We sample 1 low-temperature answer for accuracy, plus
            # args.num_generations high-temperature answers for invariance/
            # entropy-style measures.
            if dataset_split == "train" and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            most_likely_answer_dict = None
            for i in range(num_generations):
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, hidden_states = model.predict(
                    local_prompt, temperature, return_latent=True
                )

                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                acc = metric(predicted_answer, example, model) if (correct_answer and compute_acc) else 0.0

                most_likely_answer_dict = build_answer_record(predicted_answer, token_log_likelihoods, hidden_states, acc)

                if i == 0:
                    logging.info("Iteration " + str(it) + ":  " + 80 * "#")
                    if args.use_context:
                        logging.info("context: ".ljust(15) + str(context))
                    logging.info("question: ".ljust(15) + question)
                    logging.info("low-t prediction: ".ljust(15) + predicted_answer)
                    logging.info("correct answer: ".ljust(15) + str(correct_answer))
                    logging.info("accuracy: ".ljust(15) + str(acc))

                    accuracies.append(acc)

                    generations[example["id"]].update(
                        {"most_likely_answer": most_likely_answer_dict, "reference": get_reference(example)}
                    )
                    generations_for_clustering.setdefault(example["id"], {})[i] = most_likely_answer_dict
                else:
                    logging.info("high-t prediction ".ljust(15) + str(i) + " : " + predicted_answer)
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, most_likely_answer_dict["embedding"], acc)
                    )
                    generations_for_clustering.setdefault(example["id"], {})[i] = most_likely_answer_dict

            generations[example["id"]]["responses"] = full_responses

            if args.compute_p_true and dataset_split == "validation":
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict["response"],
                    [r[0] for r in full_responses], p_true_few_shot_prompt, hint=args.p_true_hint,
                )
                p_trues.append(p_true)
                logging.info("p_true: %s", p_true)

        save(generations, f"{dataset_split}_generations.pkl", out_dir=args.output_dir)
        save(generations_for_clustering, f"{dataset_split}_generations_for_clustering.pkl", out_dir=args.output_dir)

        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        if args.use_wandb:
            wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == "validation":
            if args.compute_p_true:
                results_dict["uncertainty_measures"] = {
                    "p_false": [1 - p for p in p_trues],
                    "p_false_fixed": [1 - np.exp(p) for p in p_trues],
                }
            save(results_dict, "uncertainty_measures.pkl", out_dir=args.output_dir)

    save(experiment_details, "experiment_details.pkl", out_dir=args.output_dir)
    logging.info("Run complete.")
    del model


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown args: {unknown}")

    bootstrap(cli_args)
    logging.info("Starting new run with args: %s", cli_args)
    logging.info("STARTING `generate`!")
    main(cli_args)
    logging.info("FINISHED `generate`!")
