#!/usr/bin/env python3
"""CLI entry point for answer generation.

Thin wrapper around ``hallucination_energy.generation.generate``. Dataset
loading, prompt construction, and the accuracy metric registry are all
vendored locally now (``hallucination_energy.data``,
``evaluation.accuracy``) — see ``generation/generate.py``'s module
docstring for what's still deferred (Semantic Entropy / p_ik computation).

``bioasq`` requires a manual dataset download; see
``hallucination_energy/data/loaders.py``.

Usage:
    python scripts/generate_dataset.py --dataset bioasq --num_samples 100 \\
        --model_name Llama-2-7b-chat --num_generations 10 [--use_wandb]
"""
from hallucination_energy.generation.generate import bootstrap, build_arg_parser, main

if __name__ == "__main__":
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown args: {unknown}")
    bootstrap(args)
    main(args)
