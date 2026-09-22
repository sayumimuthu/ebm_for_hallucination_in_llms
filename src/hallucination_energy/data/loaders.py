"""Dataset loaders, vendored from the upstream Semantic-Entropy-Probes (SEP)
project's ``uncertainty/data/data_utils.py``.

Only the loaders for datasets on this project's benchmark shortlist are
kept: ``squad``, ``trivia_qa``, ``nq``, ``bioasq`` (priority order — see
README), plus ``svamp`` since ``generation.generate`` already special-cases
it. Dropped from upstream: ``med_qa`` and ``record`` (not reachable via the
CLI's dataset choices, and ``record`` requires a local file path that isn't
part of this repo).

``bioasq`` requires a manual download: it is not distributed via the
``datasets`` library. Register at http://participants-area.bioasq.org/datasets/
(training set 11b) and place the file at
``<this file's directory>/bioasq/training11b.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional, Tuple

import datasets


def load_ds(dataset_name: str, seed: int, add_options: Optional[bool] = None) -> Tuple[Any, Any]:
    """Load a (train, validation) dataset pair by name.

    Every example is normalized to ``{"question", "context", "answers":
    {"text": [...]}, "id"}`` so downstream code (``generation.generate``,
    ``data.preprocessing``) doesn't need to special-case dataset schemas.
    """
    train_dataset, validation_dataset = None, None

    if dataset_name == "squad":
        # HF Hub deprecated resolving bare, un-namespaced dataset ids via
        # loading scripts; "squad_v2" alone now 404s (HfUriError). Use the
        # canonical namespaced id.
        dataset = datasets.load_dataset("rajpurkar/squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == "svamp":
        dataset = datasets.load_dataset("ChilleD/SVAMP")
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        def reformat(x):
            return {
                "question": x["Question"],
                "context": x["Body"],
                "type": x["Type"],
                "equation": x["Equation"],
                "id": x["ID"],
                "answers": {"text": [str(x["Answer"])]},
            }

        train_dataset = [reformat(d) for d in train_dataset]
        _validation_dataset = [reformat(d) for d in validation_dataset]
        # Merge training with test set for more samples (matches upstream).
        validation_dataset = _validation_dataset + train_dataset

    elif dataset_name == "nq":
        # Same fix as squad_v2 above: use the namespaced id.
        dataset = datasets.load_dataset("google-research-datasets/nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

        def md5hash(s):
            return str(int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16))

        def reformat(x):
            return {
                "question": x["question"] + "?",
                "answers": {"text": x["answer"]},
                "context": "",
                "id": md5hash(str(x["question"])),
            }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")["unmodified"]
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    elif dataset_name == "bioasq":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "bioasq", "training11b.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "bioasq requires a manual download (not distributed via the `datasets` "
                "library). Register at http://participants-area.bioasq.org/datasets/ "
                f"(training set 11b) and place the file at: {path}"
            )
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {"question": [], "answers": [], "id": []}

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if isinstance(question["exact_answer"], list):
                exact_answers = [
                    ans[0] if isinstance(ans, list) else ans for ans in question["exact_answer"]
                ]
            else:
                exact_answers = [question["exact_answer"]]
            dataset_dict["answers"].append(
                {"text": exact_answers, "answer_start": [0] * len(exact_answers)}
            )
            dataset_dict["id"].append(question["id"])

        dataset_dict["context"] = [None] * len(dataset_dict["id"])
        dataset = datasets.Dataset.from_dict(dataset_dict)

        # 80/20 test/train: only a small slice is needed for few-shot prompt
        # construction; the bulk goes to validation (matches upstream).
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]
        logging.info("Loaded bioasq: %d train, %d validation.", len(train_dataset), len(validation_dataset))

    else:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            "Supported: squad, trivia_qa, nq, bioasq, svamp."
        )

    return train_dataset, validation_dataset
