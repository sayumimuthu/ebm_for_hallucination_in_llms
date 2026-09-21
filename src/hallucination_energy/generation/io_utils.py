"""Save generation artifacts to local disk.

Upstream's ``uncertainty/utils/utils.py`` had a ``save()`` that
unconditionally wrote to ``wandb.run.dir`` — which crashes with
``AttributeError: 'NoneType' object has no attribute 'dir'`` unless
``wandb.init()`` was called first, making it impossible to actually run
generation without wandb despite a separate "*_nowb.py" entry point
existing upstream that never called ``wandb.init()``. This version writes
to a plain local directory by default and only *additionally* mirrors to
the active wandb run if one exists, so ``generation.generate --use_wandb``
and ``generation.generate`` (no flag) both work.
"""
from __future__ import annotations

import os
import pickle
from typing import Any


def save(obj: Any, filename: str, out_dir: str = ".") -> str:
    """Pickle ``obj`` to ``<out_dir>/<filename>``, mirroring to the active
    wandb run (if any). Returns the local path written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    try:
        import wandb

        if wandb.run is not None:
            wandb.save(path)
    except Exception:  # pragma: no cover - optional dependency / no active run
        pass

    return path
