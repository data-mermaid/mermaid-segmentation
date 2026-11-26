"""The only writer of training checkpoints, and the one place that names their directory.

Serialization: ``torch.save`` (pickle) of the full training state, which a resume reads back
with ``weights_only=True``.
"""

import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mermaidseg.config import to_plain_dict
from mermaidseg.model.meta import MetaModel

logger = logging.getLogger(__name__)


def checkpoint_dir(output_dir: str | Path, run_name: str) -> Path:
    """``<output_dir>/model_checkpoints/<run_name>``, where one run keeps its files."""
    return Path(output_dir) / "model_checkpoints" / run_name


def prune_rolling_checkpoints(
    directory: str | Path, keep_name: str, prefix: str = "model_preempt"
) -> None:
    """Remove other rolling preempt checkpoints in ``directory``.

    Interval checkpoints (``model_step*``, ``model_epoch*``) use other prefixes and
    are left in place. ``.partial`` files from an interrupted atomic save are removed
    too. ``keep_name`` is the checkpoint that was just renamed into place.
    """
    if not prefix or not os.path.isdir(directory):
        return
    for entry in os.listdir(directory):
        if entry == keep_name or not entry.startswith(prefix):
            continue
        path = os.path.join(directory, entry)
        try:
            os.remove(path)
        except OSError:
            logger.warning("Failed to remove old rolling checkpoint %s", path)


def _to_python(value: Any) -> Any:
    """``value`` as Python floats and lists, which ``torch.load(weights_only=True)`` accepts."""
    if isinstance(value, (np.ndarray, np.generic, torch.Tensor)):
        return value.tolist()
    return value


def write_checkpoint(
    path: str | Path,
    meta_model: MetaModel,
    *,
    config: Any,
    lineage: dict[str, Any],
    epoch: int,
    epoch_complete: bool,
    metrics: Mapping[str, Any],
    atomic: bool = False,
) -> None:
    """Write the training state of ``meta_model`` to ``path``.

    ``epoch_complete`` is false for a mid-epoch step checkpoint so a resume continues that
    epoch instead of skipping it. ``lineage`` is stored in every checkpoint so an export
    reads the lineage of the session that wrote it.

    ``atomic=True`` writes ``<path>.partial`` and renames it into place, so a kill during
    ``torch.save`` does not leave a truncated file at ``path``. The rolling preempt
    checkpoints use this. Interval checkpoints keep the direct write.

    The live module stays on its device. A detached CPU copy of the state dict is saved
    so a DDP reducer keeps the parameter storage it was built with.
    """
    path = Path(path)
    checkpoint: dict[str, Any] = {
        "config": to_plain_dict(config),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in meta_model.model.state_dict().items()
        },
        "optimizer_state_dict": meta_model.optimizer.state_dict(),
        "epoch": epoch,
        "epoch_complete": bool(epoch_complete),
        "global_step": int(meta_model.global_step),
        "warmup_iters_completed": int(meta_model._warmup_iters_completed),
        "timestamp": time.strftime("%Y%m%d%H%M%S"),
        "metrics": {key: _to_python(value) for key, value in metrics.items()},
        "scheduler_state_dict": meta_model.scheduler.state_dict(),
        "lineage": lineage,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        partial_path = path.with_name(path.name + ".partial")
        torch.save(checkpoint, partial_path)
        os.replace(partial_path, path)
    else:
        if path.exists():
            logger.warning(
                "Checkpoint file already exists and will be overwritten: %s. "
                "Use a different output dir or run name when resuming training "
                "to avoid silently overwriting prior checkpoints.",
                path,
            )
        torch.save(checkpoint, path)
    logger.info("Checkpoint saved locally: %s", path)
