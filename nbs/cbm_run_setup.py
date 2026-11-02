"""Small helpers for ``Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py``.

Kept apart from the script so they can be unit-tested without running training.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

OUTPUT_DIR_ENV = "MERMAIDSEG_OUTPUT_DIR"
CONCEPT_MAPPING_ENV = "MERMAIDSEG_CONCEPT_MAPPING_PATH"
DEFAULT_OUTPUT_DIR = "."
DEFAULT_CONCEPT_MAPPING = Path(__file__).resolve().parent.parent / "configs" / "class_to_concepts.csv"


def resolve_output_dir(flag: str | None, env: Mapping[str, str] | None = None) -> str:
    """Checkpoint root: ``--output-dir`` wins, then ``MERMAIDSEG_OUTPUT_DIR``, then ``.``."""
    env = os.environ if env is None else env
    return flag or env.get(OUTPUT_DIR_ENV) or DEFAULT_OUTPUT_DIR


def resolve_concept_mapping_default(env: Mapping[str, str] | None = None) -> str:
    """Concept CSV used when the training config names none."""
    env = os.environ if env is None else env
    return env.get(CONCEPT_MAPPING_ENV) or str(DEFAULT_CONCEPT_MAPPING)


def logger_id2label(target_id2label: Mapping[int, str]) -> dict[int, str]:
    """The label map the Logger records: class 0 is the ignore index."""
    return {0: "ignore", **target_id2label}


def val_input_spec(
    val_transform: Any, fallback_size: int | Sequence[int] | None = None
) -> tuple[tuple[int, int], tuple[float, ...], tuple[float, ...]]:
    """Return ``(size_hw, mean, std)`` from a validation ``albumentations.Compose``.

    The size comes from its ``Resize``; without one, ``fallback_size`` (the model config
    ``input_size``) is used. ``Normalize`` is required.
    """
    by_name = {type(t).__name__: t for t in getattr(val_transform, "transforms", [])}
    normalize = by_name.get("Normalize")
    if normalize is None:
        raise ValueError("The validation transform has no Normalize; cannot record mean/std.")
    resize = by_name.get("Resize")
    if resize is not None:
        size = (int(resize.height), int(resize.width))
    elif fallback_size is not None:
        size = (
            (int(fallback_size), int(fallback_size))
            if isinstance(fallback_size, int)
            else (int(fallback_size[0]), int(fallback_size[1]))
        )
    else:
        raise ValueError("The validation transform has no Resize and no fallback input_size.")
    return size, tuple(float(v) for v in normalize.mean), tuple(float(v) for v in normalize.std)


def sidecar_kwargs(
    *,
    run_name: str,
    model_cfg: Mapping[str, Any],
    target_id2label: Mapping[int, str],
    concept_id2name: Mapping[Any, str],
    concept_value2id: Mapping[str, Any],
    val_transform: Any,
) -> dict[str, Any]:
    """Keyword arguments for ``mermaidseg.inference.write_sidecar`` (minus ``out_dir``).

    ``input_size`` is the validated size (val transform); ``model_config_input_size`` is the
    model config's ``input_size``, which a later evaluation may swap for another.
    """
    config_size = model_cfg.get("input_size")
    size, mean, std = val_input_spec(val_transform, config_size)
    if config_size is None:
        model_config_size = None
    elif isinstance(config_size, int):
        model_config_size = (config_size, config_size)
    else:
        model_config_size = (int(config_size[0]), int(config_size[1]))
    return {
        "run_name": run_name,
        "model_cfg": model_cfg,
        "id2label": logger_id2label(target_id2label),
        "concept_id2name": concept_id2name,
        "concept_value2id": concept_value2id,
        "input_size": size,
        "model_config_input_size": model_config_size,
        "mean": mean,
        "std": std,
    }


def run_checkpoint_dir(output_dir: str, run_name: str) -> str:
    """Folder where ``Logger.save_model_checkpoint`` writes this run's checkpoints."""
    return os.path.join(output_dir, "model_checkpoints", run_name)
