"""Small helpers for ``scripts/train.py``.

Kept apart from the script so they can be unit-tested without running training.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import albumentations as A

logger = logging.getLogger(__name__)

OUTPUT_DIR_ENV = "MERMAIDSEG_OUTPUT_DIR"
CONCEPT_MAPPING_ENV = "MERMAIDSEG_CONCEPT_MAPPING_PATH"
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs")
DEFAULT_CONCEPT_MAPPING = _REPO_ROOT / "configs" / "class_to_concepts.csv"


def resolve_output_dir(flag: str | None, env: Mapping[str, str] | None = None) -> str:
    """Checkpoint root: ``--output-dir`` wins, then ``MERMAIDSEG_OUTPUT_DIR``, then ``<repo>/outputs``."""
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

    The size comes from its ``Resize``; without any spatial transform, ``fallback_size`` (the
    model config ``input_size``) is used. Any other spatial transform (SmallestMaxSize,
    CenterCrop, ...) raises, as its output size is not recorded here. ``Normalize`` is required.
    """
    transforms = list(getattr(val_transform, "transforms", []))
    by_name = {type(t).__name__: t for t in transforms}
    normalize = by_name.get("Normalize")
    if normalize is None:
        raise ValueError("The validation transform has no Normalize; cannot record mean/std.")
    unknown = [
        type(t).__name__
        for t in transforms
        if isinstance(t, A.DualTransform) and not isinstance(t, A.Resize)
    ]
    if unknown:
        raise ValueError(
            f"The validation transform has spatial transforms {unknown} whose output size "
            "cannot be recorded; only Resize is supported."
        )
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
    val_transforms: Sequence[Any],
) -> dict[str, Any]:
    """Keyword arguments for ``mermaidseg.inference.write_sidecar`` (minus ``out_dir``).

    ``val_transforms`` holds every validation dataset's transform; they must agree on size and
    normalisation. ``input_size`` is the validated size; ``model_config_input_size`` is the
    model config's ``input_size``, which a later evaluation may swap for another.
    """
    config_size = model_cfg.get("input_size")
    specs = {val_input_spec(t, config_size) for t in val_transforms}
    if not specs:
        raise ValueError("No validation transforms given; cannot record the input spec.")
    if len(specs) > 1:
        raise ValueError(
            f"Validation datasets differ in size/mean/std ({sorted(specs)}); one sidecar "
            "cannot describe them all."
        )
    size, mean, std = specs.pop()
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


def write_run_sidecar(
    out_dir: str | Path, *, model_cfg: Mapping[str, Any], sidecar_kwargs: Mapping[str, Any]
) -> Path | None:
    """Write the run's ``model.json``, or warn and return None for a model class with no bundle."""
    from mermaidseg.inference import write_sidecar
    from mermaidseg.inference.bundle import TRAINED_CLASSES

    name = model_cfg.get("name")
    if name not in TRAINED_CLASSES:
        logger.warning(
            "Model class %r cannot be exported as a bundle; skipping model.json for %s.",
            name,
            out_dir,
        )
        return None
    return write_sidecar(out_dir, **sidecar_kwargs)


def check_wandb_auth() -> None:
    """Confirm wandb is authenticated in this environment.

    Checks for a resolved API key (via ``WANDB_API_KEY``, ``wandb login``, or
    ``~/.netrc``) without starting a run. Call before
    ``Logger(..., enable_wandb=True)`` so a missing login surfaces immediately
    instead of mid-training.

    Raises:
        RuntimeError: If no wandb API key can be resolved.
    """
    import wandb

    api_key = wandb.api.api_key
    if not api_key:
        raise RuntimeError(
            "wandb is not authenticated. Run `wandb login` in a terminal, "
            "set WANDB_API_KEY, or call wandb.login(key=...) in a notebook cell."
        )
    print(f"wandb authenticated (API key ...{api_key[-4:]})")


def check_env_wandb() -> None:
    """Validate the environment for wandb-backed, AWS-free training runs.

    Prints the resolved wandb / HF / cache-offline settings and confirms wandb
    authentication (via ``check_wandb_auth``). Use this at the top of training
    scripts instead of ``check_env`` / ``check_aws_session`` when tracking to
    wandb and reading data from the pre-warmed local cache.

    Raises:
        RuntimeError: If wandb cannot resolve an API key.
    """
    print(f"WANDB_PROJECT       : {os.getenv('WANDB_PROJECT', '(not set — using config default)')}")
    print(f"WANDB_ENTITY        : {os.getenv('WANDB_ENTITY', '(not set — using default entity)')}")
    print(f"WANDB_DIR           : {os.getenv('WANDB_DIR', '(not set — using ./wandb)')}")
    print(f"HF_TOKEN            : {'set' if os.getenv('HF_TOKEN') else '(not set)'}")
    offline = os.getenv("MERMAIDSEG_S3_OFFLINE")
    print(
        "MERMAIDSEG_S3_OFFLINE: "
        + (f"{offline} (cache misses will raise)" if offline else "(not set — S3 fallback enabled)")
    )
    check_wandb_auth()
