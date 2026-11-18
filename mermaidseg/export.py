"""Training-side writers of the self-describing ``model.json`` and the exported bundle.

``write_sidecar`` writes the sidecar a run starts with; ``export_bundle`` turns a training
checkpoint into a bundle directory that ``mermaidseg.inference.bundle.load_bundle`` reads.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from mermaidseg.inference.bundle import MODEL_JSON, SCHEMA_VERSION, TASK, WEIGHTS
from mermaidseg.inference.preprocess import CHANNEL_ORDER, INTERPOLATION
from mermaidseg.run_setup import read_checkpoint

# Sections that must not change when a run resumes into an existing sidecar.
_CONTRACT_KEYS = ("architecture", "id2label", "concept_id2name", "preprocessing")


def _jsonable(value: Any) -> Any:
    """``value`` with mapping keys as strings and sequences as lists."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _build_sidecar(
    *,
    run_name: str,
    model_class: str,
    model_kwargs: dict,
    encoder_config: dict,
    id2label: Mapping[int, str],
    concept_id2name: Mapping[int, str],
    input_size: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    lineage: dict,
) -> dict:
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "task": TASK,
            "run_name": run_name,
            "architecture": {
                "class": model_class,
                "kwargs": model_kwargs,
                "encoder_config": encoder_config,
            },
            "id2label": id2label,
            "concept_id2name": concept_id2name,
            "preprocessing": {
                "size_hw": input_size,
                "mean": mean,
                "std": std,
                "channel_order": CHANNEL_ORDER,
                "interpolation": INTERPOLATION,
            },
            "lineage": lineage,
        }
    )


def _check_resume(existing: dict, new: dict) -> None:
    """Raise ValueError naming the contract sections where ``existing`` differs from ``new``."""
    differing = [k for k in _CONTRACT_KEYS if existing.get(k) != new[k]]
    if differing:
        raise ValueError(
            f"{MODEL_JSON} exists with a different {', '.join(differing)}; "
            "a resumed run must keep the contract of its sidecar"
        )


def write_sidecar(
    run_dir: str | Path,
    *,
    run_name: str,
    model_class: str,
    model_kwargs: dict,
    encoder_config: dict,
    id2label: Mapping[int, str],
    concept_id2name: Mapping[int, str],
    input_size: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    lineage: dict,
) -> Path:
    """Write ``model.json`` atomically into ``run_dir`` and return its path.

    An existing sidecar must carry the same architecture, id2label, concept_id2name and
    preprocessing; only its lineage is replaced.

    Raises:
        ValueError: The existing sidecar differs; the message names the differing keys.
    """
    sidecar = _build_sidecar(
        run_name=run_name,
        model_class=model_class,
        model_kwargs=model_kwargs,
        encoder_config=encoder_config,
        id2label=id2label,
        concept_id2name=concept_id2name,
        input_size=input_size,
        mean=mean,
        std=std,
        lineage=lineage,
    )
    path = Path(run_dir) / MODEL_JSON
    if path.is_file():
        _check_resume(json.loads(path.read_text()), sidecar)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(sidecar, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _normalized(value: Any) -> Any:
    """``value`` as it reads back from JSON: string keys, lists for tuples."""
    return json.loads(json.dumps(value))


def _require_clean_tree(lineage: dict) -> None:
    if lineage.get("git_dirty") is not False:
        raise ValueError(
            "the run's git tree was dirty or unknown; deployment artifacts must come from a clean tree"
        )


def _require_empty_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output directory {out_dir} exists and is not empty")


def _require_parent_exists(out_dir: Path) -> None:
    if not out_dir.parent.is_dir():
        raise FileNotFoundError(f"parent directory of {out_dir} does not exist: {out_dir.parent}")


def _require_matching_config(
    checkpoint_config: Any, resolved_config: Any, checkpoint: Path
) -> None:
    """Raise ValueError naming the first top-level key where the two configs differ."""
    left, right = _normalized(checkpoint_config), _normalized(resolved_config)
    if left == right:
        return
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError(f"{checkpoint} config does not match its lineage's resolved_config")
    key = next(k for k in [*left, *right] if left.get(k) != right.get(k))
    raise ValueError(
        f"{checkpoint} config differs from its lineage's resolved_config at key {key!r}"
    )


def _untied_cpu_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """CPU, contiguous copies that share no storage, as safetensors requires."""
    return {k: v.detach().to("cpu").contiguous().clone() for k, v in state_dict.items()}


def _current_umask() -> int:
    """The process umask, read by setting and restoring it."""
    previous = os.umask(0)
    os.umask(previous)
    return previous


def _write_bundle_dir(
    out_dir: Path, state_dict: Mapping[str, torch.Tensor], metadata: dict[str, Any]
) -> None:
    """Write the bundle into a fresh temporary sibling of ``out_dir`` and rename it onto ``out_dir``."""
    staging = Path(
        tempfile.mkdtemp(prefix=out_dir.name + ".", suffix=".partial", dir=out_dir.parent)
    )
    try:
        save_file(_untied_cpu_state(state_dict), staging / WEIGHTS)
        (staging / MODEL_JSON).write_text(json.dumps(metadata, indent=2))
        os.chmod(staging, 0o777 & ~_current_umask())
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_bundle(checkpoint: Path, out_dir: Path) -> Path:
    """Export ``checkpoint`` into a bundle directory at ``out_dir`` and return it.

    The sidecar ``model.json`` beside the checkpoint supplies the architecture, label maps
    and preprocessing; the lineage is the checkpoint's own, from the session that wrote it.

    Raises:
        FileNotFoundError: The sidecar is missing or ``out_dir``'s parent does not exist.
        RuntimeError: The checkpoint cannot be loaded or lacks a required key, ``lineage``
            included; the message names its path.
        ValueError: The checkpoint's tree was dirty or unknown, ``out_dir`` is not empty, or
            the checkpoint's config differs from its ``lineage.resolved_config``.
    """
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    sidecar_path = checkpoint.parent / MODEL_JSON
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    _require_empty_out_dir(out_dir)
    _require_parent_exists(out_dir)
    ckpt = read_checkpoint(
        checkpoint,
        required_keys=("config", "model_state_dict", "epoch", "global_step", "lineage"),
    )
    _require_clean_tree(ckpt["lineage"])
    _require_matching_config(ckpt["config"], ckpt["lineage"].get("resolved_config"), checkpoint)
    metadata = {
        **sidecar,
        "lineage": ckpt["lineage"],
        "weights": WEIGHTS,
        "checkpoint": {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"]},
    }
    _write_bundle_dir(out_dir, ckpt["model_state_dict"], metadata)
    return out_dir
