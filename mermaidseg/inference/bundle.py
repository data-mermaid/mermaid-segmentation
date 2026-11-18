"""The self-describing ``model.json`` a run writes at start, and the exported bundle built on it.

A bundle is a directory holding ``model.json`` and ``weights.safetensors``. ``export_bundle``
produces it from a training checkpoint; ``load_bundle`` rebuilds the model from it without the hub.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from mermaidseg.inference.preprocess import CHANNEL_ORDER, INTERPOLATION
from mermaidseg.model.models import MODEL_CLASSES

MODEL_JSON = "model.json"
WEIGHTS = "weights.safetensors"
SCHEMA_VERSION = 1
TASK = "segmentation_cbm"
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


def _require_clean_tree(sidecar: dict) -> None:
    if sidecar.get("lineage", {}).get("git_dirty") is not False:
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
        raise ValueError(f"{checkpoint} config does not match the sidecar's resolved_config")
    key = next(k for k in [*left, *right] if left.get(k) != right.get(k))
    raise ValueError(
        f"{checkpoint} config differs from the sidecar's resolved_config at key {key!r}"
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

    The sidecar ``model.json`` beside the checkpoint supplies the architecture and lineage.

    Raises:
        FileNotFoundError: The sidecar is missing or ``out_dir``'s parent does not exist.
        ValueError: The run's tree was dirty or unknown, ``out_dir`` is not empty, or the
            checkpoint's config differs from the sidecar's ``lineage.resolved_config``.
    """
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    sidecar_path = checkpoint.parent / MODEL_JSON
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    _require_clean_tree(sidecar)
    _require_empty_out_dir(out_dir)
    _require_parent_exists(out_dir)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require_matching_config(
        ckpt["config"], sidecar.get("lineage", {}).get("resolved_config"), checkpoint
    )
    metadata = {
        **sidecar,
        "weights": WEIGHTS,
        "checkpoint": {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"]},
    }
    _write_bundle_dir(out_dir, ckpt["model_state_dict"], metadata)
    return out_dir


@dataclass(frozen=True)
class ModelBundle:
    """A loaded bundle: its ``model.json`` metadata and the model built from it."""

    path: Path
    metadata: dict
    model: nn.Module
    device: torch.device

    @property
    def id2label(self) -> dict[int, str]:
        """Class names by integer id."""
        return {int(k): v for k, v in self.metadata["id2label"].items()}

    @property
    def concept_id2name(self) -> dict[int, str]:
        """Concept channel names by integer id."""
        return {int(k): v for k, v in self.metadata["concept_id2name"].items()}

    @property
    def input_size(self) -> tuple[int, int]:
        """``(height, width)`` the images are resized to."""
        height, width = self.metadata["preprocessing"]["size_hw"]
        return (int(height), int(width))

    @property
    def mean(self) -> tuple[float, ...]:
        """Per-channel normalisation mean."""
        return tuple(self.metadata["preprocessing"]["mean"])

    @property
    def std(self) -> tuple[float, ...]:
        """Per-channel normalisation standard deviation."""
        return tuple(self.metadata["preprocessing"]["std"])

    @property
    def config(self) -> dict:
        """The run's resolved config, from the lineage."""
        return self.metadata["lineage"]["resolved_config"]


def _check_metadata(metadata: dict, path: Path) -> None:
    for key, expected in (("schema_version", SCHEMA_VERSION), ("task", TASK)):
        if metadata.get(key) != expected:
            raise ValueError(f"{path}: {key} {metadata.get(key)!r} is not supported ({expected!r})")
    preprocessing = metadata["preprocessing"]
    for key, expected in (("channel_order", CHANNEL_ORDER), ("interpolation", INTERPOLATION)):
        if preprocessing.get(key) != expected:
            raise ValueError(
                f"{path}: preprocessing.{key} {preprocessing.get(key)!r} is not supported ({expected!r})"
            )


def _build_encoder(encoder_config: dict) -> PreTrainedModel:
    """A randomly initialised float32 encoder from its recorded config; never touches the hub."""
    rest = dict(encoder_config)
    model_type = rest.pop("model_type")
    return AutoModel.from_config(AutoConfig.for_model(model_type, **rest), dtype=torch.float32)


def load_bundle(path: str | Path, device: torch.device | str) -> ModelBundle:
    """Load the bundle directory ``path`` onto ``device`` as an eval-mode model.

    The encoder is built from ``architecture.encoder_config`` and the weights come from
    ``weights.safetensors`` with a strict state-dict match, so no hub access is needed.
    ``concept_value2id`` keys are stringified in JSON: bool keys arrive as ``"True"``/``"False"``.
    The model reads only the taxonomic string keys and the widths of their value lists.

    Raises:
        ValueError: ``schema_version``, ``task``, ``preprocessing.channel_order`` or
            ``preprocessing.interpolation`` is not supported.
        RuntimeError: The weights do not match the architecture's state dict.
    """
    path = Path(path)
    metadata = json.loads((path / MODEL_JSON).read_text())
    _check_metadata(metadata, path)
    architecture = metadata["architecture"]
    kwargs = {k: v for k, v in architecture["kwargs"].items() if k != "encoder_name"}
    kwargs["input_size"] = tuple(kwargs["input_size"])
    model = MODEL_CLASSES[architecture["class"]](
        base_encoder=_build_encoder(architecture["encoder_config"]), **kwargs
    )
    model.load_state_dict(load_file(path / metadata["weights"]), strict=True)
    device = torch.device(device)
    return ModelBundle(path=path, metadata=metadata, model=model.to(device).eval(), device=device)
