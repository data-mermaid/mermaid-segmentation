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

from mermaidseg.data.config import TAXONOMY_NAME
from mermaidseg.inference.bundle import (
    MODEL_JSON,
    SCHEMA_VERSION,
    TASK,
    WEIGHTS,
    WEIGHTS_SHA256,
    sha256_file,
)
from mermaidseg.inference.preprocess import CHANNEL_ORDER, INTERPOLATION
from mermaidseg.run_setup import read_checkpoint
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ManifestNotFound, PinMismatch
from segdata_spec.git import on_remote
from segdata_spec.manifest import Pin
from segdata_spec.sync import check_published_manifest

LABEL_SPACE_DIR = "label_space"

# Sections that must not change when a run resumes into an existing sidecar.
_CONTRACT_KEYS = ("architecture", "id2label", "concept_id2name", "preprocessing", "label_space")


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
    label_space: Mapping[str, str],
    lineage: dict,
) -> dict:
    kwargs = {k: v for k, v in model_kwargs.items() if k != "encoder_name"}
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "task": TASK,
            "run_name": run_name,
            "architecture": {
                "class": model_class,
                "encoder_name": model_kwargs["encoder_name"],
                "kwargs": kwargs,
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
            "label_space": label_space,
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
    label_space: Mapping[str, str],
    lineage: dict,
) -> Path:
    """Write ``model.json`` atomically into ``run_dir`` and return its path.

    ``label_space`` maps each file of the run's label space copy to its sha256. An existing
    sidecar must carry the same architecture, id2label, concept_id2name, preprocessing and
    label_space; only its lineage is replaced.

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
        label_space=label_space,
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


def _require_clean_tree(lineage: dict) -> None:
    if lineage.get("git_dirty") is not False:
        raise ValueError(
            "the run's git tree was dirty; deployment artifacts must come from a clean tree"
        )


def _pins(lineage: dict) -> list[tuple[str, str, str, str]]:
    """The ``(kind, name, version, manifest_sha256)`` of every pin in ``lineage['data_version']``."""
    data_version = lineage["data_version"]
    taxonomy = data_version["taxonomy"]
    pins = [
        ("datasets", name, entry["version"], entry["manifest_sha256"])
        for name, entry in data_version["datasets"].items()
    ]
    pins.append(("taxonomy", TAXONOMY_NAME, taxonomy["version"], taxonomy["manifest_sha256"]))
    return pins


def _require_pinned_manifests(lineage: dict, client: Any) -> None:
    """Raise ValueError unless every pinned manifest is in the bucket with its recorded sha256."""
    for kind, name, version, expected in _pins(lineage):
        pin = Pin(name=name, version=version, manifest_sha256=expected)
        try:
            check_published_manifest(client, DATASET_BUCKET, pin, taxonomy=(kind == "taxonomy"))
        except ManifestNotFound as exc:
            raise ValueError(
                f"{kind} manifest for {name!r} version {version!r} is missing: {exc}"
            ) from exc
        except PinMismatch as exc:
            raise ValueError(f"{kind} manifest for {name!r} version {version!r}: {exc}") from exc


def _require_pushed_commit(lineage: dict, repo_dir: Path) -> None:
    """Raise ValueError unless the lineage's commit is on a remote-tracking branch."""
    sha = lineage["git_sha"]
    if not on_remote(repo_dir, sha):
        raise ValueError(
            f"commit {sha} is not on any remote branch; "
            "run `git fetch` and, if it is still missing, push it"
        )


def _require_empty_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"output directory {out_dir} exists and is not empty")


def require_label_space(source: Path, listed: Mapping[str, str]) -> None:
    """Raise ValueError unless ``source`` holds every file of ``listed`` with its sha256."""
    if not source.is_dir():
        raise ValueError(f"label space directory not found: {source}")
    for name, expected in listed.items():
        path = source / name
        if name != Path(name).name or not path.is_file():
            raise ValueError(f"label space file {name!r} listed in {MODEL_JSON} is missing: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"label space file {name!r} differs from the sha256 in {MODEL_JSON}")


def _require_parent_exists(out_dir: Path) -> None:
    if not out_dir.parent.is_dir():
        raise FileNotFoundError(f"parent directory of {out_dir} does not exist: {out_dir.parent}")


def _untied_cpu_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """CPU, contiguous copies that share no storage, as safetensors requires."""
    return {k: v.detach().to("cpu").contiguous().clone() for k, v in state_dict.items()}


def _current_umask() -> int:
    """The process umask, read by setting and restoring it."""
    previous = os.umask(0)
    os.umask(previous)
    return previous


def _write_bundle_dir(
    out_dir: Path,
    state_dict: Mapping[str, torch.Tensor],
    metadata: dict[str, Any],
    label_space_source: Path,
) -> None:
    """Write the bundle into a fresh temporary sibling of ``out_dir`` and rename it onto ``out_dir``.

    The bundle's ``label_space/`` holds the files that ``metadata['label_space']`` lists,
    copied from ``label_space_source``.
    """
    staging = Path(
        tempfile.mkdtemp(prefix=out_dir.name + ".", suffix=".partial", dir=out_dir.parent)
    )
    try:
        save_file(_untied_cpu_state(state_dict), staging / WEIGHTS)
        digest = sha256_file(staging / WEIGHTS)
        (staging / MODEL_JSON).write_text(
            json.dumps({**metadata, WEIGHTS_SHA256: digest}, indent=2)
        )
        (staging / LABEL_SPACE_DIR).mkdir()
        for name in metadata["label_space"]:
            shutil.copy2(label_space_source / name, staging / LABEL_SPACE_DIR / name)
        os.chmod(staging, 0o777 & ~_current_umask())
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_bundle(
    checkpoint: Path, out_dir: Path, *, s3_client: Any, repo_dir: Path | None = None
) -> Path:
    """Export ``checkpoint`` into a bundle directory at ``out_dir`` and return it.

    The sidecar ``model.json`` beside the checkpoint supplies the architecture, label maps
    and preprocessing, and the ``label_space/`` directory beside it supplies the label space
    files that the bundle carries; the lineage is the checkpoint's own, from the session that
    wrote it. ``s3_client`` reads the pinned manifests; ``repo_dir`` (default: the working
    directory) is the git checkout whose remote-tracking branches must contain the lineage's
    commit.

    Raises:
        FileNotFoundError: The sidecar is missing, ``out_dir``'s parent does not exist, or a
            file the sidecar lists is missing from the label space directory.
        RuntimeError: The checkpoint cannot be loaded or lacks a required key, ``lineage``
            included; the message names its path.
        ValueError: The checkpoint's tree was dirty, ``out_dir`` is not empty, a pinned
            manifest is missing from the bucket or has another sha256, or the lineage's commit
            is not on any remote-tracking branch of ``repo_dir``.
    """
    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    sidecar_path = checkpoint.parent / MODEL_JSON
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    _require_empty_out_dir(out_dir)
    _require_parent_exists(out_dir)
    label_space_source = checkpoint.parent / LABEL_SPACE_DIR
    ckpt = read_checkpoint(
        checkpoint,
        required_keys=("config", "model_state_dict", "epoch", "global_step", "lineage"),
    )
    _require_clean_tree(ckpt["lineage"])
    _require_pushed_commit(ckpt["lineage"], Path.cwd() if repo_dir is None else Path(repo_dir))
    _require_pinned_manifests(ckpt["lineage"], s3_client)
    metadata = {
        **sidecar,
        "lineage": ckpt["lineage"],
        "weights": WEIGHTS,
        "checkpoint": {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"]},
    }
    _write_bundle_dir(out_dir, ckpt["model_state_dict"], metadata, label_space_source)
    return out_dir
