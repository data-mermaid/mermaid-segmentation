"""The self-describing sidecar ``model.json`` a run writes into its run directory at start."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mermaidseg.inference.preprocess import CHANNEL_ORDER, INTERPOLATION

MODEL_JSON = "model.json"
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
