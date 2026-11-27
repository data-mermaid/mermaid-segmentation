"""Registry and model construction shared by the training pipeline and the evaluation suite.

Every run reads its concept channels and target classes from the committed label space that
``training.label_space`` names, so a run trained on part of the data (a data-size profile) and
an evaluation that only builds val splits use the same layout.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from mermaidseg.config import to_plain_dict
from mermaidseg.data.base import VersionDataset
from mermaidseg.data.config import DataSection
from mermaidseg.dataset_reconciliation.artifact_meta import file_sha1
from mermaidseg.dataset_reconciliation.concept_removal import load_concept_removal
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    ConceptSourceMask,
    load_concept_source_mask,
)
from mermaidseg.dataset_reconciliation.label_space import (
    META_YAML,
    SOURCE_LABELS_CSV,
    LabelSpace,
    label_keys_sha1,
    read_label_space,
)
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.io import REPO_ROOT, load_section, resolve_repo_path
from mermaidseg.lineage import data_version
from mermaidseg.model.meta import MetaModel

DEFAULT_SCHEMA_CONFIG = REPO_ROOT / "configs" / "data_config.yaml"

# Keys `write_checkpoint` writes that a weights-only load and a resume read.
MODEL_CHECKPOINT_KEYS = ("model_state_dict", "lineage")
RESUME_CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "warmup_iters_completed",
    "global_step",
    "epoch",
    "epoch_complete",
    "lineage",
)


def concept_mapping_path(training_cfg: Any) -> str:
    """The resolved ``concept_mapping_path`` of the training config."""
    return str(resolve_repo_path(training_cfg.concept_mapping_path))


def concept_removal_list(training_cfg: Any, mapping_path: str | Path) -> list[str]:
    """Channel names listed by ``training.concept_removal`` (a CSV path), or none."""
    path = training_cfg["concept_removal"]
    if not path:
        return []
    return load_concept_removal(resolve_repo_path(path), concepts_csv=mapping_path)


def _check_input_sha1(
    meta: Mapping[str, Any], training_cfg: Any, key: str, meta_key: str, directory: Path
) -> None:
    """Raise ValueError unless the file ``training_cfg[key]`` names has the sha1 ``meta`` records."""
    value = training_cfg[key]
    current = file_sha1(resolve_repo_path(value)) if value else None
    if current != meta.get(meta_key):
        raise ValueError(
            f"{key}: {value or 'none'} has sha1 {current}, but the label space {directory} "
            f"records {meta_key} {meta.get(meta_key)}"
        )


def load_run_concept_source_mask(
    training_cfg: Any, schema: ConceptSchema, *, label_space_dir: Path | None = None
) -> ConceptSourceMask | None:
    """The keep table of ``training.concept_source_mask`` (a CSV path), or ``None``.

    The mask must be built for the concept table whose sha1 the label space records. That label
    space is ``label_space_dir``, else the one ``training.label_space`` names. With
    ``label_space_dir`` (a trained bundle's), the mask and removal files must also be those the
    label space was built from.

    Raises:
        ValueError: With ``label_space_dir``, the mask or removal file differs from the one the
            label space records.
    """
    path = training_cfg.concept_source_mask
    if not path:
        return None
    directory = label_space_dir or resolve_repo_path(training_cfg.label_space)
    meta = yaml.safe_load((directory / META_YAML).read_text())
    if label_space_dir is not None:
        _check_input_sha1(
            meta, training_cfg, "concept_source_mask", "concept_source_mask_sha1", directory
        )
        _check_input_sha1(meta, training_cfg, "concept_removal", "concept_removal_sha1", directory)
    removal = training_cfg.concept_removal
    return load_concept_source_mask(
        resolve_repo_path(path),
        schema,
        concepts_csv_sha1=meta["concepts_csv_sha1"],
        removed_concepts=load_concept_removal(resolve_repo_path(removal)) if removal else [],
    )


# Training key -> label space meta key of the sha1 of the file it names.
_LABEL_SPACE_FILES = {
    "concept_mapping_path": "concepts_csv_sha1",
    "concept_removal": "concept_removal_sha1",
    "concept_source_mask": "concept_source_mask_sha1",
}


def _pin(pin: Mapping[str, Any] | None) -> tuple[str, str] | None:
    return None if pin is None else (pin["version"], pin["manifest_sha256"])


def _describe_pin(pin: tuple[str, str] | None) -> str:
    return "nothing" if pin is None else f"version {pin[0]} (manifest {pin[1]})"


def _pin_mismatches(
    space_version: Mapping[str, Any], run_version: Mapping[str, Any], datasets: Iterable[str]
) -> list[str]:
    """The run's dataset pins of ``datasets``, and its taxonomy pin, that the label space lacks."""
    space_datasets = space_version.get("datasets") or {}
    problems = []
    for name in sorted(datasets):
        run_pin = _pin(run_version["datasets"][name])
        space_pin = _pin(space_datasets.get(name))
        if run_pin != space_pin:
            problems.append(
                f"dataset {name}: the run pins {_describe_pin(run_pin)}, "
                f"the label space {_describe_pin(space_pin)}"
            )
    run_taxonomy = _pin(run_version["taxonomy"])
    space_taxonomy = _pin(space_version.get("taxonomy"))
    if run_taxonomy != space_taxonomy:
        problems.append(
            f"taxonomy: the run pins {_describe_pin(run_taxonomy)}, "
            f"the label space {_describe_pin(space_taxonomy)}"
        )
    return problems


def _input_mismatches(meta: Mapping[str, Any], training_cfg: Any) -> list[str]:
    """The label keys and input files of ``training_cfg`` that differ from the label space meta."""
    problems = []
    keys_sha1 = label_keys_sha1(training_cfg)
    if keys_sha1 != meta.get("label_keys_sha1"):
        problems.append(
            f"label_keys_sha1: the run's training keys give {keys_sha1}, "
            f"the label space {meta.get('label_keys_sha1')}"
        )
    for key, meta_key in _LABEL_SPACE_FILES.items():
        value = training_cfg[key]
        current = file_sha1(resolve_repo_path(value)) if value else None
        if current != meta.get(meta_key):
            problems.append(
                f"{key}: {value or 'none'} has sha1 {current}, "
                f"the label space {meta_key} {meta.get(meta_key)}"
            )
    return problems


def _missing_reader_labels(space: LabelSpace, readers: Iterable[VersionDataset]) -> list[str]:
    """Per reader, its ``(dataset, version, label_id)`` labels absent from ``source_labels``."""
    labels = space.source_labels
    known = {
        (dataset, version, int(label_id))
        for dataset, version, label_id in zip(
            labels["dataset"], labels["version"], labels["label_id"], strict=True
        )
    }
    problems = []
    for ds in readers:
        version = ds.version.manifest.version
        missing = sorted(
            int(i)
            for i in ds.version.labels.column("label_id").to_pylist()
            if (ds.source_name, version, int(i)) not in known
        )
        if missing:
            more = " ..." if len(missing) > 10 else ""
            problems.append(
                f"dataset {ds.source_name} {version}: label ids {missing[:10]}{more} "
                f"are not in {SOURCE_LABELS_CSV}"
            )
    return problems


def _check_label_space_current(
    directory: Path,
    space: LabelSpace,
    training_cfg: Any,
    section: DataSection,
    readers: Sequence[VersionDataset],
    *,
    check_inputs: bool,
) -> None:
    """Raise ValueError naming every way ``space`` is stale for this run.

    The run's dataset and taxonomy pins must be those the label space was built from, and every
    label of its readers must have a row. With ``check_inputs``, its label keys and input files
    must be those the label space was built from too. The label space may hold datasets the run
    does not build.
    """
    problems = [
        *_pin_mismatches(
            space.meta.get("data_version") or {},
            data_version(section),
            [ds.source_name for ds in readers],
        ),
        *(_input_mismatches(space.meta, training_cfg) if check_inputs else []),
        *_missing_reader_labels(space, readers),
    ]
    if problems:
        raise ValueError(
            f"label space {directory} does not match this run; rebuild it with "
            "scripts/build_label_space.py:\n" + "\n".join(problems)
        )


def build_registry(
    dataset_dict: Mapping[tuple[str, str], VersionDataset],
    training_cfg: Any,
    *,
    section: DataSection,
    schema_config: str | Path = DEFAULT_SCHEMA_CONFIG,
    device: torch.device | str = "cpu",
    label_space_dir: Path | None = None,
) -> tuple[SourceLabelRegistry, ConceptSchema]:
    """Label registry over ``dataset_dict``, from the label space of ``training.label_space``.

    With ``label_space_dir`` (a trained bundle's label space), that directory is read instead and
    only the dataset pins and reader labels are checked: the label space is fixed, so the
    repository's label keys and input files are not compared with it.

    The registry reads one reader per source: its train split, else the first split listed.
    Every split of a source shares its global offset (see
    :func:`mermaidseg.data.factory.source_offsets`), so any split stands for the source. The
    concept channels and target classes are those of the label space.

    Args:
        dataset_dict: Built readers keyed by ``(name, split)``.
        training_cfg: The ``training`` section.
        section: The ``data`` section the readers were built from; its pins must be those of
            the label space.
        schema_config: Data config whose datasets the built datasets must be among.
        device: Device of the registry's lookup tensors.
        label_space_dir: Label space directory to read in place of ``training.label_space``.

    Raises:
        SystemExit: A built dataset is not in ``schema_config``.
        ValueError: ``training_cfg`` names no label space, or the label space does not match the
            run; the message names every mismatch.
    """
    by_source: dict[str, VersionDataset] = {}
    for (_, split), ds in dataset_dict.items():
        if ds.source_name not in by_source or split == "train":
            by_source[ds.source_name] = ds
    registry_datasets = list(by_source.values())
    schema_sources = set(load_section(schema_config, "data")["datasets"])
    missing = set(by_source) - schema_sources
    if missing:
        raise SystemExit(
            f"built datasets {sorted(missing)} are missing from the frozen concept schema "
            f"({schema_config})"
        )
    if label_space_dir is None and "label_space" not in training_cfg:
        raise ValueError(
            "the run config has no training.label_space; it was trained before label spaces "
            "existed, so retrain it or evaluate with a config that names one"
        )
    directory = label_space_dir or resolve_repo_path(training_cfg.label_space)
    space = read_label_space(directory)
    _check_label_space_current(
        directory,
        space,
        training_cfg,
        section,
        registry_datasets,
        check_inputs=label_space_dir is None,
    )
    schema = ConceptSchema.from_label_space(space)
    registry = SourceLabelRegistry(registry_datasets, space).to(device)
    return registry, schema


def build_meta_model(
    cfg: Any,
    registry: SourceLabelRegistry,
    device: torch.device | str,
    *,
    model: torch.nn.Module | None = None,
) -> MetaModel:
    """MetaModel for ``cfg`` (model and training sections) over ``registry``.

    A prebuilt ``model`` is used as is; otherwise the model is built from ``cfg.model``.
    """
    return MetaModel(
        run_name=cfg.run_name,
        num_classes=registry.num_target_classes,
        num_concepts=registry.num_concepts,
        device=device,
        model_kwargs=cfg.model.copy(),
        training_kwargs=cfg.training.copy(),
        source_to_target_lookup=registry.source_to_target,
        source_to_concepts_lookup=registry.source_to_concepts,
        concept_value2id=registry.concept_value2id,
        model=model,
    )


def read_checkpoint(path: str | Path, required_keys: Sequence[str]) -> dict:
    """Load a checkpoint onto the CPU with ``weights_only=True`` and check its keys.

    Raises:
        RuntimeError: If the file cannot be loaded or lacks a required key; the message names
            ``path``.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:
        raise RuntimeError(f"cannot load checkpoint {str(path)!r}: {err}") from err
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise RuntimeError(f"checkpoint {str(path)!r} is missing required keys {missing}")
    return checkpoint


def load_model_weights(meta_model: MetaModel, checkpoint: dict) -> None:
    """Load the model weights of a loaded ``checkpoint`` into ``meta_model``, strictly."""
    meta_model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def write_resolved_config(cfg: Any, run_dir: str | Path) -> None:
    """Write the resolved run config as ``config.yaml`` in ``run_dir``."""
    text = yaml.safe_dump(to_plain_dict(cfg), sort_keys=False)
    (Path(run_dir) / "config.yaml").write_text(text)


def check_env() -> None:
    """Print the HF token setting of this environment."""
    print(f"HF_TOKEN            : {'set' if os.getenv('HF_TOKEN') else '(not set)'}")
