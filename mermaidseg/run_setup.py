"""Registry and model construction shared by the training pipeline and the evaluation suite.

Every run freezes its concept channels to the datasets of ``configs/data_config.yaml`` and
its target classes to the sorted ``class_subset``, so a run trained on part of the data (a
data-size profile) and an evaluation that only builds val splits use the same layout.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from mermaidseg.config import to_plain_dict
from mermaidseg.data.base import VersionDataset
from mermaidseg.dataset_reconciliation.concept_removal import load_concept_removal
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    ConceptSourceMask,
    load_concept_source_mask,
)
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.io import REPO_ROOT, load_section, resolve_repo_path
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


def target_labels(training_cfg: Any) -> list[str]:
    """Target classes 1..M: the sorted, lower-cased ``class_subset``."""
    return sorted(dict.fromkeys(str(label).lower() for label in training_cfg.class_subset))


def concept_removal_list(training_cfg: Any, mapping_path: str | Path) -> list[str]:
    """Channel names listed by ``training.concept_removal`` (a CSV path), or none."""
    path = training_cfg["concept_removal"]
    if not path:
        return []
    return load_concept_removal(resolve_repo_path(path), concepts_csv=mapping_path)


def load_run_concept_source_mask(
    training_cfg: Any, schema: ConceptSchema
) -> ConceptSourceMask | None:
    """The keep table of ``training.concept_source_mask`` (a CSV path), or ``None``."""
    path = training_cfg.concept_source_mask
    if not path:
        return None
    mapping_path = concept_mapping_path(training_cfg)
    return load_concept_source_mask(
        resolve_repo_path(path),
        schema,
        concepts_csv=mapping_path,
        removed_concepts=concept_removal_list(training_cfg, mapping_path),
    )


def build_registry(
    dataset_dict: Mapping[tuple[str, str], VersionDataset],
    training_cfg: Any,
    *,
    schema_config: str | Path = DEFAULT_SCHEMA_CONFIG,
    benthic_hierarchy: Mapping[str, str | None],
    device: torch.device | str = "cpu",
) -> tuple[SourceLabelRegistry, ConceptSchema]:
    """Label registry over ``dataset_dict``, with concepts frozen to ``schema_config``.

    The concept channels come from the dataset names of the ``data`` section of
    ``schema_config``. The registry reads one reader per source: its train split, else the first
    split listed. Every split of a source shares its global offset (see
    :func:`mermaidseg.data.factory.source_offsets`), so any split stands for the source.

    Args:
        dataset_dict: Built readers keyed by ``(name, split)``.
        training_cfg: The ``training`` section.
        schema_config: Data config whose datasets freeze the concept channels.
        benthic_hierarchy: Benthic-attribute name -> parent name, read only with
            ``training.label_roll_up`` (see ``mermaidseg.data.factory.load_benthic_hierarchy``).
        device: Device of the registry's lookup tensors.

    Raises:
        SystemExit: A built dataset is not in ``schema_config``.
    """
    by_source: dict[str, VersionDataset] = {}
    for (_, split), ds in dataset_dict.items():
        if ds.source_name not in by_source or split == "train":
            by_source[ds.source_name] = ds
    registry_datasets = list(by_source.values())
    run_sources = set(by_source)
    schema_sources = set(load_section(schema_config, "data")["datasets"])
    missing = run_sources - schema_sources
    if missing:
        raise SystemExit(
            f"built datasets {sorted(missing)} are missing from the frozen concept schema "
            f"({schema_config})"
        )
    mapping_path = concept_mapping_path(training_cfg)
    removed = concept_removal_list(training_cfg, mapping_path)
    schema = ConceptSchema.from_csv(mapping_path, sources=schema_sources, removed_concepts=removed)
    if removed:
        print(
            f"Removed {len(removed)} concept channels ({training_cfg.concept_removal}); "
            f"{schema.num_channels} remain"
        )
    registry = SourceLabelRegistry(
        registry_datasets,
        target_labels=target_labels(training_cfg),
        target_label_subset=training_cfg.class_subset,
        concept_schema=schema,
        label_roll_up=training_cfg.label_roll_up,
        benthic_hierarchy=benthic_hierarchy,
    ).to(device)
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
