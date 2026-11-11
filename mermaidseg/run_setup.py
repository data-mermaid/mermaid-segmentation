"""Registry and model construction shared by the training pipeline and the evaluation suite.

Every run freezes its concept channels to the datasets of ``configs/data_config.yaml`` and
its target classes to the sorted ``class_subset``, so a run trained on part of the data (a
data-size profile) and an evaluation that only builds val splits use the same layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.dataset_reconciliation.concepts import DEFAULT_CLASS_TO_CONCEPTS_CSV
from mermaidseg.datasets import configured_source_names
from mermaidseg.io import setup_config
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.models import align_peft_checkpoint_state_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_CONFIG = REPO_ROOT / "configs" / "data_config.yaml"


def concept_mapping_path(training_cfg: Any) -> str:
    """The training config's ``concept_mapping_path``, else configs/class_to_concepts.csv."""
    return str(training_cfg.get("concept_mapping_path") or DEFAULT_CLASS_TO_CONCEPTS_CSV)


def target_labels(training_cfg: Any) -> list[str] | None:
    """Target classes 1..M: the sorted, lower-cased ``class_subset`` (None without one)."""
    subset = training_cfg.get("class_subset")
    if subset is None:
        return None
    return sorted(dict.fromkeys(str(label).lower() for label in subset))


def build_registry(
    dataset_dict: Mapping[tuple[str, str], Any],
    training_cfg: Any,
    *,
    schema_config: str | Path = DEFAULT_SCHEMA_CONFIG,
    device: torch.device | str = "cpu",
) -> tuple[SourceLabelRegistry, ConceptSchema]:
    """Label registry over ``dataset_dict``, with concepts frozen to ``schema_config``.

    The registry is attached to every dataset, which then yields global source labels.
    """
    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    run_sources = {ds.SOURCE_NAME for ds in registry_datasets}
    schema_sources = configured_source_names(setup_config({"data": str(schema_config)}).data)
    missing = run_sources - schema_sources
    if missing:
        raise SystemExit(
            f"built datasets {sorted(missing)} are missing from the frozen concept schema "
            f"({schema_config})"
        )
    mapping_path = concept_mapping_path(training_cfg)
    schema = ConceptSchema.from_csv(mapping_path, sources=schema_sources)
    registry = SourceLabelRegistry(
        registry_datasets,
        target_labels=target_labels(training_cfg),
        target_label_subset=training_cfg.class_subset,
        compute_concepts=training_cfg.training_mode != "standard",
        concept_mapping_path=mapping_path,
        concept_schema=schema,
        label_roll_up=training_cfg.get("label_roll_up", False),
    ).to(device)
    attach_registry(registry, dataset_dict.values())
    if registry.num_concepts != schema.num_channels:
        raise RuntimeError(
            f"registry has {registry.num_concepts} concept channels, schema {schema.num_channels}"
        )
    return registry, schema


def build_meta_model(
    cfg: Any, registry: SourceLabelRegistry, device: torch.device | str
) -> MetaModel:
    """MetaModel for ``cfg`` (model and training sections) over ``registry``."""
    return MetaModel(
        run_name=cfg.run_name,
        num_classes=registry.num_target_classes,
        num_concepts=registry.num_concepts or None,
        device=device,
        model_kwargs=cfg.model.copy(),
        training_kwargs=cfg.training.copy(),
        source_to_target_lookup=registry.source_to_target,
        source_to_concepts_lookup=registry.source_to_concepts,
        concept_matrix=registry.concept_matrix,
        conceptid2labelid=registry.conceptid2labelid(),
        concept_value2id=registry.concept_value2id,
    )


def load_model_weights(
    meta_model: MetaModel, checkpoint_path: str | Path, device: torch.device | str
) -> dict:
    """Load a checkpoint's model weights into ``meta_model``; returns the checkpoint dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state_dict, dict):
        raise ValueError(f"Checkpoint at {str(checkpoint_path)!r} has no model_state_dict")
    state_dict = align_peft_checkpoint_state_dict(raw_state_dict, meta_model.model)
    meta_model.model.load_state_dict(state_dict)
    return checkpoint
