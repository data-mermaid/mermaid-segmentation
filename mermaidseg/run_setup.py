"""Registry and model construction shared by the training pipeline and the evaluation suite.

Every run freezes its concept channels to the datasets of ``configs/data_config.yaml`` and
its target classes to the sorted ``class_subset``, so a run trained on part of the data (a
data-size profile) and an evaluation that only builds val splits use the same layout.
"""

from __future__ import annotations

import os
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
from mermaidseg.dataset_reconciliation.concept_removal import load_concept_removal
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    ConceptSourceMask,
    load_concept_source_mask,
)
from mermaidseg.datasets import configured_source_names
from mermaidseg.io import setup_config
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.models import align_peft_checkpoint_state_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_CONFIG = REPO_ROOT / "configs" / "data_config.yaml"


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


def resolve_repo_path(path: str | Path) -> Path:
    """An absolute path as given; a relative path resolved against the repository root.

    The current directory never matters, so entrypoints run from anywhere.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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
        compute_concepts=training_cfg.training_mode != "standard",
        concept_schema=schema,
        label_roll_up=training_cfg.label_roll_up,
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
    print("wandb authenticated")


def check_env_wandb() -> None:
    """Validate the environment for wandb-backed, AWS-free training runs.

    Prints the resolved wandb / HF / cache-offline settings and confirms wandb
    authentication (via ``check_wandb_auth``).

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
