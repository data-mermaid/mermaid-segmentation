"""Artifact and model loading for evaluation and inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

import mermaidseg.model.models as mm_models
from mermaidseg.config import lookup_class
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS, parse_concept_rank
from mermaidseg.model.models import MODEL_CLASSES, ConceptBottleneckDPTLoRADINOv3

CBMModel = ConceptBottleneckDPTLoRADINOv3


@dataclass
class DemoArtifacts:
    """Config, label maps, and checkpoint path needed to build the demo model."""

    model_cfg: dict[str, Any]
    id2label: dict[int, str]
    concept_id2name: dict[int, str]
    checkpoint_path: str


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a dict")
    return data


def _load_int_keyed_json(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_artifacts(
    checkpoint: str | Path,
    model_config: str | Path,
    id2label: str | Path | None = None,
    concept_id2name: str | Path | None = None,
) -> DemoArtifacts:
    """Load checkpoint and model config from local paths."""
    demo_dir = Path(__file__).resolve().parent
    id2label_path = Path(id2label or demo_dir / "id2label.json")
    concept_path = Path(concept_id2name or demo_dir / "concept_id2name.json")
    cfg = _load_yaml(model_config)
    model_cfg = dict(cfg.get("model") or cfg)
    return DemoArtifacts(
        model_cfg=model_cfg,
        id2label=_load_int_keyed_json(id2label_path),
        concept_id2name=_load_int_keyed_json(concept_path),
        checkpoint_path=str(checkpoint),
    )


def _num_concepts_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    weight = state_dict.get("concept_classifier.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def _derive_concept_value2id(
    concept_id2name: Mapping[int, str],
) -> dict[str, dict[str, list[int]]]:
    concept_value2id: dict[str, dict[str, list[int]]] = {}
    for rank in TAXONOMIC_CONCEPTS:
        entries: list[tuple[int, str]] = []
        for idx, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0])):
            parsed_rank, value = parse_concept_rank(name)
            if parsed_rank == rank:
                entries.append((idx, value))
        if not entries:
            continue
        width = len(entries)
        mapping: dict[str, list[int]] = {"not_given": [0] * width}
        for i, (_, value) in enumerate(entries):
            encoded = [1] * width
            encoded[i] = 2
            mapping[value] = encoded
        concept_value2id[rank] = mapping
    return concept_value2id


def build_model(artifacts: DemoArtifacts, device: torch.device | str) -> CBMModel:
    """Build the concept-bottleneck model and load its checkpoint weights."""
    num_classes = max(artifacts.id2label.keys()) + 1
    ckpt = torch.load(artifacts.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    num_concepts = _num_concepts_from_state_dict(state_dict) or len(artifacts.concept_id2name)
    if num_concepts <= 0:
        raise ValueError("Could not determine num_concepts from checkpoint or concept_id2name.")
    if not artifacts.concept_id2name:
        raise ValueError("concept_id2name is required.")

    model_cfg = dict(artifacts.model_cfg)
    model_cls = lookup_class(MODEL_CLASSES, "model", model_cfg.pop("name"))
    model_kwargs: dict[str, Any] = model_cfg
    if "input_size" in model_kwargs and isinstance(model_kwargs["input_size"], list):
        model_kwargs["input_size"] = tuple(model_kwargs["input_size"])
    model_kwargs.setdefault(
        "concept_value2id",
        _derive_concept_value2id(artifacts.concept_id2name),
    )

    # detach_concepts only changes gradients, which inference never computes.
    model = model_cls(
        num_classes=num_classes,
        num_concepts=num_concepts,
        detach_concepts=False,
        **model_kwargs,
    )
    state_dict = mm_models.align_peft_checkpoint_state_dict(state_dict, model)
    # Checkpoints from code that deletes the first DPT fusion layer's residual_layer1 lack its weights. That block
    # never runs (layer 0 gets no residual), so its missing keys are harmless; any other mismatch still fails.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "fusion_stage.layers.0.residual_layer1." not in k]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()
