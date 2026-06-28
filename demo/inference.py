"""Artifact loading and single-image inference for the demo."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import yaml
from numpy.typing import NDArray

import mermaidseg.model.models as mm_models
from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS, parse_concept_rank
from mermaidseg.model.models import (
    ConceptBottleneckDINOv3,
    ConceptBottleneckDPTDINOv3,
    ConceptBottleneckDPTLoRADINOv3,
)

CBMModel = ConceptBottleneckDINOv3 | ConceptBottleneckDPTLoRADINOv3 | ConceptBottleneckDPTDINOv3

SUPPORTED_CBM_MODELS: frozenset[str] = frozenset(
    {
        "ConceptBottleneckDINOv3",
        "ConceptBottleneckDPTLoRADINOv3",
        "ConceptBottleneckDPTDINOv3",
    }
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 512)


@dataclass
class DemoArtifacts:
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


def _normalize_checkpoint_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in sd.items():
        if key.startswith("encoder.model."):
            key = key.replace("encoder.model.", "encoder.", 1)
        if ".base_model.model.model." in key:
            key = key.replace(".base_model.model.model.", ".base_model.model.", 1)
        normalized[key] = value
    return normalized


def build_model(artifacts: DemoArtifacts, device: torch.device | str) -> CBMModel:
    num_classes = max(artifacts.id2label.keys()) + 1
    ckpt = torch.load(artifacts.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = _normalize_checkpoint_state_dict(state_dict)

    num_concepts = _num_concepts_from_state_dict(state_dict) or len(artifacts.concept_id2name)
    if num_concepts <= 0:
        raise ValueError("Could not determine num_concepts from checkpoint or concept_id2name.")
    if not artifacts.concept_id2name:
        raise ValueError("concept_id2name is required.")

    model_cfg = dict(artifacts.model_cfg)
    model_name = model_cfg.pop("name", "ConceptBottleneckDINOv3")
    if model_name not in SUPPORTED_CBM_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {sorted(SUPPORTED_CBM_MODELS)}"
        )
    model_kwargs: dict[str, Any] = model_cfg
    if "input_size" in model_kwargs and isinstance(model_kwargs["input_size"], list):
        model_kwargs["input_size"] = tuple(model_kwargs["input_size"])
    model_kwargs.setdefault(
        "concept_value2id",
        _derive_concept_value2id(artifacts.concept_id2name),
    )

    model_cls = getattr(mm_models, model_name)
    model = model_cls(num_classes=num_classes, num_concepts=num_concepts, **model_kwargs)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_transforms(
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> tuple[A.Compose, A.Compose]:
    height, width = int(input_size[0]), int(input_size[1])
    model_transforms = [
        A.Resize(height=height, width=width, p=1),
        A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    display_transforms = [A.Resize(height=height, width=width, p=1)]
    return A.Compose(model_transforms), A.Compose(display_transforms)


def preprocess(
    image_rgb_uint8: NDArray[np.uint8],
    model_transform: A.Compose,
    display_transform: A.Compose,
) -> tuple[torch.Tensor, NDArray[np.uint8]]:
    normalized = model_transform(image=image_rgb_uint8)["image"]
    display = display_transform(image=image_rgb_uint8)["image"]
    image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor, display.astype(np.uint8)


@torch.no_grad()
def predict_concepts(model: CBMModel, image_tensor: torch.Tensor) -> torch.Tensor:
    outputs = model(image_tensor)
    return outputs.hidden_states


@torch.no_grad()
def classes_from_concepts(model: CBMModel, concept_activations: torch.Tensor) -> torch.Tensor:
    return model.concept_classifier(concept_activations)


@torch.no_grad()
def predict(
    model: CBMModel, image_tensor: torch.Tensor
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    concept_probs = predict_concepts(model, image_tensor)
    class_logits = classes_from_concepts(model, concept_probs)
    class_probs = torch.softmax(class_logits, dim=1)
    pred_mask = class_probs.argmax(dim=1)
    return (
        class_probs[0].cpu().numpy().astype(np.float32),
        concept_probs[0].cpu().numpy().astype(np.float32),
        pred_mask[0].cpu().numpy().astype(np.int64),
    )


def default_taxonomy_csv() -> str:
    """Default path to class-to-concepts CSV.

    Prefers the copy bundled next to this file (used on HF Spaces, where only ``demo/`` is uploaded)
    and falls back to the repo ``configs/`` copy.
    """
    demo_dir = Path(__file__).resolve().parent
    bundled = demo_dir / "class_to_concepts.csv"
    if bundled.is_file():
        return str(bundled)
    return str(demo_dir.parent / "configs" / "class_to_concepts.csv")


def default_model_config() -> str:
    """Default path to the model config YAML (bundled demo copy, else repo configs)."""
    demo_dir = Path(__file__).resolve().parent
    bundled = demo_dir / "model_config_cbm_dpt_lora_vitl.yaml"
    if bundled.is_file():
        return str(bundled)
    return str(demo_dir.parent / "configs" / "model_config_cbm_dpt_lora_vitl.yaml")


def resolve_paths() -> tuple[str, str, str]:
    """Resolve checkpoint, model config, and taxonomy CSV from env vars or defaults."""
    checkpoint = os.environ.get("DEMO_CHECKPOINT", "")
    model_config = os.environ.get("DEMO_MODEL_CONFIG", default_model_config())
    taxonomy_csv = os.environ.get("DEMO_TAXONOMY_CSV", default_taxonomy_csv())
    if not checkpoint:
        raise ValueError(
            "Set DEMO_CHECKPOINT to a local checkpoint path, or pass --checkpoint on the CLI."
        )
    return checkpoint, model_config, taxonomy_csv
