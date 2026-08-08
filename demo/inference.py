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
from PIL import Image

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
DEFAULT_TILE_OVERLAP = 0.2


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


def build_model(artifacts: DemoArtifacts, device: torch.device | str) -> CBMModel:
    num_classes = max(artifacts.id2label.keys()) + 1
    ckpt = torch.load(artifacts.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

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
    state_dict = mm_models.align_peft_checkpoint_state_dict(state_dict, model)
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


def tile_starts(length: int, tile_size: int, min_overlap: float = DEFAULT_TILE_OVERLAP) -> list[int]:
    """Return tile origin positions along one axis with at least ``min_overlap`` overlap."""
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - min_overlap)))
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def parse_processing_resolution(value: str) -> tuple[int, int]:
    """Parse ``HxW`` processing resolution (height, width)."""
    normalized = value.lower().replace("x", ",").replace(" ", "")
    parts = [p for p in normalized.split(",") if p]
    if len(parts) != 2:
        raise ValueError(
            f"processing_resolution must be HEIGHTxWIDTH or HEIGHT,WIDTH; got {value!r}"
        )
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"processing_resolution dimensions must be positive; got {value!r}")
    return height, width


def _extract_tile(
    image_rgb: NDArray[np.uint8],
    y0: int,
    x0: int,
    tile_height: int,
    tile_width: int,
) -> NDArray[np.uint8]:
    h, w = image_rgb.shape[:2]
    patch = image_rgb[y0 : min(y0 + tile_height, h), x0 : min(x0 + tile_width, w)]
    if patch.shape[0] == tile_height and patch.shape[1] == tile_width:
        return patch
    padded = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    padded[: patch.shape[0], : patch.shape[1]] = patch
    return padded


def tile_blend_weights(height: int, width: int) -> NDArray[np.float32]:
    """Separable linear distance-to-edge weights for feathering overlapping tiles.

    Each axis weight grows linearly from the border toward the center
    (``min(i + 1, size - i)``), so overlapping tiles blend by proximity to their
    own border: two tiles sharing an overlap produce a single linear ramp that is
    equal-weighted at the midpoint.
    """
    wy = np.minimum(np.arange(1, height + 1), np.arange(height, 0, -1)).astype(np.float32)
    wx = np.minimum(np.arange(1, width + 1), np.arange(width, 0, -1)).astype(np.float32)
    return wy[:, None] * wx[None, :]


@torch.no_grad()
def predict_tiled(
    model: CBMModel,
    image_rgb_uint8: NDArray[np.uint8],
    model_transform: A.Compose,
    display_transform: A.Compose,
    tile_size: tuple[int, int],
    device: torch.device | str,
    *,
    processing_resolution: tuple[int, int] | None = None,
    min_overlap: float = DEFAULT_TILE_OVERLAP,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.uint8]]:
    """Run CBM inference, optionally using overlapping tiles at ``processing_resolution``."""
    if processing_resolution is None:
        image_tensor, display_rgb = preprocess(image_rgb_uint8, model_transform, display_transform)
        class_probs, concept_probs, pred_mask = predict(model, image_tensor.to(device))
        return class_probs, concept_probs, pred_mask, display_rgb

    proc_height, proc_width = processing_resolution
    tile_height, tile_width = tile_size
    resized = np.asarray(
        Image.fromarray(image_rgb_uint8).resize((proc_width, proc_height), Image.BILINEAR),
        dtype=np.uint8,
    )

    num_classes = model.concept_classifier.out_channels
    num_concepts = model.concept_classifier.in_channels
    class_acc = np.zeros((num_classes, proc_height, proc_width), dtype=np.float32)
    concept_acc = np.zeros((num_concepts, proc_height, proc_width), dtype=np.float32)
    weight = np.zeros((proc_height, proc_width), dtype=np.float32)

    y_starts = tile_starts(proc_height, tile_height, min_overlap)
    x_starts = tile_starts(proc_width, tile_width, min_overlap)

    blend = tile_blend_weights(tile_height, tile_width)

    for y0 in y_starts:
        for x0 in x_starts:
            tile_rgb = _extract_tile(resized, y0, x0, tile_height, tile_width)
            image_tensor, _ = preprocess(tile_rgb, model_transform, display_transform)
            tile_class_probs, tile_concept_probs, _ = predict(model, image_tensor.to(device))

            y1 = y0 + tile_height
            x1 = x0 + tile_width
            class_acc[:, y0:y1, x0:x1] += tile_class_probs * blend
            concept_acc[:, y0:y1, x0:x1] += tile_concept_probs * blend
            weight[y0:y1, x0:x1] += blend

    weight = np.maximum(weight, 1e-6)
    class_probs = class_acc / weight[None, ...]
    concept_probs = concept_acc / weight[None, ...]
    pred_mask = class_probs.argmax(axis=0).astype(np.int64)
    return class_probs, concept_probs, pred_mask, resized


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
    """Default path to class-to-concepts CSV (repo ``configs/class_to_concepts.csv``)."""
    demo_dir = Path(__file__).resolve().parent
    return str(demo_dir.parent / "configs" / "class_to_concepts.csv")


def resolve_paths() -> tuple[str, str, str]:
    """Resolve checkpoint, model config, and taxonomy CSV from env vars or defaults."""
    demo_dir = Path(__file__).resolve().parent
    checkpoint = os.environ.get("DEMO_CHECKPOINT", "")
    model_config = os.environ.get(
        "DEMO_MODEL_CONFIG",
        str(demo_dir.parent / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"),
    )
    taxonomy_csv = os.environ.get("DEMO_TAXONOMY_CSV", default_taxonomy_csv())
    if not checkpoint:
        raise ValueError(
            "Set DEMO_CHECKPOINT to a local checkpoint path, or pass --checkpoint on the CLI."
        )
    return checkpoint, model_config, taxonomy_csv
