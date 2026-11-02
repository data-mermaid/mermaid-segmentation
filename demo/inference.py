"""Artifact loading and single-image inference for the demo.

Model loading, preprocessing and tiling/blending are owned by ``mermaidseg.inference``; this
module adapts them to the demo's call signatures.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from mermaidseg.inference import (
    SUPPORTED_CBM_MODELS,
    CBMPredictor,
    ModelBundle,
    load_bundle,
    load_checkpoint_model,
)
from mermaidseg.inference import preprocess as _preprocess_image
from mermaidseg.inference.bundle import INFERENCE_CLASS, MANIFEST_NAME
from mermaidseg.inference.preprocess import (
    DEFAULT_TILE_OVERLAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    resize_rgb,
    tile_blend_weights,
    tile_starts,
)
from mermaidseg.inference.preprocess import predict_tiled as _predict_tiled
from mermaidseg.model.models import (
    ConceptBottleneckDINOv3,
    ConceptBottleneckDPTDINOv3,
    ConceptBottleneckDPTLoRADINOv3,
)

CBMModel = ConceptBottleneckDINOv3 | ConceptBottleneckDPTLoRADINOv3 | ConceptBottleneckDPTDINOv3

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_TILE_OVERLAP",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_CBM_MODELS",
    "CBMModel",
    "DemoArtifacts",
    "build_model",
    "build_transforms",
    "classes_from_concepts",
    "default_taxonomy_csv",
    "load_artifacts",
    "model_input_size",
    "parse_processing_resolution",
    "predict",
    "predict_concepts",
    "predict_tiled",
    "preprocess",
    "resolve_paths",
    "tile_blend_weights",
    "tile_starts",
]

DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 512)


logger = logging.getLogger(__name__)

@dataclass
class DemoArtifacts:
    model_cfg: dict[str, Any]
    id2label: dict[int, str]
    concept_id2name: dict[int, str]
    checkpoint_path: str
    bundle: ModelBundle | None = None


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


def model_input_size(model_cfg: dict[str, Any]) -> tuple[int, int]:
    """The model config's ``input_size`` (height, width), or ``DEFAULT_INPUT_SIZE`` if absent."""
    size = model_cfg.get("input_size")
    if isinstance(size, list | tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    return DEFAULT_INPUT_SIZE


def _is_bundle(path: str | Path) -> bool:
    return (Path(path) / MANIFEST_NAME).is_file()


def load_artifacts(
    checkpoint: str | Path,
    model_config: str | Path | None = None,
    id2label: str | Path | None = None,
    concept_id2name: str | Path | None = None,
) -> DemoArtifacts:
    """Load checkpoint and model config from local paths.

    ``checkpoint`` may also be a bundle directory (``model.json`` + weights). The bundle supplies
    the model config, labels and concepts, so passing ``id2label`` or ``concept_id2name`` with one
    raises ``ValueError`` (``model_config`` is ignored).
    """
    demo_dir = Path(__file__).resolve().parent
    if _is_bundle(checkpoint):
        if id2label or concept_id2name:
            raise ValueError(
                "The bundle carries its own classes and concept names; do not pass "
                "id2label/concept_id2name."
            )
        # Evaluation and the demos need batched eager forwards, never the exported graph.
        bundle = load_bundle(checkpoint, format="checkpoint")
        height, width = bundle.input_size
        return DemoArtifacts(
            model_cfg={"name": INFERENCE_CLASS, "input_size": [height, width]},
            id2label=dict(enumerate(bundle.classes)),
            concept_id2name={i + 1: name for i, name in enumerate(bundle.concept_names)},
            checkpoint_path=str(checkpoint),
            bundle=bundle,
        )
    if model_config is None:
        raise ValueError("model_config is required unless checkpoint is a bundle directory.")
    if not (id2label and concept_id2name):
        logger.warning(
            "Using the default label maps in %s for %s; pass both id2label and concept_id2name "
            "if the checkpoint was trained on other labels.",
            demo_dir,
            checkpoint,
        )
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


def build_model(artifacts: DemoArtifacts, device: torch.device | str) -> CBMModel:
    if artifacts.bundle is not None:
        return artifacts.bundle.model.to(device).eval()
    return load_checkpoint_model(
        artifacts.checkpoint_path,
        artifacts.model_cfg,
        artifacts.id2label,
        artifacts.concept_id2name,
        device,
    )


class _Transform:
    """Resize (and optionally normalize) to ``input_size``, as ``mermaidseg.inference`` does."""

    def __init__(self, input_size: tuple[int, int], normalize: bool):
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.normalize = normalize

    def __call__(self, image_rgb_uint8: NDArray[np.uint8]) -> NDArray[Any] | torch.Tensor:
        if self.normalize:
            return _preprocess_image(image_rgb_uint8, self.input_size, IMAGENET_MEAN, IMAGENET_STD)
        return resize_rgb(image_rgb_uint8, self.input_size)


def build_transforms(
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> tuple[_Transform, _Transform]:
    """Return ``(model_transform, display_transform)`` for ``preprocess``."""
    return _Transform(input_size, normalize=True), _Transform(input_size, normalize=False)


def preprocess(
    image_rgb_uint8: NDArray[np.uint8],
    model_transform: _Transform,
    display_transform: _Transform,
) -> tuple[torch.Tensor, NDArray[np.uint8]]:
    image_tensor = model_transform(image_rgb_uint8).unsqueeze(0).float()
    display = display_transform(image_rgb_uint8)
    return image_tensor, np.asarray(display, dtype=np.uint8)


@torch.no_grad()
def predict_concepts(model: CBMModel, image_tensor: torch.Tensor) -> torch.Tensor:
    outputs = model(image_tensor)
    return outputs.hidden_states


@torch.no_grad()
def classes_from_concepts(model: CBMModel, concept_activations: torch.Tensor) -> torch.Tensor:
    return model.concept_classifier(concept_activations)


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


@torch.no_grad()
def predict_tiled(
    model: CBMModel,
    image_rgb_uint8: NDArray[np.uint8],
    model_transform: _Transform,
    display_transform: _Transform,
    tile_size: tuple[int, int],
    device: torch.device | str,
    *,
    processing_resolution: tuple[int, int] | None = None,
    min_overlap: float = DEFAULT_TILE_OVERLAP,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.uint8]]:
    """Run CBM inference, optionally using overlapping tiles at ``processing_resolution``.

    Without ``processing_resolution`` the whole image is resized to the model input (no tiling).
    """
    if processing_resolution is None:
        image_tensor, display_rgb = preprocess(image_rgb_uint8, model_transform, display_transform)
        class_probs, concept_probs, pred_mask = predict(model, image_tensor.to(device))
        return class_probs, concept_probs, pred_mask, display_rgb

    predictor = CBMPredictor(model, input_size=tile_size, device=device)
    class_probs, concept_probs = _predict_tiled(
        predictor.predict,
        image_rgb_uint8,
        tile_size,
        processing_resolution=processing_resolution,
        min_overlap=min_overlap,
    )
    pred_mask = class_probs.argmax(axis=0).astype(np.int64)
    return class_probs, concept_probs, pred_mask, resize_rgb(image_rgb_uint8, processing_resolution)


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
