"""Model and artifact loading + single-image inference for the demo.

Kept UI-agnostic so the same helpers can back a Gradio app, a FastAPI service,
or a notebook. The forward pass is intentionally split into
:func:`predict_concepts` (expensive encoder + concept head) and
:func:`classes_from_concepts` (cheap 1x1 concept→class conv) so future concept
intervention can edit concept logits and re-run only the downstream step.
"""

from __future__ import annotations

import colorsys
import functools
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    TAXONOMIC_CONCEPTS,
)
from mermaidseg.io import ConfigDict, load_config, setup_config
from mermaidseg.model.models import ConceptBottleneckDINOv3

# Imagenet stats; used as a fallback Normalize for the demo's val pipeline when
# the training config's transforms are not recoverable (e.g. when the config
# came from an MLflow-logged JSON where A.Compose objects were stringified).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 512)


@dataclass
class DemoArtifacts:
    """All non-model state the demo needs at startup."""

    cfg: ConfigDict
    id2label: dict[int, str]
    concept_id2name: dict[int, str]
    checkpoint_path: str


def _load_config_file(path: str | Path) -> ConfigDict:
    """Load a single merged config from YAML or JSON, without preprocessing.

    Used for MLflow-logged ``config/config.json`` and for the legacy
    single-file local case. The new four-file form goes through
    :func:`mermaidseg.io.setup_config` instead (see :func:`load_artifacts_local`).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a dict")
    return ConfigDict(data)


def _load_int_keyed_json(path: str | Path) -> dict[int, str]:
    """Load a JSON dict with string-int keys and coerce keys to int."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _resolve_training_mode(cfg: ConfigDict) -> str | None:
    """Return ``training_mode`` from the new ``cfg.training`` location, with legacy fallback."""
    training = cfg.get("training")
    if isinstance(training, Mapping):
        mode = training.get("training_mode")
        if mode is not None:
            return mode
    return cfg.get("training_mode")


def load_artifacts_local(
    checkpoint: str | Path,
    id2label: str | Path,
    *,
    config: str | Path | None = None,
    data_config: str | Path | None = None,
    model_config: str | Path | None = None,
    training_config: str | Path | None = None,
    logger_config: str | Path | None = None,
    concept_id2name: str | Path | None = None,
) -> DemoArtifacts:
    """Load demo artifacts from local files.

    Supports two config layouts:

    1. **Merged**: pass ``config=<path>`` pointing at a single YAML/JSON that
       already contains ``model``/``training``/``data`` sections. This is the
       format MLflow logs to ``config/config.json``.
    2. **Split (new)**: pass any combination of ``data_config``, ``model_config``,
       ``training_config``, ``logger_config`` paths; they are merged via
       :func:`mermaidseg.io.setup_config`.

    Asserts the resolved config declares ``training_mode: concept-bottleneck``
    since only that mode produces both per-class logits and per-concept logits
    in the same forward pass.
    """
    split_paths = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "logger": logger_config,
    }
    any_split = any(v is not None for v in split_paths.values())

    if config is not None and any_split:
        raise ValueError(
            "Pass either --config (merged) OR the split config paths (--data-config / "
            "--model-config / --training-config / --logger-config), not both."
        )
    if config is None and not any_split:
        raise ValueError(
            "Provide a config: either --config <merged.json|.yaml> or at least "
            "--model-config + --training-config (data/logger optional)."
        )

    if config is not None:
        cfg = _load_config_file(config)
        cfg_source: str = str(config)
    else:
        cfg = setup_config({k: str(v) for k, v in split_paths.items() if v is not None})
        cfg_source = ", ".join(f"{k}={v}" for k, v in split_paths.items() if v is not None)

    mode = _resolve_training_mode(cfg)
    if mode != "concept-bottleneck":
        hint = ""
        if mode is None:
            if config is not None:
                top_keys = sorted(cfg.keys())
                hint = (
                    f" The merged --config at {config} has no 'training.training_mode' "
                    f"(top-level keys: {top_keys}). If you meant to pass the per-file split, "
                    "use --model-config / --training-config / --data-config / --logger-config "
                    "instead of --config."
                )
            elif training_config is None:
                hint = (
                    " 'training.training_mode' lives in --training-config. Pass "
                    "--training-config configs/training_config_cbm.yaml (alongside the other "
                    "split configs you already provided)."
                )
        raise ValueError(
            f"Demo only supports training_mode='concept-bottleneck', got {mode!r} from {cfg_source}.{hint}"
        )
    return DemoArtifacts(
        cfg=cfg,
        id2label=_load_int_keyed_json(id2label),
        concept_id2name=_load_int_keyed_json(concept_id2name) if concept_id2name else {},
        checkpoint_path=str(checkpoint),
    )


_CONCEPT_NAME_ARTIFACT_CANDIDATES: tuple[str, ...] = (
    "metadata/concept_id2name.json",  # logged by Logger.log_reconciliation
    "metadata/id2concept.json",       # logged by Logger when id2concept= is passed
)


def load_artifacts_mlflow(run_id: str, dst_dir: str | Path | None = None) -> DemoArtifacts:
    """Download artifacts produced by ``Logger`` for an MLflow run and load them.

    Expects the run to contain:

    - ``best-model/*.pt`` — the checkpoint pickle saved by ``Logger.save_model_checkpoint``.
    - ``config/config.json`` — the merged training config (post-:func:`setup_config`).
    - ``metadata/id2label.json`` — class id → name map (required).
    - One of ``metadata/concept_id2name.json`` / ``metadata/id2concept.json`` — concept id → name
      map. Optional: if neither is present, the demo falls back to ``concept_<i>`` placeholders
      derived from the model's ``num_concepts``.
    """
    import mlflow

    dst = Path(dst_dir) if dst_dir else Path(tempfile.mkdtemp(prefix="mermaidseg_demo_"))
    dst.mkdir(parents=True, exist_ok=True)

    def _pull(artifact_path: str) -> str:
        return mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=str(dst)
        )

    # ``best-model/`` directory contains the checkpoint pickle plus metadata.json.
    # The pickle was named after the epoch checkpoint when logged (e.g. ``model_epoch9``),
    # so pull the whole directory and pick the largest non-metadata file.
    best_dir = Path(_pull("best-model"))
    candidates = [p for p in best_dir.iterdir() if p.suffix != ".json"]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under MLflow best-model/ at {best_dir}")
    ckpt_path = max(candidates, key=lambda p: p.stat().st_size)

    config_path = _pull("config/config.json")
    id2label_path = _pull("metadata/id2label.json")

    concept_id2name_path: str | None = None
    for cand in _CONCEPT_NAME_ARTIFACT_CANDIDATES:
        try:
            concept_id2name_path = _pull(cand)
            break
        except Exception:  # noqa: BLE001 — mlflow raises a variety of errors when absent
            continue

    return load_artifacts_local(
        checkpoint=ckpt_path,
        config=config_path,
        id2label=id2label_path,
        concept_id2name=concept_id2name_path,
    )


def _num_concepts_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    """Infer the model's concept-head channel count from a checkpoint state dict.

    Returns ``None`` if the expected concept-classifier weight isn't present.
    """
    weight = state_dict.get("concept_classifier.weight")
    if weight is None:
        return None
    # Conv2d weight shape: (num_classes, num_concepts, 1, 1)
    return int(weight.shape[1])


def _derive_concept_value2id(
    concept_id2name: Mapping[int, str],
) -> dict[str, dict[str, list[int]]]:
    """Rebuild taxonomic ``concept_value2id`` from demo ``rank__value`` channel names.

    Matches the binary-style encodings used during training: each concrete value
    within a rank maps to a list with ``2`` at its column index and ``1`` elsewhere.
    """
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


def build_model(
    artifacts: DemoArtifacts, device: torch.device | str
) -> ConceptBottleneckDINOv3:
    """Instantiate ``ConceptBottleneckDINOv3`` from ``artifacts.cfg`` and load weights.

    ``num_concepts`` is taken from the checkpoint's ``concept_classifier.weight``
    when present (most reliable: it always matches the trained model), then
    falls back to ``len(concept_id2name)`` if no checkpoint hint is available.

    ``concept_value2id`` is derived from ``artifacts.concept_id2name`` so the
    model's :meth:`ConceptBottleneckDINOv3.concept_outputs_activation` chunks
    the concept axis the same way training did (one softmax group per taxonomic
    rank, sigmoid for the binary tail).
    """
    num_classes = max(artifacts.id2label.keys()) + 1

    ckpt = torch.load(artifacts.checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    def _strip_encoder_model_prefix(sd: dict) -> dict:
        """Older HF DINOv3 nested params under `encoder.model.*`; current versions
        use `encoder.*`. Strip the extra `model.` so checkpoints stay portable."""
        return {
            (k.replace("encoder.model.", "encoder.", 1) if k.startswith("encoder.model.") else k): v
            for k, v in sd.items()
        }

    state_dict = _strip_encoder_model_prefix(state_dict)

    num_concepts = _num_concepts_from_state_dict(state_dict) or len(artifacts.concept_id2name)
    if num_concepts <= 0:
        raise ValueError(
            "Could not determine num_concepts: checkpoint has no 'concept_classifier.weight' "
            "and concept_id2name is empty. Provide --concept-id2name or use a checkpoint that "
            "includes the concept classifier weights."
        )

    if not artifacts.concept_id2name:
        raise ValueError(
            "concept_id2name is required to derive concept_value2id for grouped "
            "concept activation. Provide --concept-id2name."
        )

    model_kwargs: dict[str, Any] = dict(artifacts.cfg.model or {})
    model_kwargs.pop("name", None)
    if "input_size" in model_kwargs and isinstance(model_kwargs["input_size"], list):
        model_kwargs["input_size"] = tuple(model_kwargs["input_size"])
    model_kwargs.setdefault(
        "concept_value2id",
        _derive_concept_value2id(artifacts.concept_id2name),
    )

    model = ConceptBottleneckDINOv3(
        num_classes=num_classes, num_concepts=num_concepts, **model_kwargs
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _extract_val_transforms_from_compose(compose: A.Compose) -> list[A.BasicTransform]:
    """Pull the ordered list of individual transforms out of an ``A.Compose``."""
    return list(compose.transforms)


def _find_val_transform_spec(
    cfg: ConfigDict,
) -> list[A.BasicTransform] | dict[str, Any] | None:
    """Locate the val pipeline in either the new or legacy config layouts.

    Returns either:

    - A list of ``A.BasicTransform`` (when the config went through
      :func:`mermaidseg.io.setup_config` and the dataset's ``val.transform`` is
      already a compiled ``A.Compose``), or
    - A dict ``{Op: kwargs}`` (raw YAML form before preprocessing, or the
      legacy ``cfg.augmentation.val`` block), or
    - ``None`` when neither shape is present (e.g. the config came from an
      MLflow-logged JSON where transforms were stringified).
    """
    data = cfg.get("data")
    if isinstance(data, Mapping):
        for ds_name, ds_cfg in data.items():
            if ds_name == "default" or not isinstance(ds_cfg, Mapping):
                continue
            val_cfg = ds_cfg.get("val")
            if not isinstance(val_cfg, Mapping):
                continue
            transform = val_cfg.get("transform")
            if isinstance(transform, A.Compose):
                return _extract_val_transforms_from_compose(transform)
            if isinstance(transform, Mapping):
                return dict(transform)

    augmentation = cfg.get("augmentation")
    if isinstance(augmentation, Mapping):
        val = augmentation.get("val")
        if isinstance(val, Mapping):
            return dict(val)
    return None


def _default_val_pipeline(cfg: ConfigDict) -> dict[str, dict[str, Any]]:
    """Build a default Resize-to-``input_size`` + ImageNet ``Normalize`` spec."""
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), Mapping) else None
    input_size = (model_cfg or {}).get("input_size", DEFAULT_INPUT_SIZE)
    height, width = int(input_size[0]), int(input_size[1])
    return {
        "Resize": {"height": height, "width": width, "p": 1},
        "Normalize": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }


def build_transforms(cfg: ConfigDict) -> tuple[A.Compose, A.Compose]:
    """Return ``(model_transform, display_transform)`` for the val pipeline.

    Both share the val pipeline's geometric ops; only ``model_transform``
    applies ``Normalize``. Looks for the val transform spec in:

    1. ``cfg.data.<dataset>.val.transform`` (new ``setup_config``-produced layout),
       which may be either an already-compiled ``A.Compose`` or a raw
       ``{Op: kwargs}`` mapping.
    2. ``cfg.augmentation.val`` (legacy single-file layout).
    3. Falls back to Resize-to-``cfg.model.input_size`` + ImageNet ``Normalize``.
    """
    spec = _find_val_transform_spec(cfg)
    if spec is None:
        spec = _default_val_pipeline(cfg)

    if isinstance(spec, list):
        model_transforms = list(spec)
        display_transforms = [t for t in spec if not isinstance(t, A.Normalize)]
    else:
        items = list(spec.items())
        model_transforms = [getattr(A, name)(**params) for name, params in items]
        display_transforms = [
            getattr(A, name)(**params) for name, params in items if name != "Normalize"
        ]

    return A.Compose(model_transforms), A.Compose(display_transforms)


def preprocess(
    image_rgb_uint8: NDArray[np.uint8],
    model_transform: A.Compose,
    display_transform: A.Compose,
) -> tuple[torch.Tensor, NDArray[np.uint8]]:
    """Apply the val pipeline; return ``(image_tensor, display_image)``.

    ``image_tensor`` has shape ``(1, 3, H, W)`` and float dtype.
    ``display_image`` has shape ``(H, W, 3)`` uint8 — same H, W as the tensor —
    so pixel-click coords on the displayed image map 1:1 to the prediction.
    """
    normalized = model_transform(image=image_rgb_uint8)["image"]
    display = display_transform(image=image_rgb_uint8)["image"]
    image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor, display.astype(np.uint8)


@torch.no_grad()
def predict_concepts(
    model: ConceptBottleneckDINOv3, image_tensor: torch.Tensor
) -> torch.Tensor:
    """Run encoder + concept head + grouped activation, return concept activations in ``[0, 1]``.

    Activations match what the model's ``concept_classifier`` consumes during
    training (per-group softmax for taxonomic ranks, sigmoid for the binary
    tail). This is the expensive step. Cache the result if you plan to
    re-classify after intervening on concept values — interventions should be
    expressed in the same ``[0, 1]`` space.
    """
    encoder_out = model.encoder(image_tensor)
    # DINOv3 emits 5 prefix tokens (CLS + 4 register tokens); strip them.
    patch_embeddings = encoder_out.last_hidden_state[:, 5:, :]
    concept_logits = model.concept_head(patch_embeddings)
    concept_logits = torch.nn.functional.interpolate(
        concept_logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
    )
    return model.concept_outputs_activation(concept_logits)


@torch.no_grad()
def classes_from_concepts(
    model: ConceptBottleneckDINOv3, concept_activations: torch.Tensor
) -> torch.Tensor:
    """Run only the 1×1 concept→class conv on activations in ``[0, 1]``.

    Cheap; safe to re-run after intervention. Input must be in the same
    ``[0, 1]`` space the model was trained on (i.e. the output of
    :func:`predict_concepts` or a direct edit thereof).
    """
    return model.concept_classifier(concept_activations)


@torch.no_grad()
def predict(
    model: ConceptBottleneckDINOv3, image_tensor: torch.Tensor
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    """Thin composition of :func:`predict_concepts` and :func:`classes_from_concepts`.

    Returns ``(class_probs, concept_probs, pred_mask)`` with leading batch dim
    squeezed. Shapes: ``(C, H, W)``, ``(K, H, W)``, ``(H, W)``.
    """
    concept_probs = predict_concepts(model, image_tensor)
    class_logits = classes_from_concepts(model, concept_probs)
    class_probs = torch.softmax(class_logits, dim=1)
    pred_mask = class_probs.argmax(dim=1)
    return (
        class_probs[0].cpu().numpy().astype(np.float32),
        concept_probs[0].cpu().numpy().astype(np.float32),
        pred_mask[0].cpu().numpy().astype(np.int64),
    )


# Semantic palette groups, keyed by class id in id2label.json. Anything not
# listed here falls into the generic "other" fallback so the palette stays
# correct even if id2label adds new entries.
BACKGROUND_IDS: frozenset[int] = frozenset({0})
SAND_IDS: frozenset[int] = frozenset({58})  # Sand
HARD_SUBSTRATE_IDS: frozenset[int] = frozenset({7, 55, 56, 57})  # Bare substrate, Rock, Rubble
ALGAE_IDS: frozenset[int] = frozenset(
    {
        9,   # Crustose coralline algae
        10,  # Cyanobacteria
        12,  # Dictyota
        26,  # Halimeda
        34,  # Lobophora
        36,  # Macroalgae
        47,  # Padina
        60,  # Seagrass
        67,  # Turbinaria-algae
        70,  # Turf algae
    }
)
SPONGE_IDS: frozenset[int] = frozenset({64})  # Sponge
CORAL_IDS: frozenset[int] = frozenset(
    {
        1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 61, 62, 63, 65, 66, 68, 69,
    }
)

SAND_RGB: tuple[int, int, int] = (237, 201, 145)
OTHER_FALLBACK_RGB: tuple[int, int, int] = (80, 60, 100)


def _hsv_ramp(
    n: int,
    hue_deg_range: tuple[float, float],
    s_range: tuple[float, float] = (0.55, 0.9),
    v_range: tuple[float, float] = (0.55, 0.95),
) -> NDArray[np.uint8]:
    """Return ``n`` evenly-spread RGB colors over an HSV hue range.

    Saturation and value zig-zag across ``s_range`` / ``v_range`` so adjacent
    indices have slightly different lightness — helps neighbors in the same
    family stay visually distinct.
    """
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    out = np.zeros((n, 3), dtype=np.uint8)
    h0, h1 = hue_deg_range
    s_lo, s_hi = s_range
    v_lo, v_hi = v_range
    for i in range(n):
        t = i / max(n - 1, 1)
        hue = ((h0 + (h1 - h0) * t) % 360) / 360.0
        # Alternate sat/val along the ramp for inter-class contrast.
        sat = s_lo + (s_hi - s_lo) * (0.5 + 0.5 * np.sin(i * 1.7))
        val = v_lo + (v_hi - v_lo) * (0.5 + 0.5 * np.cos(i * 2.3))
        r, g, b = colorsys.hsv_to_rgb(hue, float(sat), float(val))
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def _gray_ramp(n: int, v_range: tuple[int, int] = (80, 180)) -> NDArray[np.uint8]:
    """Return ``n`` evenly-spaced grayscale colors with V in ``v_range``."""
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = v_range
    values = np.linspace(lo, hi, n, dtype=np.int32)
    out = np.stack([values, values, values], axis=1).astype(np.uint8)
    return out


def make_color_palette(num_classes: int) -> NDArray[np.uint8]:
    """Semantically grouped per-class RGB palette.

    - Background → black.
    - Sand → sand tan.
    - Hard substrate (bare substrate, rock, rubble) → grays.
    - Algae (incl. cyanobacteria) → greens.
    - Corals (hard + soft + zoanthids + fire coral + gorgonian) → warm hues
      sweeping red → orange → pink → magenta → purple.
    - Sponges → cyan/blue.
    - Anything else → a single distinct dark purple so it's clearly "other".
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    def _assign(ids: frozenset[int], colors: NDArray[np.uint8]) -> None:
        present = sorted(i for i in ids if 0 <= i < num_classes)
        for idx, cid in enumerate(present):
            palette[cid] = colors[idx]

    _assign(HARD_SUBSTRATE_IDS, _gray_ramp(len(HARD_SUBSTRATE_IDS)))
    # Hue range tightened around pure green (120°): the previous (95, 150)
    # tail drifted into yellow-green and cyan/teal, washing out the "algae"
    # read. (90, 135) still gives 11 distinguishable algae classes but every
    # color reads as unambiguously green. Also bump saturation so the greens
    # are vivid rather than pastel.
    _assign(
        ALGAE_IDS,
        _hsv_ramp(
            len(ALGAE_IDS),
            hue_deg_range=(90, 135),
            s_range=(0.75, 0.95),
            v_range=(0.55, 0.9),
        ),
    )
    _assign(SPONGE_IDS, _hsv_ramp(len(SPONGE_IDS), hue_deg_range=(180, 215)))

    # Warm wrap: red→orange→pink→magenta→purple spans 285°→360°+30°.
    # Generate over 285→390 then take % 360 inside _hsv_ramp.
    _assign(CORAL_IDS, _hsv_ramp(len(CORAL_IDS), hue_deg_range=(285, 390)))

    for sid in SAND_IDS:
        if 0 <= sid < num_classes:
            palette[sid] = SAND_RGB

    # Mark "other" classes (not background, not assigned above) explicitly.
    assigned = BACKGROUND_IDS | SAND_IDS | HARD_SUBSTRATE_IDS | ALGAE_IDS | SPONGE_IDS | CORAL_IDS
    for cid in range(num_classes):
        if cid in assigned:
            continue
        if cid in BACKGROUND_IDS:
            continue
        palette[cid] = OTHER_FALLBACK_RGB

    for bid in BACKGROUND_IDS:
        if 0 <= bid < num_classes:
            palette[bid] = (0, 0, 0)

    return palette


def compose_overlay(
    display_rgb: NDArray[np.uint8],
    mask: NDArray[np.int64],
    palette: NDArray[np.uint8],
    opacity: float,
) -> NDArray[np.uint8]:
    """Blend ``palette[mask]`` over ``display_rgb`` at ``opacity``; keep background pixels untouched."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    overlay = palette[mask]  # (H, W, 3) uint8
    blended = display_rgb.astype(np.float32) * (1.0 - opacity) + overlay.astype(np.float32) * opacity
    composite = blended.astype(np.uint8)
    background = mask == 0
    composite[background] = display_rgb[background]
    return composite


# Pre-resizing the composite to this size before sending it to Gradio means
# the natural image size == the displayed pixel size, so the demo's click
# coordinates don't get warped by Gradio's `object-fit: scale-down` letterboxing.
DISPLAY_SIZE = 720


def resize_for_display(image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Bilinear-resize an RGB composite to ``DISPLAY_SIZE x DISPLAY_SIZE``."""
    pil = Image.fromarray(image_rgb).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def draw_click_marker(
    image_rgb: NDArray[np.uint8],
    xy: tuple[int, int] | None,
) -> NDArray[np.uint8]:
    """Draw a small ring + center dot at ``xy`` on a display-space image.

    ``xy`` is in display coordinates (i.e. relative to ``DISPLAY_SIZE``).
    Returns ``image_rgb`` unchanged when ``xy`` is ``None``. Operates on a
    copy so the caller's array stays intact.
    """
    if xy is None:
        return image_rgb
    x, y = int(xy[0]), int(xy[1])
    pil = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Outer ring (white) + inner dot (red) so the marker reads on any
    # background and stays small enough to not occlude much.
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 255, 255, 240),
        width=2,
    )
    draw.ellipse(
        (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
        outline=(0, 0, 0, 200),
        width=1,
    )
    draw.ellipse(
        (x - 3, y - 3, x + 3, y + 3),
        fill=(255, 60, 60, 255),
        outline=(255, 255, 255, 240),
        width=1,
    )
    composed = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.asarray(composed, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Concept rank parsing & per-rank palettes
# ---------------------------------------------------------------------------

# Full taxonomy from kingdom down to genus. ``species`` is intentionally
# omitted because :func:`initialize_taxonomic_binary_concept_mapping` drops
# it before training, so the model has no species channels.
RANK_ORDER: tuple[str, ...] = (
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
)

# Overlay-mode-specific solid colors (RGB) for the single-concept modes.
# Algae is a saturated, pure green (no blue) so the heatmap reads as
# unambiguously "green" rather than the previous teal-leaning (0, 200, 80).
ALGAE_OVERLAY_RGB: tuple[int, int, int] = (10, 230, 25)
BLEACHED_OVERLAY_RGB: tuple[int, int, int] = (255, 255, 255)

# Conditional concept-panel groups, shown below the taxonomy tree when the
# corresponding gating concept fires (see ``app.py``).
CORAL_MORPHOLOGY_HEALTH_CONCEPTS: tuple[str, ...] = (
    "oval", "bushy", "tabular", "external_polyps", "branching", "massive",
    "lobed_brain", "digitate", "tubular", "cup_coral", "round", "plating",
    "foliose", "solitary", "brain", "meandroid", "fleshy", "encrusting",
    "columnar", "arborescent", "phaceloid", "free_living", "submassive",
    "corymbose", "dead", "bleached",
)
SUBSTRATE_CONCEPTS: tuple[str, ...] = (
    "algae", "background", "anthropogenic", "trash", "transect",
    "macroalgae", "dark","human", "sand", "hard_substrate", 
)
# Gating sigmoid concepts and thresholds (see ``app.py``).
CORAL_GATE_CONCEPT: str = "order__scleractinia"
CORAL_GATE_THRESHOLD: float = 0.6
SUBSTRATE_GATE_CONCEPT: str = "kingdom__animalia"
SUBSTRATE_GATE_THRESHOLD_BELOW: float = 0.7

# Distinct hue ranges per rank so each rank's palette feels different and the
# eye doesn't conflate e.g. a kingdom-mode overlay with a phylum-mode one.
_RANK_HUE_RANGES: dict[str, tuple[float, float]] = {
    "kingdom": (0, 360),
    "phylum": (30, 300),
    "class": (180, 540),    # cyan → red wrap
    "order": (60, 330),
    "family": (0, 720),     # two full sweeps to keep ~90 families distinct
    "genus": (0, 1080),     # three full sweeps for ~180 genera
}


def parse_concept_rank(name: str) -> tuple[str | None, str]:
    """Split a concept name into ``(rank, value)`` if it has a ``<rank>__<value>`` form.

    Morphology / health / auxiliary concepts (``oval``, ``bleached``, ``algae`` …)
    have no ``__`` separator; for those the rank is ``None`` and ``value`` is
    the raw name. Only the first ``__`` is treated as a separator so e.g.
    ``genus__pocillopora_damicornis`` stays intact on the value side.
    """
    if "__" not in name:
        return None, name
    rank, value = name.split("__", 1)
    return rank, value


def build_rank_index(
    concept_names: list[str],
) -> dict[str, list[tuple[int, str]]]:
    """Group ``concept_names`` by taxonomic rank.

    Returns a dict keyed by every rank in :data:`RANK_ORDER` (always present,
    possibly empty) mapping to ``[(channel_idx, value_name), ...]`` in the
    order they appear in ``concept_names`` — that order matches the model's
    concept-classifier input channels, so callers can use these indices to
    slice ``concept_probs`` directly.
    """
    index: dict[str, list[tuple[int, str]]] = {rank: [] for rank in RANK_ORDER}
    for idx, name in enumerate(concept_names):
        rank, value = parse_concept_rank(name)
        if rank in index:
            index[rank].append((idx, value))
    return index


def find_concept_channel(concept_names: list[str], name: str) -> int | None:
    """Return the channel index for ``name``, or ``None`` if absent."""
    try:
        return concept_names.index(name)
    except ValueError:
        return None


@functools.lru_cache(maxsize=4)
def load_taxonomy_parents(
    csv_path: str | Path = DEFAULT_CLASS_TO_CONCEPTS_CSV,
) -> dict[str, str]:
    """Read ``class_to_concepts.csv`` and return a ``child -> parent`` concept map.

    Both keys and values are full ``<rank>__<value>`` concept names (matching
    what's stored in ``concept_id2name.json``). Only adjacent ranks in
    :data:`RANK_ORDER` produce edges, so the resulting map walks one rank at
    a time from genus back to kingdom. Rows with placeholder values
    (``not_given``, ``none``, blank) are skipped.

    Cached because the CSV has ~2k rows; reading it once at demo startup and
    then on every pixel click is plenty cheap, but no need to re-read.
    """
    df = pd.read_csv(csv_path)
    parents: dict[str, str] = {}
    placeholders = {"not_given", "none", "", None}
    for child_rank, parent_rank in zip(RANK_ORDER[1:], RANK_ORDER[:-1]):
        if child_rank not in df.columns or parent_rank not in df.columns:
            continue
        sub = df[[child_rank, parent_rank]].dropna()
        for child_val, parent_val in sub.itertuples(index=False, name=None):
            if child_val in placeholders or parent_val in placeholders:
                continue
            child_key = f"{child_rank}__{child_val}"
            parent_key = f"{parent_rank}__{parent_val}"
            # First write wins; the same child should always point at the
            # same parent, but in case of CSV inconsistency, prefer the
            # earlier (typically canonical) row.
            parents.setdefault(child_key, parent_key)
    return parents


def make_rank_palette(values: list[str], rank: str) -> dict[str, NDArray[np.uint8]]:
    """Assign each concrete ``rank`` value a distinct RGB color.

    Uses :func:`_hsv_ramp` with a rank-specific hue range so adjacent ranks
    don't paint the world the same color. Values are sorted before assignment
    so the same value always gets the same color across runs.
    """
    sorted_values = sorted(values)
    hue_range = _RANK_HUE_RANGES.get(rank, (0, 360))
    colors = _hsv_ramp(len(sorted_values), hue_deg_range=hue_range)
    return {value: colors[i] for i, value in enumerate(sorted_values)}


# ---------------------------------------------------------------------------
# Per-mode overlay compositor
# ---------------------------------------------------------------------------


def _blend(
    display_rgb: NDArray[np.uint8],
    color_per_pixel: NDArray[np.uint8],     # (H, W, 3) uint8
    alpha_per_pixel: NDArray[np.float32],   # (H, W) in [0, 1]
) -> NDArray[np.uint8]:
    """Alpha-blend ``color_per_pixel`` over ``display_rgb`` with per-pixel alpha."""
    a = np.clip(alpha_per_pixel, 0.0, 1.0)[..., None]
    blended = display_rgb.astype(np.float32) * (1.0 - a) + color_per_pixel.astype(np.float32) * a
    return blended.astype(np.uint8)


def compose_concept_overlay(
    display_rgb: NDArray[np.uint8],
    concept_probs: NDArray[np.float32] | None,
    pred_mask: NDArray[np.int64] | None,
    class_palette: NDArray[np.uint8],
    mode: str,
    rank_index: dict[str, list[tuple[int, str]]],
    rank_palettes: dict[str, dict[str, NDArray[np.uint8]]],
    algae_idx: int | None,
    bleached_idx: int | None,
    opacity: float,
) -> NDArray[np.uint8]:
    """Render the prediction overlay for any of the supported modes.

    - ``classes``: same as :func:`compose_overlay` (palette[mask] at slider opacity).
    - ``kingdom`` / ``phylum`` / ``class`` / ``order`` / ``family`` / ``genus``:
      per-pixel argmax across that rank's concept channels, colored by the
      rank palette; per-pixel alpha = sigmoid of the argmax channel × slider.
    - ``algae`` / ``bleached``: solid color, per-pixel alpha = that one
      concept's sigmoid × slider.

    Returns the source-resolution composite; callers still need
    :func:`resize_for_display` for the Gradio image.
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))

    if mode == "classes":
        if pred_mask is None:
            return display_rgb
        return compose_overlay(display_rgb, pred_mask, class_palette, opacity)

    if concept_probs is None:
        return display_rgb

    H, W = display_rgb.shape[:2]

    if mode in RANK_ORDER:
        entries = rank_index.get(mode, [])
        palette = rank_palettes.get(mode, {})
        if not entries or not palette:
            return display_rgb
        channel_idxs = np.asarray([idx for idx, _ in entries], dtype=np.int64)
        values = [val for _, val in entries]
        rank_probs = concept_probs[channel_idxs]  # (K, H, W)
        argmax = rank_probs.argmax(axis=0)        # (H, W) — index into ``values``
        alpha = np.take_along_axis(rank_probs, argmax[None, ...], axis=0)[0]
        alpha = alpha * opacity

        color_lut = np.zeros((len(values), 3), dtype=np.uint8)
        for i, value in enumerate(values):
            rgb = palette.get(value)
            if rgb is not None:
                color_lut[i] = rgb
        color_per_pixel = color_lut[argmax]  # (H, W, 3)
        return _blend(display_rgb, color_per_pixel, alpha)

    if mode == "algae":
        if algae_idx is None:
            return display_rgb
        alpha = concept_probs[algae_idx] * opacity
        color = np.broadcast_to(np.array(ALGAE_OVERLAY_RGB, dtype=np.uint8), (H, W, 3))
        return _blend(display_rgb, color, alpha)

    if mode == "bleached":
        if bleached_idx is None:
            return display_rgb
        alpha = concept_probs[bleached_idx] * opacity
        color = np.broadcast_to(np.array(BLEACHED_OVERLAY_RGB, dtype=np.uint8), (H, W, 3))
        return _blend(display_rgb, color, alpha)

    return display_rgb


# ---------------------------------------------------------------------------
# Click-side renderers (top-3 classes + taxonomy tree)
# ---------------------------------------------------------------------------


_PANEL_TITLE_STYLE = (
    "margin:0 0 2px 0;font-size:13px;font-weight:600;color:#555;"
    "text-transform:uppercase;letter-spacing:0.04em"
)


def _panel_html(title: str, body: str) -> str:
    """Wrap ``body`` with a compact, uppercase section title.

    When ``title`` is empty/falsy the title row is omitted entirely so the
    section blends in with adjacent content (used for the conditional
    concepts panel, which carries its own per-section titles).
    """
    title_html = (
        f'<div style="{_PANEL_TITLE_STYLE}">{title}</div>' if title else ""
    )
    return (
        f'<div style="padding:2px 2px 4px 2px">'
        f'{title_html}'
        f'{body}'
        f'</div>'
    )


def render_top_classes_html(
    items: list[tuple[str, float]],
    title: str = "Top classes at clicked pixel",
    colors: list[tuple[int, int, int]] | None = None,
) -> str:
    """Render top-K classes as inline spans whose font-size scales with softmax.

    ``items`` is ``[(class_name, softmax_prob), ...]`` sorted high-to-low.
    Larger probability → bigger and more saturated text; small probabilities
    fade out toward gray so the top item visually dominates.

    When ``colors`` is provided (parallel to ``items``), the probability
    suffix is rendered in that color — useful for mirroring the overlay's
    class palette so a viewer can map each line to the on-image color.

    The ``title`` is rendered as a block-level heading above the chips
    (using the shared ``mermaid-section-title`` CSS class) so the title and
    chips read as one labelled section. The empty-state placeholder still
    uses a small italic hint.
    """
    title_html = (
        f'<div class="mermaid-section-title" style="display:block">{title}</div>'
        if title
        else ""
    )
    if not items:
        empty = (
            '<div style="color:#888;font-style:italic;font-size:12px">'
            'Click a pixel to see top classes.</div>'
        )
        return (
            f'<div style="padding:0;margin:0;line-height:1.25">'
            f'{title_html}{empty}</div>'
        )
    spans: list[str] = []
    for i, (name, p) in enumerate(items):
        p_clamped = float(np.clip(p, 0.0, 1.0))
        size_px = 14.0 + 42.0 * p_clamped
        opacity_pct = 35 + int(65 * p_clamped)
        if colors is not None and i < len(colors):
            r, g, b = colors[i]
            prob_style = (
                f"font-size:0.55em;color:rgb({r},{g},{b});"
                "font-weight:600;background:rgba(0,0,0,0.04);"
                "padding:1px 4px;border-radius:3px;margin-left:4px"
            )
        else:
            prob_style = "font-size:0.55em;opacity:0.7;margin-left:4px"
        spans.append(
            f'<span style="font-size:{size_px:.0f}px; margin-right:18px; '
            f'opacity:{opacity_pct / 100:.2f}; vertical-align:middle;">'
            f'{name} <small style="{prob_style}">{p_clamped:.2f}</small>'
            f'</span>'
        )
    return (
        f'<div style="padding:0;margin:0;line-height:1.25">'
        f'{title_html}'
        f'<div>{"".join(spans)}</div></div>'
    )


def render_concept_chip_html(
    sections: list[tuple[str, list[tuple[str, float]]]],
    empty_hint: str = "No conditional concepts active at this pixel.",
    title: str = "Conditional concepts",
) -> str:
    """Render one or more titled groups of concept chips for the gated panel.

    Each ``(section_title, items)`` becomes a small inline subtitle followed
    by inline chips. Chip font size scales with the sigmoid value (same
    convention as :func:`render_top_classes_html` but a touch smaller).
    Sections with no items are skipped. When all sections are empty, returns
    ``empty_hint`` rendered in muted italic.

    The ``title`` (when truthy) is rendered using the shared
    ``mermaid-section-title`` CSS class so the gated panel matches the
    sibling section headings ("Predicted MERMAID Classes" etc.).
    """
    title_html = (
        f'<div class="mermaid-section-title" style="display:block">{title}</div>'
        if title
        else ""
    )
    nonempty = [(section_title, items) for section_title, items in sections if items]
    if not nonempty:
        if not empty_hint:
            # Render nothing visible when there's no hint to show — keeps
            # the page clean when neither conditional gate is active.
            return f'<div style="padding:0;margin:0">{title_html}</div>' if title else ""
        body = (
            f'<div style="color:#888;font-style:italic;font-size:12px">'
            f'{empty_hint}</div>'
        )
        return f'<div style="padding:0;margin:0">{title_html}{body}</div>'

    parts: list[str] = []
    for section_title, items in nonempty:
        chips: list[str] = []
        for name, p in items:
            p_clamped = float(np.clip(p, 0.0, 1.0))
            size_px = 12.0 + 32.0 * p_clamped
            opacity_pct = 35 + int(65 * p_clamped)
            chips.append(
                f'<span style="font-size:{size_px:.0f}px; margin-right:16px; '
                f'opacity:{opacity_pct / 100:.2f}; vertical-align:middle;">'
                f'{name} <small style="font-size:0.55em;opacity:0.7">{p_clamped:.2f}</small>'
                f'</span>'
            )
        parts.append(
            f'<div style="margin-bottom:4px">'
            f'<div style="font-size:11px;color:#777;margin:2px 0">{section_title}</div>'
            f'<div style="line-height:1.25;padding:2px 2px">{"".join(chips)}</div>'
            f'</div>'
        )
    return (
        f'<div style="padding:0;margin:0">{title_html}{"".join(parts)}</div>'
    )


def render_top_bottom_other_html(
    top_items: list[tuple[str, float]],
    bottom_items: list[tuple[str, float]],
    title: str = "Predicted Concepts: Other",
    empty_hint: str = "Click a pixel to see other predicted concepts.",
    bottom_color: str = "#c0392b",
) -> str:
    """Render two labeled rows of "other" (non-rank) concept chips.

    ``top_items`` and ``bottom_items`` are each ``[(name, sigmoid), ...]``
    sorted highest→lowest. The top row uses the same probability-scaled
    sizing/opacity as :func:`render_concept_chip_html` so high-confidence
    concepts visually dominate. The bottom row is rendered at a fixed
    readable size in ``bottom_color`` (red by default) with full opacity,
    because by construction those sigmoids are near zero and would
    otherwise vanish under the probability-driven size/opacity scaling.

    When both lists are empty, falls back to ``empty_hint`` in muted italic.
    """
    title_html = (
        f'<div class="mermaid-section-title" style="display:block">{title}</div>'
        if title
        else ""
    )
    if not top_items and not bottom_items:
        body = (
            f'<div style="color:#888;font-style:italic;font-size:12px">'
            f'{empty_hint}</div>'
        )
        return f'<div style="padding:0;margin:0">{title_html}{body}</div>'

    def _top_chip(name: str, p: float) -> str:
        p_clamped = float(np.clip(p, 0.0, 1.0))
        size_px = 12.0 + 32.0 * p_clamped
        opacity_pct = 35 + int(65 * p_clamped)
        return (
            f'<span style="font-size:{size_px:.0f}px; margin-right:16px; '
            f'opacity:{opacity_pct / 100:.2f}; vertical-align:middle;">'
            f'{name} '
            f'<small style="font-size:0.55em;opacity:0.7">{p_clamped:.2f}</small>'
            f'</span>'
        )

    def _bottom_chip(name: str, p: float) -> str:
        p_clamped = float(np.clip(p, 0.0, 1.0))
        return (
            f'<span style="font-size:18px; margin-right:16px; '
            f'color:{bottom_color}; font-weight:600; vertical-align:middle;">'
            f'{name} '
            f'<small style="font-size:0.65em;color:{bottom_color};opacity:0.85">'
            f'{p_clamped:.2f}</small></span>'
        )

    rows: list[str] = []
    if top_items:
        chips = "".join(_top_chip(n, p) for n, p in top_items)
        rows.append(
            '<div style="margin-bottom:6px">'
            '<div style="font-size:14px;color:#444;margin:2px 0;'
            'font-weight:600;text-transform:uppercase;letter-spacing:0.04em">'
            f'Top {len(top_items)}'
            '</div>'
            f'<div style="line-height:1.25;padding:2px 2px">{chips}</div>'
            '</div>'
        )
    if bottom_items:
        chips = "".join(_bottom_chip(n, p) for n, p in bottom_items)
        rows.append(
            '<div style="margin-bottom:4px">'
            f'<div style="font-size:14px;color:{bottom_color};margin:2px 0;'
            'font-weight:600;text-transform:uppercase;letter-spacing:0.04em">'
            f'Bottom {len(bottom_items)}'
            '</div>'
            f'<div style="line-height:1.25;padding:2px 2px">{chips}</div>'
            '</div>'
        )
    return (
        f'<div style="padding:0;margin:0">{title_html}{"".join(rows)}</div>'
    )


def render_taxonomy_tree(
    concept_probs_at_pixel: NDArray[np.float32] | None,
    rank_index: dict[str, list[tuple[int, str]]],
    parents: dict[str, str],
    top_k: int = 3,
):
    """Build a matplotlib figure showing top-``k`` concepts per rank with parent edges.

    Each rank becomes a column from left (kingdom) to right (genus). At every
    column the ``top_k`` concept channels with the highest sigmoid are drawn
    as text-only nodes; font size scales with sigmoid. Edges are drawn only
    between a node and its actual taxonomic parent when both happen to be
    selected at adjacent ranks; otherwise the deeper node is rendered
    standalone ("orphaned").

    Returns a ``matplotlib.figure.Figure`` ready to hand to ``gr.Plot``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.0))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.02)
    ax.set_axis_off()
    ax.set_xlim(-0.5, len(RANK_ORDER) - 0.5)
    ax.set_ylim(-0.55, top_k + 0.05)

    if concept_probs_at_pixel is None or concept_probs_at_pixel.size == 0:
        ax.text(
            0.5, 0.5,
            "Click a pixel to see the taxonomy.",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#666",
        )
        return fig

    # For each rank, pick the top-k channels by sigmoid. Store both:
    #   selected[rank] = list[(value_name, prob, y_pos)]  (top-down: y=top_k-1 at top)
    # and a fast lookup ``selected_lookup[rank][value_name] = y_pos`` for
    # finding a parent's position when drawing edges.
    selected: dict[str, list[tuple[str, float, int]]] = {}
    selected_lookup: dict[str, dict[str, int]] = {}
    for rank in RANK_ORDER:
        entries = rank_index.get(rank, [])
        if not entries:
            selected[rank] = []
            selected_lookup[rank] = {}
            continue
        idxs = np.asarray([idx for idx, _ in entries], dtype=np.int64)
        values = [val for _, val in entries]
        probs = concept_probs_at_pixel[idxs]
        order = np.argsort(probs)[::-1][:top_k]
        chosen: list[tuple[str, float, int]] = []
        lookup: dict[str, int] = {}
        for rank_pos, j in enumerate(order):
            j_int = int(j)
            value = values[j_int]
            p = float(probs[j_int])
            y = top_k - 1 - rank_pos  # top of column = best
            chosen.append((value, p, y))
            lookup[value] = y
        selected[rank] = chosen
        selected_lookup[rank] = lookup

    # Column headers, placed just above the top node row inside the tightened ylim.
    for x, rank in enumerate(RANK_ORDER):
        ax.text(
            x, top_k - 0.25,
            rank,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#444",
        )

    # Edges first so node text renders on top of any lines.
    for x_child, child_rank in enumerate(RANK_ORDER):
        if x_child == 0:
            continue
        parent_rank = RANK_ORDER[x_child - 1]
        parent_lookup = selected_lookup.get(parent_rank, {})
        for value, p_child, y_child in selected[child_rank]:
            full_child = f"{child_rank}__{value}"
            full_parent = parents.get(full_child)
            if full_parent is None:
                continue
            parent_value = full_parent.split("__", 1)[1] if "__" in full_parent else full_parent
            if parent_value not in parent_lookup:
                continue
            y_parent = parent_lookup[parent_value]
            line_alpha = max(0.15, min(1.0, p_child))
            line_width = 0.5 + 3.5 * max(0.0, min(1.0, p_child))
            ax.plot(
                [x_child - 1, x_child],
                [y_parent, y_child],
                color="#444",
                alpha=line_alpha,
                linewidth=line_width,
                solid_capstyle="round",
                zorder=1,
            )

    # Nodes (text only) — font size scales with sigmoid.
    for x, rank in enumerate(RANK_ORDER):
        for value, p, y in selected[rank]:
            p_clamped = float(np.clip(p, 0.0, 1.0))
            size = 8.0 + 16.0 * p_clamped
            text_alpha = 0.35 + 0.65 * p_clamped
            ax.text(
                x, y,
                f"{value}\n({p_clamped:.2f})",
                ha="center", va="center",
                fontsize=size,
                color="#111",
                alpha=text_alpha,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="#ccc",
                    alpha=0.9,
                    linewidth=0.5,
                ),
                zorder=2,
            )

    fig.tight_layout()
    return fig
