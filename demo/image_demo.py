"""Single-image CBM prediction demo.

Runs the concept-bottleneck model on one image (resized to the model input
resolution, like the Gradio app), upsamples the small per-pixel reductions back
to the original image resolution, and renders:

1. The MERMAID class argmax, colorized and blended over the RGB as
   ``0.7 * class_color + 0.3 * rgb``, plus a legend PDF that lists only the
   predicted classes covering more than ``CLASS_MIN_PIXELS`` pixels.
2. A handful of (composite) concept maps. Each concept expression is evaluated
   per-pixel to a value ``v`` in ``[0, 1]`` and rendered as::

       rgb                              where v <  0.5
       inferno(v) * 0.7 + 0.3 * rgb     where v >= 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from numpy.typing import NDArray
from PIL import Image

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from concept_expr import ConceptResolver, evaluate  # noqa: E402
from inference import (  # noqa: E402
    build_model,
    build_transforms,
    load_artifacts,
    predict,
    preprocess,
)
from rendering import make_color_palette  # noqa: E402

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = _DEMO_DIR.parent

# --------------------------------------------------------------------------- #
# Configuration (edit these).
# --------------------------------------------------------------------------- #
DEFAULT_IMAGE = _DEMO_DIR / "input.png"
DEFAULT_CHECKPOINT = Path("/Users/jonathan/Downloads/model_epoch101")
DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"

INFER_SIZE = 512

# Output 1 (MERMAID class): drop classes with fewer than this many pixels from
# the legend, and never legend the ``ignore`` class (id 0).
CLASS_MIN_PIXELS = 100
CLASS_COLOR_ALPHA = 0.7  # 0.7 * class_color + 0.3 * rgb

_SUBSTRATE_GRAY = (140, 140, 140)

# Override MERMAID class colors (id2label label -> RGB). The shared palette in
# ``rendering.py`` groups "bare substrate" with algae (green); force it gray.
CLASS_COLOR_OVERRIDES: dict[str, tuple[int, int, int]] = {
    "bare substrate": _SUBSTRATE_GRAY,
    "dark": _SUBSTRATE_GRAY,
    "sand": _SUBSTRATE_GRAY,
    "pavona": _SUBSTRATE_GRAY,
    "pocillopora": (0, 200, 220),
    "stylophora": (255, 40, 200),
}

# MERMAID class labels to always omit from the legend (even if present/large).
CLASS_LEGEND_EXCLUDE: frozenset[str] = frozenset({"dark", "sand", "pavona"})

# Output 2 (composite concepts). Each entry is (display name, expression). The
# expression language lives in ``concept_expr.py``: bare morphology/non-coral
# concepts resolve directly, and taxonomic channels use their full name
# (e.g. ``genus__acropora``).
CONCEPT_THRESHOLD = 0.6
CONCEPT_CMAP = "inferno"
CONCEPT_COLOR_ALPHA = 0.7  # inferno(v) * 0.7 + 0.3 * rgb
COMPOSITE_CONCEPTS: list[tuple[str, str]] = [
    ("branching", "branching"),
    ("plating", "plating"),
    ("encrusting", "encrusting"),
    ("massive", "massive"),
    ("algae", "algae"),
    ("acropora", "genus__acropora"),
    ("branching_not_acropora", "branching * (1 - genus__acropora)"),
    ("massive_not_brain", "massive * (1 - brain)"),
    ("branching_plating", "branching * plating"),
    ("corymbose", "corymbose"),
    ("massive_not_porites", "massive * (1 - genus__porites)"),
]


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _concept_names(concept_id2name: dict[int, str]) -> list[str]:
    return [name for _, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0]))]


def _resize_map(arr: NDArray, out_hw: tuple[int, int], *, nearest: bool) -> NDArray:
    """Resize a 2D map to ``(height, width)`` using PIL (avoids a cv2 dependency)."""
    height, width = out_hw
    resample = Image.NEAREST if nearest else Image.BILINEAR
    if nearest:
        pil = Image.fromarray(arr.astype(np.int32), mode="I")
        return np.asarray(pil.resize((width, height), resample), dtype=arr.dtype)
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(pil.resize((width, height), resample), dtype=np.float32)


def _save_png(rgb: NDArray[np.uint8], path: Path) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)
    logger.info("Wrote %s (%dx%d)", path, rgb.shape[1], rgb.shape[0])


def _save_legend_pdf(
    entries: list[tuple[str, tuple[int, int, int]]], path: Path, title: str | None = None
) -> None:
    handles = [
        Patch(facecolor=np.array(color, dtype=np.float32) / 255.0, edgecolor="black", label=name)
        for name, color in entries
    ]
    fig, ax = plt.subplots(figsize=(4.5, 0.5 * max(1, len(entries)) + 1.0))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", frameon=False, title=title, fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


# --------------------------------------------------------------------------- #
# Main pipeline.
# --------------------------------------------------------------------------- #
def run(
    image_path: Path,
    output_dir: Path,
    checkpoint: Path,
    model_config: Path,
    id2label: Path | None,
    concept_id2name: Path | None,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts(
        checkpoint=checkpoint,
        model_config=model_config,
        id2label=id2label,
        concept_id2name=concept_id2name,
    )
    model = build_model(artifacts, device)
    model_transform, display_transform = build_transforms((INFER_SIZE, INFER_SIZE))

    concept_names = _concept_names(artifacts.concept_id2name)
    num_concepts = model.concept_classifier.in_channels
    if len(concept_names) < num_concepts:
        concept_names.extend(f"concept_{i}" for i in range(len(concept_names), num_concepts))
    resolver = ConceptResolver.from_concept_names(concept_names)

    logger.info("Loading image %s", image_path)
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    logger.info("Image size: %dx%d", width, height)

    image_tensor, _ = preprocess(rgb, model_transform, display_transform)
    class_probs, concept_probs, _ = predict(model, image_tensor.to(device))
    logger.info(
        "Inference done: class_probs %s, concept_probs %s",
        class_probs.shape,
        concept_probs.shape,
    )

    rgb_f = rgb.astype(np.float32)

    # ----------------------------------------------------------------- #
    # Output 1: MERMAID class argmax blended over the RGB + legend.
    # ----------------------------------------------------------------- #
    num_classes = max(artifacts.id2label.keys()) + 1
    class_palette = make_color_palette(num_classes)
    label2id = {label: cid for cid, label in artifacts.id2label.items()}
    for label, color in CLASS_COLOR_OVERRIDES.items():
        cid = label2id.get(label)
        if cid is not None and cid < num_classes:
            class_palette[cid] = np.array(color, dtype=np.uint8)

    class_idx_small = class_probs.argmax(axis=0).astype(np.int32)
    class_idx = _resize_map(class_idx_small, (height, width), nearest=True)
    class_color = class_palette[class_idx].astype(np.float32)
    class_blend = CLASS_COLOR_ALPHA * class_color + (1.0 - CLASS_COLOR_ALPHA) * rgb_f
    _save_png(np.clip(class_blend, 0, 255).astype(np.uint8), output_dir / "mermaid_class.png")

    class_counts = np.bincount(class_idx.reshape(-1), minlength=num_classes)
    present = [
        cid
        for cid in range(num_classes)
        if cid != 0
        and class_counts[cid] > CLASS_MIN_PIXELS
        and artifacts.id2label.get(cid) not in CLASS_LEGEND_EXCLUDE
    ]
    present.sort(key=lambda cid: int(class_counts[cid]), reverse=True)
    class_legend = [
        (artifacts.id2label.get(cid, f"class_{cid}").title(), tuple(int(v) for v in class_palette[cid]))
        for cid in present
    ]
    _save_legend_pdf(
        class_legend or [("(none)", (0, 0, 0))],
        output_dir / "mermaid_legend.pdf",
        title="MERMAID classes",
    )

    # ----------------------------------------------------------------- #
    # Output 2: composite concept maps.
    # ----------------------------------------------------------------- #
    cmap = plt.colormaps[CONCEPT_CMAP]
    for name, expression in COMPOSITE_CONCEPTS:
        value_small = evaluate(expression, concept_probs, resolver)
        value = _resize_map(value_small.astype(np.float32), (height, width), nearest=False)
        value = np.clip(value, 0.0, 1.0)

        concept_color = (cmap(value)[..., :3] * 255.0).astype(np.float32)
        blended = CONCEPT_COLOR_ALPHA * concept_color + (1.0 - CONCEPT_COLOR_ALPHA) * rgb_f
        mask = (value >= CONCEPT_THRESHOLD)[..., None]
        out = np.where(mask, blended, rgb_f)
        _save_png(np.clip(out, 0, 255).astype(np.uint8), output_dir / f"concept_{name}.png")
        logger.info("Concept %r (%s): max=%.3f, >=%.1f in %.2f%% of pixels",
                    name, expression, float(value.max()), CONCEPT_THRESHOLD,
                    100.0 * float(mask.mean()))

    logger.info("Done. Outputs written to %s", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE, help="Input image")
    p.add_argument("--output-dir", type=Path, default=_DEMO_DIR / "image_out", help="Output dir")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint")
    p.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Model config YAML"
    )
    p.add_argument("--id2label", type=Path, default=None, help="Path to id2label.json")
    p.add_argument(
        "--concept-id2name", type=Path, default=None, help="Path to concept_id2name.json"
    )
    p.add_argument("--device", default=None, help="Torch device (default: cuda/mps if available)")
    return p


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else _default_device()
    logger.info("Using device %s", device)
    run(
        image_path=args.image,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
        device=device,
    )


if __name__ == "__main__":
    main()
