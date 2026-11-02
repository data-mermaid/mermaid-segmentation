"""Orthomosaic CBM prediction demo.

Runs the concept-bottleneck model over overlapping 2048x2048 patches of a large
orthomosaic (each patch resized to 512x512 for inference), stitches predictions
back to full resolution with a max-confidence merge, and renders:

1. MERMAID class argmax (colorized) + a legend PDF.
2. A 6-way benthos argmax (Hard Coral / Soft Coral / Sand / Rock / Sponge /
   Other) + a legend PDF.
3. A cnidarian-genus overlay (opacity 1) keeping confident, large genera + a
   legend PDF.
4. A morphology overlay (weighted argmax over a few morph concepts) blended over
   the RGB where the winning concept is confident + a legend PDF.

Only small per-pixel reductions are kept at full resolution (never the full
concept stack), so memory stays bounded even for ~130 MP mosaics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from inference import (  # noqa: E402
    build_model,
    build_transforms,
    load_artifacts,
    predict,
    preprocess,
    tile_starts,
)
from rendering import make_color_palette  # noqa: E402

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = _DEMO_DIR.parent

# --------------------------------------------------------------------------- #
# Configuration (edit these).
# --------------------------------------------------------------------------- #
DEFAULT_IMAGE = Path(
    "/Users/jonathan/mit/aggregate_external_datasets/datasets/benthos_yuval/RS24.png"
)
DEFAULT_CHECKPOINT = Path("/Users/jonathan/Downloads/model_epoch162")
DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"
DEFAULT_TAXONOMY_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"

PATCH_SIZE = 2048
INFER_SIZE = 512
TILE_OVERLAP = 0.2

# Output 1 (MERMAID class): drop classes with fewer than this many pixels.
CLASS_MIN_PIXELS = 1000

# Output 5 (genus overlay) filtering.
GENUS_CONF_THRESHOLD = 0.85
GENUS_MIN_PIXELS = 1000

# Output 6 (morphology overlay) gate on the *unweighted* winning concept prob.
MORPH_PROB_THRESHOLD = 0.6

# Output 3: 6-way benthos argmax. Each entry is (display name, [concept channels
# multiplied together], hardcoded RGB color you can edit).
BENTHOS_SPEC: list[tuple[str, list[str], tuple[int, int, int]]] = [
    ("Hard Coral", ["class__hexacorallia", "live"], (214, 39, 40)),
    ("Soft Coral", ["class__octocorallia", "live"], (255, 127, 14)),
    ("Sand", ["sand"], (237, 201, 145)),
    ("Rock", ["hard_substrate"], (140, 140, 140)),
    ("Sponge", ["phylum__porifera"], (148, 103, 189)),
    ("Other", ["anthropogenic"], (44, 160, 44)),
]

# Output 6: weighted morphology argmax. Each entry is (display name, concept
# channel, argmax weight/multiplier, hardcoded RGB color you can edit).
MORPH_SPEC: list[tuple[str, str, float, tuple[int, int, int]]] = [
    ("Flabello-Meandroid", "lobed_brain", 2.0, (31, 119, 180)),
    ("Meandroid", "brain", 1.5, (255, 127, 14)),
    ("Massive", "massive", 1.0, (44, 160, 44)),
    ("Branching", "branching", 1.0, (214, 39, 40)),
    ("Free-Living", "free_living", 1.0, (148, 103, 189)),
]

# Optional hardcoded genus colors (display genus value -> RGB). Genera not listed
# get an automatically generated distinct color.
GENUS_COLORS: dict[str, tuple[int, int, int]] = {}

# Force these MERMAID classes to reuse the benthos concept colors so the class
# map (output 1) and the benthos map (output 3) are visually consistent. Maps a
# MERMAID class label (from id2label.json) -> a benthos display name in
# ``BENTHOS_SPEC``. Edit freely.
CLASS_TO_BENTHOS_COLOR: dict[str, str] = {
    "soft coral": "Soft Coral",
    "sand": "Sand",
    "turf algae": "Rock",
    "bare substrate": "Rock",
    "sponge": "Sponge",
    "hard coral": "Hard Coral",
    "human": "Other",
    "porites":"Xenia"
}


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _concept_names(concept_id2name: dict[int, str]) -> list[str]:
    return [name for _, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0]))]


def _name_to_index(concept_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(concept_names)}


def _cnidarian_genus_channels(
    taxonomy_csv: Path, name2idx: dict[str, int]
) -> list[tuple[int, str]]:
    """Return ``(channel_index, genus_value)`` for genera under phylum Cnidaria."""
    df = pd.read_csv(taxonomy_csv)
    if "phylum" not in df.columns or "genus" not in df.columns:
        raise ValueError(f"{taxonomy_csv} must contain 'phylum' and 'genus' columns")
    placeholders = {"not_given", "none", "", "nan"}
    genera: set[str] = set()
    sub = df[df["phylum"].astype(str).str.lower() == "cnidaria"]
    for value in sub["genus"].dropna():
        v = str(value).strip().lower()
        if v and v not in placeholders:
            genera.add(v)
    entries: list[tuple[int, str]] = []
    for value in sorted(genera):
        channel = f"genus__{value}"
        idx = name2idx.get(channel)
        if idx is not None:
            entries.append((idx, value))
    return entries


def _resize_to(arr: NDArray, out_hw: tuple[int, int], *, nearest: bool = True) -> NDArray:
    """Resize a 2D array to ``(height, width)`` (cv2 takes width, height)."""
    height, width = out_hw
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if arr.dtype == np.float16:
        resized = cv2.resize(arr.astype(np.float32), (width, height), interpolation=interp)
        return resized.astype(np.float16)
    return cv2.resize(arr, (width, height), interpolation=interp)


def _save_big_png(rgb: NDArray[np.uint8], path: Path) -> None:
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


def _auto_colors(n: int) -> list[tuple[int, int, int]]:
    if n == 0:
        return []
    cmap = plt.colormaps["hsv"]
    return [
        tuple(int(c * 255) for c in cmap(i / max(1, n))[:3])  # type: ignore[misc]
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Full-resolution merge canvases (max-confidence overlap resolution).
# --------------------------------------------------------------------------- #
class MergeCanvas:
    """Holds an index map, a confidence map, and optional extra aligned maps.

    Overlapping patches are merged by keeping, per pixel, the values from the
    patch with the highest confidence.
    """

    def __init__(
        self,
        height: int,
        width: int,
        idx_dtype: np.dtype,
        extra_dtypes: dict[str, np.dtype] | None = None,
    ) -> None:
        self.idx = np.zeros((height, width), dtype=idx_dtype)
        self.conf = np.full((height, width), -1.0, dtype=np.float16)
        self.extra: dict[str, NDArray] = {
            name: np.zeros((height, width), dtype=dtype)
            for name, dtype in (extra_dtypes or {}).items()
        }

    def merge(
        self,
        y0: int,
        x0: int,
        idx_patch: NDArray,
        conf_patch: NDArray[np.float32],
        extra_patches: dict[str, NDArray] | None = None,
    ) -> None:
        y1, x1 = y0 + idx_patch.shape[0], x0 + idx_patch.shape[1]
        region_conf = self.conf[y0:y1, x0:x1].astype(np.float32)
        better = conf_patch > region_conf
        self.conf[y0:y1, x0:x1] = np.where(
            better, conf_patch.astype(np.float16), self.conf[y0:y1, x0:x1]
        )
        self.idx[y0:y1, x0:x1] = np.where(
            better, idx_patch.astype(self.idx.dtype), self.idx[y0:y1, x0:x1]
        )
        for name, patch in (extra_patches or {}).items():
            target = self.extra[name]
            target[y0:y1, x0:x1] = np.where(
                better, patch.astype(target.dtype), target[y0:y1, x0:x1]
            )


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
    taxonomy_csv: Path,
    device: torch.device,
    patch_size: int,
    tile_overlap: float,
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
    name2idx = _name_to_index(concept_names)

    # Resolve channel indices for each output.
    benthos_channel_groups = [
        [name2idx[name] for name in names] for _, names, _ in BENTHOS_SPEC
    ]
    morph_channels = [name2idx[name] for _, name, _, _ in MORPH_SPEC]
    morph_weights = np.array([w for _, _, w, _ in MORPH_SPEC], dtype=np.float32)
    genus_entries = _cnidarian_genus_channels(taxonomy_csv, name2idx)
    if not genus_entries:
        raise ValueError("No cnidarian genus channels resolved from taxonomy CSV.")
    genus_channels = [idx for idx, _ in genus_entries]
    genus_values = [value for _, value in genus_entries]
    logger.info("Resolved %d cnidarian genus channels.", len(genus_channels))

    # Load full-resolution RGB mosaic.
    logger.info("Loading orthomosaic %s", image_path)
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    logger.info("Orthomosaic size: %dx%d", width, height)

    # Full-resolution canvases (small per-pixel reductions only).
    class_canvas = MergeCanvas(height, width, np.dtype(np.uint8))
    benthos_canvas = MergeCanvas(height, width, np.dtype(np.uint8))
    genus_canvas = MergeCanvas(height, width, np.dtype(np.uint16))
    morph_canvas = MergeCanvas(
        height, width, np.dtype(np.uint8), extra_dtypes={"prob": np.dtype(np.float16)}
    )

    y_starts = tile_starts(height, patch_size, tile_overlap)
    x_starts = tile_starts(width, patch_size, tile_overlap)
    total = len(y_starts) * len(x_starts)
    logger.info(
        "Tiling %d patches (%d x %d) of size %d, overlap %.0f%%",
        total,
        len(y_starts),
        len(x_starts),
        patch_size,
        tile_overlap * 100,
    )

    with tqdm(total=total, desc="Patches") as progress:
        for y0 in y_starts:
            for x0 in x_starts:
                ph = min(patch_size, height - y0)
                pw = min(patch_size, width - x0)
                patch_rgb = rgb[y0 : y0 + ph, x0 : x0 + pw]
                image_tensor, _ = preprocess(patch_rgb, model_transform, display_transform)
                class_probs, concept_probs, _ = predict(model, image_tensor.to(device))

                # --- MERMAID class (output 1) ---
                class_idx = class_probs.argmax(axis=0).astype(np.uint8)
                class_conf = class_probs.max(axis=0).astype(np.float32)

                # --- Benthos 6-way (output 3) ---
                benthos_stack = np.stack(
                    [
                        np.prod([concept_probs[c] for c in group], axis=0)
                        for group in benthos_channel_groups
                    ],
                    axis=0,
                )
                benthos_idx = benthos_stack.argmax(axis=0).astype(np.uint8)
                benthos_conf = benthos_stack.max(axis=0).astype(np.float32)

                # --- Cnidarian genus (output 5) ---
                genus_stack = concept_probs[genus_channels]
                genus_idx = genus_stack.argmax(axis=0).astype(np.uint16)
                genus_conf = genus_stack.max(axis=0).astype(np.float32)

                # --- Morphology weighted argmax (output 6) ---
                morph_stack = concept_probs[morph_channels]  # (M, h, w) unweighted probs
                weighted = morph_stack * morph_weights[:, None, None]
                morph_idx = weighted.argmax(axis=0).astype(np.uint8)
                morph_weighted_conf = weighted.max(axis=0).astype(np.float32)
                morph_unweighted = np.take_along_axis(
                    morph_stack, morph_idx[None, ...].astype(np.intp), axis=0
                )[0].astype(np.float32)

                # Upsample reductions to patch resolution and merge.
                out_hw = (ph, pw)
                class_canvas.merge(
                    y0,
                    x0,
                    _resize_to(class_idx, out_hw),
                    _resize_to(class_conf, out_hw),
                )
                benthos_canvas.merge(
                    y0,
                    x0,
                    _resize_to(benthos_idx, out_hw),
                    _resize_to(benthos_conf, out_hw),
                )
                genus_canvas.merge(
                    y0,
                    x0,
                    _resize_to(genus_idx, out_hw),
                    _resize_to(genus_conf, out_hw),
                )
                morph_canvas.merge(
                    y0,
                    x0,
                    _resize_to(morph_idx, out_hw),
                    _resize_to(morph_weighted_conf, out_hw),
                    extra_patches={"prob": _resize_to(morph_unweighted, out_hw)},
                )
                progress.update(1)

    # ----------------------------------------------------------------- #
    # Output 1 + 2: MERMAID class argmax + legend.
    # ----------------------------------------------------------------- #
    num_classes = max(artifacts.id2label.keys()) + 1
    class_palette = make_color_palette(num_classes)
    # Override select class colors so they match the benthos concept colors.
    benthos_color_by_name = {name: color for name, _, color in BENTHOS_SPEC}
    label2id = {name: cid for cid, name in artifacts.id2label.items()}
    for class_label, benthos_name in CLASS_TO_BENTHOS_COLOR.items():
        cid = label2id.get(class_label)
        color = benthos_color_by_name.get(benthos_name)
        if cid is not None and color is not None and cid < num_classes:
            class_palette[cid] = np.array(color, dtype=np.uint8)

    # Drop classes with too few predicted pixels (remap them to background id 0).
    class_idx_full = class_canvas.idx
    class_counts = np.bincount(class_idx_full.reshape(-1), minlength=num_classes)
    small_classes = [
        cid
        for cid in range(num_classes)
        if cid != 0 and 0 < class_counts[cid] < CLASS_MIN_PIXELS
    ]
    if small_classes:
        drop = np.isin(class_idx_full, np.array(small_classes, dtype=class_idx_full.dtype))
        class_idx_full = np.where(drop, 0, class_idx_full).astype(class_canvas.idx.dtype)
        logger.info(
            "Dropped %d MERMAID classes with <%d px.", len(small_classes), CLASS_MIN_PIXELS
        )

    class_rgb = class_palette[class_idx_full]
    _save_big_png(class_rgb, output_dir / "mermaid_class_argmax.png")

    present = [
        int(i)
        for i in sorted(np.unique(class_idx_full))
        if int(i) != 0 and class_counts[int(i)] >= CLASS_MIN_PIXELS
    ]
    class_legend = [
        (artifacts.id2label.get(i, f"class_{i}").title(), tuple(int(v) for v in class_palette[i]))
        for i in present
    ]
    _save_legend_pdf(class_legend, output_dir / "mermaid_legend.pdf", title="MERMAID classes")

    # ----------------------------------------------------------------- #
    # Output 3 + 4: benthos argmax + legend.
    # ----------------------------------------------------------------- #
    benthos_lut = np.array([color for _, _, color in BENTHOS_SPEC], dtype=np.uint8)
    benthos_rgb = benthos_lut[benthos_canvas.idx]
    _save_big_png(benthos_rgb, output_dir / "benthos_argmax.png")
    benthos_legend = [(name, color) for name, _, color in BENTHOS_SPEC]
    _save_legend_pdf(benthos_legend, output_dir / "benthos_legend.pdf", title="Benthos")

    # ----------------------------------------------------------------- #
    # Output 5: cnidarian genus overlay (opacity 1) + legend.
    # ----------------------------------------------------------------- #
    genus_mask = genus_canvas.conf.astype(np.float32) > GENUS_CONF_THRESHOLD
    genus_idx_full = genus_canvas.idx
    kept: list[tuple[int, str, int]] = []  # (local index, genus value, pixel count)
    for local_idx, value in enumerate(genus_values):
        count = int(np.count_nonzero(genus_mask & (genus_idx_full == local_idx)))
        if count > GENUS_MIN_PIXELS:
            kept.append((local_idx, value, count))
    kept.sort(key=lambda t: t[2], reverse=True)
    logger.info("Retained %d cnidarian genera (>%d px).", len(kept), GENUS_MIN_PIXELS)

    genus_overlay = rgb.copy()
    auto = _auto_colors(len(kept))
    genus_legend: list[tuple[str, tuple[int, int, int]]] = []
    for order, (local_idx, value, _count) in enumerate(kept):
        color = GENUS_COLORS.get(value, auto[order])
        sel = genus_mask & (genus_idx_full == local_idx)
        genus_overlay[sel] = np.array(color, dtype=np.uint8)
        genus_legend.append((value.title(), tuple(int(c) for c in color)))
    _save_big_png(genus_overlay, output_dir / "genus_overlay.png")
    _save_legend_pdf(
        genus_legend or [("(none retained)", (0, 0, 0))],
        output_dir / "legend_genus.pdf",
        title="Cnidarian genera",
    )

    # ----------------------------------------------------------------- #
    # Output 6: morphology overlay + legend.
    # ----------------------------------------------------------------- #
    morph_lut = np.array([color for _, _, _, color in MORPH_SPEC], dtype=np.uint8)
    morph_prob = morph_canvas.extra["prob"].astype(np.float32)
    morph_mask = morph_prob > MORPH_PROB_THRESHOLD
    morph_overlay = rgb.copy()
    morph_colored = morph_lut[morph_canvas.idx]
    morph_overlay[morph_mask] = morph_colored[morph_mask]
    _save_big_png(morph_overlay, output_dir / "morpho_overlay.png")
    morph_legend = [(name, color) for name, _, _, color in MORPH_SPEC]
    _save_legend_pdf(morph_legend, output_dir / "morpho_legend.pdf", title="Morphology")

    logger.info("Done. Outputs written to %s", output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE, help="Input orthomosaic")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint")
    p.add_argument(
        "--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Model config YAML"
    )
    p.add_argument("--id2label", type=Path, default=None, help="Path to id2label.json")
    p.add_argument(
        "--concept-id2name", type=Path, default=None, help="Path to concept_id2name.json"
    )
    p.add_argument(
        "--taxonomy-csv", type=Path, default=DEFAULT_TAXONOMY_CSV, help="class_to_concepts CSV"
    )
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE, help="Patch size in pixels")
    p.add_argument(
        "--tile-overlap", type=float, default=TILE_OVERLAP, help="Minimum tile overlap fraction"
    )
    p.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device %s", device)
    run(
        image_path=args.image,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
        taxonomy_csv=args.taxonomy_csv,
        device=device,
        patch_size=args.patch_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
