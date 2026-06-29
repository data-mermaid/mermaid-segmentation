"""Export structured CBM inference outputs for a single image.

Runs the demo CBM model and writes per-rank taxonomy arrays, multi-hot concepts,
and MERMAID class predictions at 512x512 resolution.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from inference import (  # noqa: E402
    build_model,
    build_transforms,
    load_artifacts,
    predict,
    preprocess,
)
from mermaidseg.dataset_reconciliation.concepts import (  # noqa: E402
    TAXONOMIC_CONCEPTS,
    parse_concept_rank,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = _DEMO_DIR.parent
DEFAULT_CHECKPOINT = Path("/Users/jonathan/Downloads/model_epoch14")
DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"
MULTI_HOT_THRESHOLD = 0.5


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="Path to input image")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write output arrays and JSON files",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model config YAML",
    )
    p.add_argument("--id2label", type=Path, default=None, help="Path to id2label.json")
    p.add_argument(
        "--concept-id2name",
        type=Path,
        default=None,
        help="Path to concept_id2name.json",
    )
    p.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    return p


def _concept_names(concept_id2name: dict[int, str]) -> list[str]:
    return [name for _, name in sorted(concept_id2name.items(), key=lambda kv: int(kv[0]))]


def _rank_channel_indices(concept_names: list[str], rank: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for idx, name in enumerate(concept_names):
        parsed_rank, value = parse_concept_rank(name)
        if parsed_rank == rank:
            entries.append((idx, value))
    return entries


def _multihot_channel_indices(concept_names: list[str]) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for idx, name in enumerate(concept_names):
        parsed_rank, value = parse_concept_rank(name)
        if parsed_rank is None:
            entries.append((idx, value))
    return entries


def _write_id2name_json(path: Path, values: list[str]) -> None:
    payload = {str(i): value for i, value in enumerate(values)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _save_hard_label_png(path: Path, hard: np.ndarray) -> None:
    """Save per-pixel class indices as a single-channel PNG (uint16 if labels exceed 255)."""
    labels = np.squeeze(hard, axis=-1) if hard.ndim == 3 else hard
    if labels.max() > 255:
        Image.fromarray(labels.astype(np.uint16), mode="I;16").save(path)
    else:
        Image.fromarray(labels.astype(np.uint8), mode="L").save(path)


def _input_size(model_cfg: dict) -> tuple[int, int]:
    size = model_cfg.get("input_size", [512, 512])
    return int(size[0]), int(size[1])


def export_structured_outputs(
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
    model_transform, display_transform = build_transforms(_input_size(artifacts.model_cfg))

    image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    image_tensor, display_image = preprocess(image_rgb, model_transform, display_transform)
    class_probs, concept_probs, _ = predict(model, image_tensor.to(device))

    names = _concept_names(artifacts.concept_id2name)

    Image.fromarray(display_image).save(output_dir / "rgb_image.png")

    class_probs_hwc = class_probs.transpose(1, 2, 0).astype(np.float32)
    np.save(output_dir / "mermaid_classification_probabilities.npy", class_probs_hwc)
    class_hard = class_probs_hwc.argmax(axis=2, keepdims=True).astype(np.int64)
    _save_hard_label_png(output_dir / "mermaid_classification_hard.png", class_hard)
    _write_id2name_json(
        output_dir / "mermaid_classification.json",
        [artifacts.id2label[i] for i in range(max(artifacts.id2label) + 1)],
    )

    for rank in TAXONOMIC_CONCEPTS:
        entries = _rank_channel_indices(names, rank)
        if not entries:
            logger.warning("No concept channels found for rank %r; skipping.", rank)
            continue
        channel_idxs = [idx for idx, _ in entries]
        values = [value for _, value in entries]
        rank_probs = concept_probs[channel_idxs].transpose(1, 2, 0).astype(np.float32)
        np.save(output_dir / f"taxonomy_{rank}_probabilities.npy", rank_probs)
        rank_hard = rank_probs.argmax(axis=2, keepdims=True).astype(np.int64)
        _save_hard_label_png(output_dir / f"taxonomy_{rank}_hard.png", rank_hard)
        _write_id2name_json(output_dir / f"taxonomy_{rank}.json", values)

    multihot_entries = _multihot_channel_indices(names)
    multihot_idxs = [idx for idx, _ in multihot_entries]
    multihot_values = [value for _, value in multihot_entries]
    multihot_probs = concept_probs[multihot_idxs].transpose(1, 2, 0).astype(np.float32)
    np.save(output_dir / "multi_hot_probabilities.npy", multihot_probs)
    multihot_hard = (multihot_probs >= MULTI_HOT_THRESHOLD).astype(np.uint8)
    np.save(output_dir / "multi_hot_hard.npy", multihot_hard)
    _write_id2name_json(output_dir / "multi_hot.json", multihot_values)

    viewer_notebook = _DEMO_DIR / "visualize_structured_outputs.ipynb"
    if viewer_notebook.is_file():
        shutil.copy2(viewer_notebook, output_dir / viewer_notebook.name)

    logger.info("Wrote structured outputs to %s", output_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device %s", device)
    export_structured_outputs(
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
