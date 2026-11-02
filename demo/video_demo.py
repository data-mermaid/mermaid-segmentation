"""Offline CBM video demo with scheduled concept highlighting (inferno panel)."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from concept_expr import (  # noqa: E402
    ConceptResolver,
    concept_names_from_id2name,
    evaluate,
    is_classes_sentinel,
)
from inference import (  # noqa: E402
    DEFAULT_TILE_OVERLAP,
    build_model,
    build_transforms,
    load_artifacts,
    parse_processing_resolution,
    predict_tiled,
    tile_starts,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = _DEMO_DIR.parent
DEFAULT_MODEL_CONFIG = _REPO_ROOT / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"

INFERNO_MASK_THRESHOLD = 0.4


@dataclass(frozen=True)
class ScheduleRow:
    timestamp: float
    display_title: str
    display_subtitle: str
    expression: str


def load_schedule(csv_path: Path) -> list[ScheduleRow]:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "display_title", "display_subtitle", "expression"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Concepts CSV missing columns: {sorted(missing)}")
    rows = [
        ScheduleRow(
            timestamp=float(row.timestamp),
            display_title=str(row.display_title),
            display_subtitle=str(row.display_subtitle),
            expression=str(row.expression).strip(),
        )
        for row in df.itertuples(index=False)
    ]
    rows.sort(key=lambda r: r.timestamp)
    if not rows:
        raise ValueError(f"No schedule rows found in {csv_path}")
    return rows


def active_schedule_row(rows: list[ScheduleRow], time_sec: float) -> ScheduleRow:
    active = rows[0]
    for row in rows:
        if row.timestamp <= time_sec:
            active = row
        else:
            break
    return active


def _input_size(model_cfg: dict) -> tuple[int, int]:
    size = model_cfg.get("input_size", [512, 512])
    return int(size[0]), int(size[1])


def _colormap_rgb(values: NDArray[np.float32], cmap_name: str) -> NDArray[np.float32]:
    cmap = plt.colormaps[cmap_name]
    rgba = cmap(np.clip(values, 0.0, 1.0))
    return (rgba[..., :3] * 255.0).astype(np.float32)


def render_inferno_mask_panel(
    display_rgb: NDArray[np.uint8],
    values: NDArray[np.float32],
    *,
    threshold: float = INFERNO_MASK_THRESHOLD,
) -> NDArray[np.uint8]:
    """Highlight activated regions with inferno; below threshold stays RGB.

    For pixels with composite activation v > threshold:
        inner = v * inferno(v) + (1 - v) * RGB
    Otherwise the pixel stays RGB.
    """
    v = np.clip(values, 0.0, 1.0)
    mask = (v > threshold).astype(np.float32)
    rgb_f = display_rgb.astype(np.float32)
    inferno = _colormap_rgb(v, "inferno")
    inner = v[..., None] * inferno + (1.0 - v[..., None]) * rgb_f
    blended = (1.0 - mask[..., None]) * rgb_f + mask[..., None] * inner
    return np.clip(blended, 0, 255).astype(np.uint8)


def composite_values(
    expression: str,
    concept_probs: NDArray[np.float32],
    class_probs: NDArray[np.float32],
    resolver: ConceptResolver,
) -> NDArray[np.float32]:
    if is_classes_sentinel(expression):
        argmax = class_probs.argmax(axis=0)
        return np.take_along_axis(class_probs, argmax[None, ...], axis=0)[0].astype(np.float32)
    return evaluate(expression, concept_probs, resolver)


def render_highlight_panel(
    display_rgb: NDArray[np.uint8],
    expression: str,
    concept_probs: NDArray[np.float32],
    class_probs: NDArray[np.float32],
    resolver: ConceptResolver,
) -> NDArray[np.uint8]:
    values = composite_values(expression, concept_probs, class_probs, resolver)
    return render_inferno_mask_panel(display_rgb, values)


def _resize_rgb(image_rgb: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    width, height = size
    return np.asarray(
        Image.fromarray(image_rgb).resize((width, height), Image.BILINEAR),
        dtype=np.uint8,
    )


def _normalize_banner_text(text: str) -> str:
    """Replace Unicode symbols that OpenCV's Hershey fonts cannot render."""
    return (
        text.replace("\u00d7", "x")  # multiplication sign
        .replace("\u2212", "-")  # minus sign
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
    )


def _banner_layout(frame_height: int) -> tuple[int, float, float, int, int, int]:
    banner_height = max(120, int(frame_height * 0.1))
    title_scale = banner_height / 68.0
    subtitle_scale = banner_height / 92.0
    title_y = int(banner_height * 0.44)
    subtitle_y = int(banner_height * 0.82)
    thickness = max(2, int(round(banner_height / 48.0)))
    return banner_height, title_scale, subtitle_scale, title_y, subtitle_y, thickness


def draw_text_banner(
    frame_bgr: NDArray[np.uint8],
    title: str,
    subtitle: str,
) -> NDArray[np.uint8]:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    banner_height, title_scale, subtitle_scale, title_y, subtitle_y, thickness = _banner_layout(h)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    title = _normalize_banner_text(title)
    subtitle = _normalize_banner_text(subtitle)
    margin_x = max(16, int(w * 0.012))

    cv2.putText(
        out,
        title,
        (margin_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            out,
            subtitle,
            (margin_x, subtitle_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            subtitle_scale,
            (210, 210, 210),
            max(2, thickness - 1),
            cv2.LINE_AA,
        )
    return out


def render_video(
    input_video: Path,
    output_video: Path,
    schedule: list[ScheduleRow],
    checkpoint: Path,
    model_config: Path,
    *,
    id2label: Path | None = None,
    concept_id2name: Path | None = None,
    device: torch.device,
    output_fps: float | None = None,
    processing_resolution: tuple[int, int] | None = None,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
) -> None:
    artifacts = load_artifacts(
        checkpoint=checkpoint,
        model_config=model_config,
        id2label=id2label,
        concept_id2name=concept_id2name,
    )
    model = build_model(artifacts, device)
    input_size = _input_size(artifacts.model_cfg)
    model_transform, display_transform = build_transforms(input_size)

    if processing_resolution is not None:
        proc_h, proc_w = processing_resolution
        stride_h = max(1, int(input_size[0] * (1.0 - tile_overlap)))
        stride_w = max(1, int(input_size[1] * (1.0 - tile_overlap)))
        n_tiles = len(tile_starts(proc_h, input_size[0], tile_overlap)) * len(
            tile_starts(proc_w, input_size[1], tile_overlap)
        )
        logger.info(
            "Tiled inference at %dx%d (tile %dx%d, stride %dx%d, overlap %.0f%%, ~%d tiles/frame)",
            proc_h,
            proc_w,
            input_size[0],
            input_size[1],
            stride_h,
            stride_w,
            tile_overlap * 100,
            n_tiles,
        )

    concept_names = concept_names_from_id2name(artifacts.concept_id2name)
    num_concepts = model.concept_classifier.in_channels
    if len(concept_names) < num_concepts:
        concept_names.extend(f"concept_{i}" for i in range(len(concept_names), num_concepts))
    resolver = ConceptResolver(concept_names)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid video dimensions for {input_video}")

    target_fps = output_fps if output_fps is not None else source_fps
    frame_stride = max(1, int(round(source_fps / target_fps))) if target_fps > 0 else 1
    effective_fps = source_fps / frame_stride

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        effective_fps,
        (source_width, source_height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output video writer: {output_video}")

    total_out_frames = max(1, (frame_count + frame_stride - 1) // frame_stride) if frame_count else None
    frame_idx = 0
    written = 0

    with tqdm(total=total_out_frames, desc="Rendering video") as progress:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            time_sec = frame_idx / source_fps if source_fps > 0 else 0.0
            active = active_schedule_row(schedule, time_sec)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            class_probs, concept_probs, _, display_rgb = predict_tiled(
                model,
                frame_rgb,
                model_transform,
                display_transform,
                input_size,
                device,
                processing_resolution=processing_resolution,
                min_overlap=tile_overlap,
            )

            panel_size = (source_width, source_height)
            highlight = render_highlight_panel(
                display_rgb,
                active.expression,
                concept_probs,
                class_probs,
                resolver,
            )
            combined_rgb = _resize_rgb(highlight, panel_size)
            combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
            combined_bgr = draw_text_banner(
                combined_bgr,
                active.display_title,
                active.display_subtitle,
            )
            writer.write(combined_bgr)
            written += 1
            progress.update(1)
            frame_idx += 1

    cap.release()
    writer.release()
    logger.info(
        "Wrote %d frames to %s (source %.2f fps -> output %.2f fps)",
        written,
        output_video,
        source_fps,
        effective_fps,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_video", type=Path, help="Path to input video")
    p.add_argument("--output", type=Path, required=True, help="Path to output mp4")
    p.add_argument(
        "--concepts-csv",
        type=Path,
        required=True,
        help="CSV schedule: timestamp,display_title,display_subtitle,expression",
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
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
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate (subsamples source video when lower than source fps)",
    )
    p.add_argument(
        "--processing-resolution",
        default=None,
        help="Optional HEIGHTxWIDTH to resize frames before tiled inference (e.g. 1080x1920)",
    )
    p.add_argument(
        "--tile-overlap",
        type=float,
        default=DEFAULT_TILE_OVERLAP,
        help="Minimum tile overlap fraction (default 0.2 => 20%% overlap, stride = 0.8 * tile size)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schedule = load_schedule(args.concepts_csv)
    processing_resolution = (
        parse_processing_resolution(args.processing_resolution)
        if args.processing_resolution
        else None
    )
    logger.info("Using device %s", device)
    logger.info("Loaded %d schedule rows from %s", len(schedule), args.concepts_csv)
    render_video(
        input_video=args.input_video,
        output_video=args.output,
        schedule=schedule,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        id2label=args.id2label,
        concept_id2name=args.concept_id2name,
        device=device,
        output_fps=args.fps,
        processing_resolution=processing_resolution,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
