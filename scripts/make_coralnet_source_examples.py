#!/usr/bin/env python3
"""One-row CoralNet source figures, in the style of the dataset appendix.

The appendix shows ten images from each source in two rows of five, with a
legend in the right third of the text width. These figures keep that text
width (6.875 in) and the same image resolution (longest edge 1024 px), put five
images in a single row, and draw the point markers at twice the appendix radius.

Each CoralNet source is shown from its own uploads. The training blacklist is
not applied: dedup drops every image of some sources (3443 has none left, 6695
keeps one), and those sources are part of this set.

Usage:
    python scripts/make_coralnet_source_examples.py
    python scripts/make_coralnet_source_examples.py --n-sources 50 --n-images 5
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import shutil
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_appendix_data import as_rgb, dataset_rng, downscale  # noqa: E402
from make_appendix_figures import (  # noqa: E402
    CORALNET_LEGEND_FONTSIZE,
    CORALNET_LEGEND_WRAP,
    DEFAULT_POINT_SCALE,
    TEXT_WIDTH_IN,
    _sparse_legend,
    assign_colors,
    draw_points,
    setup_style,
)

logger = logging.getLogger("coralnet_source_examples")

_REPO_ROOT = _SCRIPTS_DIR.parent
_CACHE = _REPO_ROOT / ".cache" / "coralnet_source_examples"
DEFAULT_MANIFEST = _CACHE / "manifest.parquet"
DEFAULT_ID2NAME = _CACHE / "coralnet_id2name.json"
DEFAULT_IMAGE_CACHE = _CACHE / "images"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "coralnet_source_examples"
DEFAULT_DATA_CONFIG = _REPO_ROOT / "configs" / "data_config_trainval.yaml"
# Final config folds validation into train. Val and test for the full-res copies
# come from the split that defined those holdouts.
DEFAULT_SPLIT_CONFIG = _REPO_ROOT / "configs" / "data_config.yaml"

BUCKET = "dev-datamermaid-sm-sources"
MANIFEST_KEY = (
    "etl-outputs/coralnet/20260623_nogit/coralnet_training_manifest_20260623_nogit.parquet"
)
ID2NAME_KEY = "coralnet-public-images/temporary/coralnet_id2name.json"

# Always included. The rest are a deterministic sample of paper sources.
REQUIRED_SOURCE_IDS: tuple[int, ...] = (4906, 3443, 4810, 3653, 3368, 4495, 6543, 6695)

# Appendix markers use DEFAULT_POINT_SCALE. These are twice that radius.
POINT_SCALE = 2.0 * DEFAULT_POINT_SCALE
N_SOURCES = 50
N_IMAGES = 5
# Tried after the first ``N_IMAGES`` if a download fails.
SPARE_IMAGES = 2

# Same page width as the appendix. The legend is narrower than the appendix's
# right third, and the image row takes the rest.
FIG_WIDTH_IN = TEXT_WIDTH_IN
LEGEND_WIDTH_IN = 2.0
# ``\appH = 0.7\appW`` in the appendix.
CELL_HEIGHT_OVER_WIDTH = 0.7
IMAGE_GAP_IN = 1.0 / 72.0
LABEL_GAP_IN = 1.0 / 72.0
STRIP_LEGEND_GAP_IN = 0.03
MARGIN_IN = 0.008
LABEL_COL_IN = 0.11
SOURCE_LABEL_FONTSIZE = 7.0
VERTICAL_PAD_IN = 0.015


@dataclass(frozen=True)
class RowLayout:
    """Inch positions for one source figure. Every figure has the same width."""

    fig_width_in: float
    margin_in: float
    label_col_in: float
    strip_x_in: float
    strip_w_in: float
    legend_x_in: float
    legend_w_in: float
    cell_w_in: float
    cell_h_in: float
    image_gap_in: float
    n_images: int

    def cell_x(self, index: int) -> float:
        return self.strip_x_in + index * (self.cell_w_in + self.image_gap_in)


def row_layout(n_images: int = N_IMAGES) -> RowLayout:
    """Equal image cells, then a narrow gap, then the legend."""
    if n_images < 1:
        raise ValueError("n_images must be positive")
    legend_w = LEGEND_WIDTH_IN
    strip_x = MARGIN_IN + LABEL_COL_IN + LABEL_GAP_IN
    legend_x = FIG_WIDTH_IN - MARGIN_IN - legend_w
    strip_w = legend_x - STRIP_LEGEND_GAP_IN - strip_x
    if strip_w <= 0:
        raise ValueError("image row and legend overlap")
    gap_total = IMAGE_GAP_IN * (n_images - 1)
    cell_w = (strip_w - gap_total) / n_images
    if cell_w <= 0:
        raise ValueError("image cells have no width")
    return RowLayout(
        fig_width_in=FIG_WIDTH_IN,
        margin_in=MARGIN_IN,
        label_col_in=LABEL_COL_IN,
        strip_x_in=strip_x,
        strip_w_in=strip_w,
        legend_x_in=legend_x,
        legend_w_in=legend_w,
        cell_w_in=cell_w,
        cell_h_in=CELL_HEIGHT_OVER_WIDTH * cell_w,
        image_gap_in=IMAGE_GAP_IN,
        n_images=n_images,
    )


def fitted_size(image_wh: tuple[int, int], cell_w: float, cell_h: float) -> tuple[float, float]:
    """Largest axis-aligned size that fits the cell without changing aspect ratio."""
    img_w, img_h = image_wh
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"bad image size {image_wh}")
    aspect = img_w / img_h
    if aspect >= cell_w / cell_h:
        return cell_w, cell_w / aspect
    return cell_h * aspect, cell_h


def choose_sources(
    eligible: Sequence[int],
    *,
    n: int,
    required: Sequence[int],
    seed: int,
) -> list[int]:
    """``required`` first, then a deterministic sample, returned sorted."""
    required_ids = list(dict.fromkeys(int(source) for source in required))
    eligible_set = {int(source) for source in eligible}
    missing = [source for source in required_ids if source not in eligible_set]
    if missing:
        raise RuntimeError(f"required sources are not eligible: {missing}")
    if len(required_ids) > n:
        raise RuntimeError(f"{len(required_ids)} required sources do not fit in n={n}")
    if len(eligible_set) < n:
        raise RuntimeError(f"only {len(eligible_set)} eligible sources, need {n}")
    pool = np.array(sorted(eligible_set - set(required_ids)), dtype=int)
    take = n - len(required_ids)
    extra: list[int] = []
    if take:
        rng = dataset_rng(seed, "coralnet-source-examples")
        extra = [int(source) for source in rng.choice(pool, size=take, replace=False)]
    return sorted(required_ids + extra)


def sample_order(image_ids: Sequence[str], seed: int, source_id: int) -> list[str]:
    """Image order for one source. Independent of the appendix figure's draw."""
    ids = [str(image_id) for image_id in image_ids]
    order = dataset_rng(seed, "coralnet-source-examples", source_id).permutation(len(ids))
    return [ids[int(index)] for index in order]


def paper_splits(path: Path) -> dict[int, str]:
    """Source id to paper split, from a data config's CoralNet whitelists."""
    import yaml

    coralnet = yaml.safe_load(path.read_text())["data"]["coralnet"]
    splits: dict[int, str] = {}
    for split, block in coralnet.items():
        if not isinstance(block, dict):
            continue
        for source in block.get("whitelist_sources") or []:
            splits[int(source)] = str(split)
    return splits


_THREAD = threading.local()


def _s3_client() -> Any:
    client = getattr(_THREAD, "client", None)
    if client is None:
        client = boto3.client("s3")
        _THREAD.client = client
    return client


def ensure_cached(path: Path, bucket: str, key: str) -> Path:
    """Return ``path``, downloading ``s3://bucket/key`` when the file is absent."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    logger.info("downloading s3://%s/%s", bucket, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    boto3.client("s3").download_file(bucket, key, str(partial))
    partial.replace(path)
    return path


def load_image_index(manifest: Path) -> pd.DataFrame:
    """One row per (source, image) with the manifest image key."""
    frame = pd.read_parquet(manifest, columns=["source_id", "image_id", "image_s3_key"])
    keys = frame["image_s3_key"]
    frame = frame[keys.map(lambda key: isinstance(key, str) and bool(key))]
    frame["source_id"] = frame["source_id"].astype(int)
    frame["image_id"] = frame["image_id"].astype(str)
    return frame.drop_duplicates(["source_id", "image_id"]).reset_index(drop=True)


def load_points(
    manifest: Path, image_ids: Sequence[str], id2name: Mapping[str, str]
) -> pd.DataFrame:
    """Point rows for ``image_ids``, with lower-case CoralNet label names."""
    table = ds.dataset(str(manifest), format="parquet").to_table(
        columns=["source_id", "image_id", "row", "col", "coralnet_id"],
        filter=ds.field("image_id").isin([str(image_id) for image_id in image_ids]),
    )
    frame = table.to_pandas()
    if frame.empty:
        raise RuntimeError("manifest has no points for the sampled images")
    frame["source_id"] = frame["source_id"].astype(int)
    frame["image_id"] = frame["image_id"].astype(str)
    names = frame["coralnet_id"].map(lambda cid: id2name.get(str(int(cid))))
    missing = frame["coralnet_id"].astype(str)
    frame["source_label_name"] = names.fillna(missing).astype(str).str.lower()
    return frame


def _download_jpeg(bucket: str, key: str, dest: Path) -> bytes:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    last: Exception | None = None
    for _attempt in range(4):
        try:
            body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
            partial.write_bytes(body)
            partial.replace(dest)
            return body
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"failed to download s3://{bucket}/{key}") from last


def fetch_resized(bucket: str, key: str, dest: Path, max_edge: int) -> tuple[Image.Image, float]:
    """Download one image and clamp its longest edge. Returns ``(image, point scale)``."""
    raw = _download_jpeg(bucket, key, dest)
    with Image.open(io.BytesIO(raw)) as image:
        return downscale(as_rgb(image), max_edge)


def _legend_kwargs(fontsize: float) -> dict[str, Any]:
    return {
        "loc": "center",
        "frameon": False,
        "handlelength": 1.0,
        "handleheight": 1.0,
        "handletextpad": 0.45,
        "columnspacing": 1.3,
        "labelspacing": 0.35,
        "borderaxespad": 0.0,
        "borderpad": 0.0,
        "fontsize": fontsize,
    }


def fit_legend(handles: Sequence[Any], max_width_in: float, fontsize: float) -> tuple[int, float]:
    """Column count that fits ``max_width_in``, and the legend height in inches."""
    import matplotlib.pyplot as plt

    if not handles:
        raise ValueError("legend has no entries")
    kwargs = _legend_kwargs(fontsize)
    fig = plt.figure(figsize=(max_width_in, 1.0))
    legend = fig.legend(handles=list(handles), ncol=1, **kwargs)
    renderer = fig.canvas.get_renderer()
    widths = [text.get_window_extent(renderer).width for text in legend.get_texts()]
    fontsize_px = legend.get_texts()[0].get_fontsize() * fig.dpi / 72.0
    column_px = float(np.mean(widths)) + (1.0 + 0.45 + 1.3) * fontsize_px
    ncol = int(max(1, min(len(handles), (max_width_in * fig.dpi + 1.3 * fontsize_px) // column_px)))
    while True:
        legend.remove()
        legend = fig.legend(handles=list(handles), ncol=ncol, **kwargs)
        bbox = legend.get_window_extent(renderer)
        width_in = bbox.width / fig.dpi
        if width_in <= max_width_in + 1e-3 or ncol == 1:
            break
        ncol -= 1
    height_in = bbox.height / fig.dpi
    plt.close(fig)
    return ncol, height_in


def text_size_in(text: str, fontsize: float) -> tuple[float, float]:
    """``(width, height)`` in inches of ``text`` in the current matplotlib font."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(1.0, 1.0))
    artist = fig.text(0.0, 0.0, text, fontsize=fontsize)
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    return bbox.width / fig.dpi, bbox.height / fig.dpi


def compose_source_figure(
    images: Sequence[Any],
    labels: Sequence[str],
    colors: Mapping[str, Sequence[int]],
    source_id: int,
    path: Path,
    *,
    n_images: int = N_IMAGES,
) -> None:
    """Write one PDF: rotated source id, one row of images, legend on the right.

    The page is exactly ``FIG_WIDTH_IN`` wide. Photos are embedded at their own
    pixel size (``interpolation="none"``), which is the appendix overlay resolution.
    """
    import matplotlib.pyplot as plt

    if len(images) != n_images:
        raise ValueError(f"source {source_id}: expected {n_images} images, got {len(images)}")
    layout = row_layout(n_images)
    handles = _sparse_legend(labels, colors, wrap=CORALNET_LEGEND_WRAP)
    ncol, legend_h = fit_legend(handles, layout.legend_w_in, CORALNET_LEGEND_FONTSIZE)
    label = f"Source {source_id}"
    label_w, _label_h = text_size_in(label, SOURCE_LABEL_FONTSIZE)
    # Rotated 90 degrees, so the text's width is the vertical space it needs.
    content_h = max(layout.cell_h_in, legend_h, label_w) + 0.008
    fig_h = content_h + 2 * VERTICAL_PAD_IN
    row_bottom = VERTICAL_PAD_IN + (content_h - layout.cell_h_in) / 2

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(layout.fig_width_in, fig_h))
    fig.patch.set_facecolor("white")
    for index, image in enumerate(images):
        disp_w, disp_h = fitted_size(image.size, layout.cell_w_in, layout.cell_h_in)
        x = layout.cell_x(index) + (layout.cell_w_in - disp_w) / 2
        y = row_bottom + (layout.cell_h_in - disp_h) / 2
        axes = fig.add_axes(
            [
                x / layout.fig_width_in,
                y / fig_h,
                disp_w / layout.fig_width_in,
                disp_h / fig_h,
            ]
        )
        axes.imshow(np.asarray(image), interpolation="none")
        axes.set_axis_off()
    fig.text(
        (layout.margin_in + layout.label_col_in / 2) / layout.fig_width_in,
        (row_bottom + layout.cell_h_in / 2) / fig_h,
        label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=SOURCE_LABEL_FONTSIZE,
        color="black",
    )
    fig.legend(
        handles=handles,
        ncol=ncol,
        bbox_to_anchor=(
            layout.legend_x_in / layout.fig_width_in,
            0.0,
            layout.legend_w_in / layout.fig_width_in,
            1.0,
        ),
        bbox_transform=fig.transFigure,
        **_legend_kwargs(CORALNET_LEGEND_FONTSIZE),
    )
    fig.savefig(path, bbox_inches=None, facecolor="white")
    plt.close(fig)


def pdf_media_box(path: Path) -> tuple[float, float]:
    """Page ``(width, height)`` in PDF points."""
    match = re.search(
        rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        path.read_bytes(),
    )
    if match is None:
        raise ValueError(f"{path} has no MediaBox")
    x0, y0, x1, y1 = (float(match.group(index)) for index in range(1, 5))
    return x1 - x0, y1 - y0


def save_fullres(
    source_id: int,
    image_ids: Sequence[str],
    keys: Mapping[str, str],
    cache_dir: Path,
    out_dir: Path,
) -> Path:
    """Copy the downloaded images, in figure order, under ``s<source_id>/``."""
    dest_dir = out_dir / f"s{source_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rank, image_id in enumerate(image_ids):
        shutil.copyfile(cache_dir / keys[image_id], dest_dir / f"{rank:02d}_{image_id}.jpg")
    return dest_dir


def _overlay(
    image: Any, points: pd.DataFrame, colors: Mapping[str, Sequence[int]], scale: float
) -> Any:
    scaled = points.copy()
    scaled["row"] = scaled["row"].astype(float) * scale
    scaled["col"] = scaled["col"].astype(float) * scale
    return draw_points(image, scaled, colors, point_scale=POINT_SCALE)


def _fetch_task(task: tuple[int, str, str, Path, int]) -> tuple[int, str, Any, float]:
    source_id, image_id, key, cache_dir, max_edge = task
    dest = cache_dir / key
    image, scale = fetch_resized(BUCKET, key, dest, max_edge)
    return source_id, image_id, image, scale


def build_figures(args: argparse.Namespace) -> None:
    manifest = ensure_cached(args.manifest, BUCKET, MANIFEST_KEY)
    id2name_path = ensure_cached(args.id2name, BUCKET, ID2NAME_KEY)
    id2name = {str(key): str(value) for key, value in json.loads(id2name_path.read_text()).items()}

    logger.info("reading image index from %s", manifest)
    index = load_image_index(manifest)
    counts = index.groupby("source_id")["image_id"].nunique()
    splits = paper_splits(args.data_config)
    held_out = {
        source: split
        for source, split in paper_splits(DEFAULT_SPLIT_CONFIG).items()
        if split in {"val", "test"}
    }
    eligible = sorted(sid for sid in splits if int(counts.get(sid, 0)) >= args.n_images)
    if args.sources:
        chosen = sorted(dict.fromkeys(int(source) for source in args.sources))
        short = [sid for sid in chosen if int(counts.get(sid, 0)) < args.n_images]
        if short:
            raise RuntimeError(f"sources have fewer than {args.n_images} images: {short}")
    else:
        chosen = choose_sources(
            eligible,
            n=args.n_sources,
            required=REQUIRED_SOURCE_IDS,
            seed=args.seed,
        )
    logger.info("%d sources: %s", len(chosen), chosen)

    queues: dict[int, list[tuple[str, str]]] = {}
    for source_id in chosen:
        group = index.loc[index["source_id"] == source_id, ["image_id", "image_s3_key"]]
        keys = dict(zip(group["image_id"], group["image_s3_key"], strict=True))
        ordered = sample_order(list(keys), args.seed, source_id)
        take = min(len(ordered), args.n_images + SPARE_IMAGES)
        queues[source_id] = [(image_id, str(keys[image_id])) for image_id in ordered[:take]]

    candidate_ids = [image_id for queue in queues.values() for image_id, _key in queue]
    logger.info("reading points for %d candidate images", len(candidate_ids))
    points = load_points(manifest, candidate_ids, id2name)
    by_image = dict(iter(points.groupby("image_id", sort=False)))

    tasks = [
        (source_id, image_id, key, args.image_cache, args.max_edge)
        for source_id, queue in queues.items()
        for image_id, key in queue
        if image_id in by_image
    ]
    logger.info("fetching %d images with %d workers", len(tasks), args.workers)
    fetched: dict[tuple[int, str], tuple[Any, float]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                source_id, image_id, image, scale = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("source %s image %s failed: %s", task[0], task[1], exc)
                continue
            fetched[(source_id, image_id)] = (image, scale)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                logger.info("fetched %d/%d images", done, len(tasks))

    selected: dict[int, list[tuple[str, pd.DataFrame]]] = {}
    for source_id, queue in queues.items():
        picked: list[tuple[str, pd.DataFrame]] = []
        for image_id, _key in queue:
            hit = fetched.get((source_id, image_id))
            frame = by_image.get(image_id)
            if hit is None or frame is None or frame.empty:
                continue
            picked.append((image_id, frame))
            if len(picked) == args.n_images:
                break
        if len(picked) < args.n_images:
            raise RuntimeError(
                f"source {source_id} produced {len(picked)} of {args.n_images} images"
            )
        selected[source_id] = picked

    shown_frames = []
    for source_id, items in selected.items():
        for _image_id, frame in items:
            part = frame.copy()
            part["source_id"] = source_id
            shown_frames.append(part)
    shown = pd.concat(shown_frames, ignore_index=True)
    order = (
        shown["source_label_name"]
        .value_counts()
        .rename_axis("source_label_name")
        .reset_index(name="n")
        .sort_values(["n", "source_label_name"], ascending=[False, True])
    )
    groups = [
        set(shown.loc[shown["source_id"] == source_id, "source_label_name"].astype(str))
        for source_id in chosen
    ]
    colors = assign_colors(order["source_label_name"].astype(str).tolist(), groups=groups)
    key_of = {image_id: key for queue in queues.values() for image_id, key in queue}

    setup_style(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {len(chosen)} CoralNet sources, {args.n_images} images in one row",
        f"# width {FIG_WIDTH_IN} in, longest image edge {args.max_edge}px, "
        f"point scale {POINT_SCALE:g} (2x the appendix)",
        f"# seed {args.seed}",
        "source_id\tsplit\tn_images_in_source\timage_ids\turl",
    ]
    for source_id in chosen:
        items = selected[source_id]
        overlays = []
        used: list[str] = []
        image_ids: list[str] = []
        for image_id, frame in items:
            image, scale = fetched[(source_id, image_id)]
            overlays.append(_overlay(image, frame, colors, scale))
            used.extend(frame["source_label_name"].astype(str).tolist())
            image_ids.append(image_id)
        out = args.output_dir / f"s{source_id}.pdf"
        compose_source_figure(overlays, used, colors, source_id, out, n_images=args.n_images)
        split = held_out.get(source_id, splits.get(source_id, ""))
        if split in {"val", "test"}:
            fullres = save_fullres(source_id, image_ids, key_of, args.image_cache, args.output_dir)
            logger.info("copied %d full-res images to %s", len(image_ids), fullres.name)
        lines.append(
            f"{source_id}\t{split}\t{int(counts[source_id])}\t"
            f"{','.join(image_ids)}\thttps://coralnet.ucsd.edu/source/{source_id}/"
        )
        logger.info("wrote %s (%d labels)", out.name, len(set(used)))
    (args.output_dir / "sources.txt").write_text("\n".join(lines) + "\n")
    logger.info("done: %s", args.output_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--id2name", type=Path, default=DEFAULT_ID2NAME)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--n-sources", type=int, default=N_SOURCES)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument(
        "--sources",
        type=int,
        nargs="*",
        default=None,
        help="Use these source ids instead of the random 50 (required ids are not added).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    build_figures(_parse_args(argv))


if __name__ == "__main__":
    main()
