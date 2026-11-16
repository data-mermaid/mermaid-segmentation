#!/usr/bin/env python3
"""Contact sheets of random images from each CoralNet val and test source.

Each source gets ``coralnet_check/<split>/<source_id>.png``: ten images in two
rows of five, with that source's point labels drawn on the photos and a legend
of the labels those points use.

Val and test are the holdout whitelists in ``configs/data_config.yaml``. The
final trainval config folds validation into train, so it is not the source of
these splits.

Usage:
    python scripts/make_coralnet_check_figures.py
    python scripts/make_coralnet_check_figures.py --sources 23 57 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_appendix_data import dataset_rng  # noqa: E402
from make_appendix_figures import (  # noqa: E402
    _sparse_legend,
    assign_colors,
    setup_style,
)
from make_coralnet_source_examples import (  # noqa: E402
    BUCKET,
    DEFAULT_ID2NAME,
    DEFAULT_IMAGE_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT_CONFIG,
    POINT_SCALE,
    _download_jpeg,
    _legend_kwargs,
    ensure_cached,
    fetch_resized,
    fit_legend,
    fitted_size,
    load_image_index,
    load_points,
)

logger = logging.getLogger("coralnet_check")

_REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "coralnet_check"
ID2NAME_KEY = "coralnet-public-images/temporary/coralnet_id2name.json"
MANIFEST_KEY = (
    "etl-outputs/coralnet/20260623_nogit/coralnet_training_manifest_20260623_nogit.parquet"
)

N_COLS = 5
N_ROWS = 2
N_IMAGES = N_COLS * N_ROWS
SPARE_IMAGES = 2
FIG_WIDTH_IN = 15.0
DPI = 160
MAX_EDGE = 1024
LEGEND_WRAP = 34
SEED = 0


def holdout_splits(path: Path) -> dict[int, str]:
    """Source id to ``val`` or ``test`` from a data config's CoralNet whitelists."""
    import yaml

    coralnet = yaml.safe_load(path.read_text())["data"]["coralnet"]
    splits: dict[int, str] = {}
    for split in ("val", "test"):
        block = coralnet.get(split)
        if not isinstance(block, dict):
            continue
        for source in block.get("whitelist_sources") or []:
            splits[int(source)] = split
    return splits


def sample_ids(image_ids: Sequence[str], seed: int, source_id: int, n: int) -> list[str]:
    """Deterministic random order. The first ``n`` successful downloads are shown."""
    ids = [str(image_id) for image_id in image_ids]
    order = dataset_rng(seed, "coralnet-check", source_id).permutation(len(ids))
    return [ids[int(index)] for index in order[:n]]


def _legend_fontsize(n_labels: int) -> float:
    if n_labels <= 18:
        return 8.0
    if n_labels <= 40:
        return 6.5
    return 5.5


def compose_grid(
    images: Sequence[Any],
    labels: Sequence[str],
    colors: Mapping[str, Sequence[int]],
    source_id: int,
    path: Path,
) -> None:
    """Write one PNG: source id, two rows of five images, legend underneath."""
    import matplotlib.pyplot as plt

    if len(images) != N_IMAGES:
        raise ValueError(f"source {source_id}: expected {N_IMAGES} images, got {len(images)}")
    handles = _sparse_legend(labels, colors, wrap=LEGEND_WRAP)
    fontsize = _legend_fontsize(len(handles))
    ncol, legend_h = fit_legend(handles, FIG_WIDTH_IN - 0.6, fontsize)

    margin = 0.16
    gap_x = 0.08
    gap_y = 0.10
    title_h = 0.38
    cell_w = (FIG_WIDTH_IN - 2 * margin - (N_COLS - 1) * gap_x) / N_COLS
    cell_h = cell_w * 0.72
    block_h = N_ROWS * cell_h + (N_ROWS - 1) * gap_y
    legend_gap = 0.18
    fig_h = margin + title_h + block_h + legend_gap + legend_h + margin

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.text(
        0.5,
        (fig_h - margin - title_h / 2) / fig_h,
        f"Source {source_id}",
        ha="center",
        va="center",
        fontsize=13,
        color="black",
    )

    top = fig_h - margin - title_h
    for index, image in enumerate(images):
        row, col = divmod(index, N_COLS)
        disp_w, disp_h = fitted_size(image.size, cell_w, cell_h)
        x = margin + col * (cell_w + gap_x) + (cell_w - disp_w) / 2
        y = top - (row + 1) * cell_h - row * gap_y + (cell_h - disp_h) / 2
        axes = fig.add_axes([x / FIG_WIDTH_IN, y / fig_h, disp_w / FIG_WIDTH_IN, disp_h / fig_h])
        axes.imshow(np.asarray(image), interpolation="lanczos")
        axes.set_axis_off()

    legend_kwargs = _legend_kwargs(fontsize)
    legend_kwargs["loc"] = "upper center"
    fig.legend(
        handles=handles,
        ncol=ncol,
        bbox_to_anchor=(0.5, (margin + legend_h) / fig_h),
        bbox_transform=fig.transFigure,
        **legend_kwargs,
    )
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _overlay(
    image: Any, points: pd.DataFrame, colors: Mapping[str, Sequence[int]], scale: float
) -> Any:
    from make_appendix_figures import draw_points

    scaled = points.copy()
    scaled["row"] = scaled["row"].astype(float) * scale
    scaled["col"] = scaled["col"].astype(float) * scale
    return draw_points(image, scaled, colors, point_scale=POINT_SCALE)


def _cache_image(key: str, cache_dir: Path) -> Path:
    """Download ``key`` into the image cache if it is not already there."""
    dest = cache_dir / key
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    _download_jpeg(BUCKET, key, dest)
    return dest


def open_cached(cache_dir: Path, key: str, max_edge: int) -> tuple[Any, float]:
    """Open a cached JPEG and clamp its longest edge. ``fetch_resized`` also downloads."""
    dest = cache_dir / key
    return fetch_resized(BUCKET, key, dest, max_edge)


def build(args: argparse.Namespace) -> None:
    manifest = ensure_cached(args.manifest, BUCKET, MANIFEST_KEY)
    id2name_path = ensure_cached(args.id2name, BUCKET, ID2NAME_KEY)
    id2name = {str(key): str(value) for key, value in json.loads(id2name_path.read_text()).items()}

    splits = holdout_splits(args.split_config)
    if args.splits:
        keep = set(args.splits)
        splits = {source: split for source, split in splits.items() if split in keep}
    if args.sources:
        wanted = [int(source) for source in args.sources]
        missing = [source for source in wanted if source not in splits]
        if missing:
            raise RuntimeError(f"sources are not val/test holdouts: {missing}")
        splits = {source: splits[source] for source in wanted}

    for split in ("val", "test"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    logger.info("reading image index")
    index = load_image_index(manifest)
    index = index[index["source_id"].isin(splits)].reset_index(drop=True)
    grouped = {int(source): frame for source, frame in index.groupby("source_id", sort=False)}
    absent = sorted(set(splits) - set(grouped))
    if absent:
        raise RuntimeError(f"holdout sources missing from the manifest: {absent}")

    queues: dict[int, list[tuple[str, str]]] = {}
    short: list[int] = []
    for source_id in sorted(splits):
        frame = grouped[source_id]
        keys = dict(zip(frame["image_id"], frame["image_s3_key"], strict=True))
        if len(keys) < N_IMAGES:
            short.append(source_id)
            continue
        ordered = sample_ids(list(keys), args.seed, source_id, N_IMAGES + SPARE_IMAGES)
        queues[source_id] = [(image_id, str(keys[image_id])) for image_id in ordered]
    if short:
        raise RuntimeError(f"sources have fewer than {N_IMAGES} images: {short}")
    logger.info(
        "%d sources (%d val, %d test), %d candidate images",
        len(queues),
        sum(split == "val" for split in splits.values()),
        sum(split == "test" for split in splits.values()),
        sum(len(queue) for queue in queues.values()),
    )

    pending = [
        source_id
        for source_id in queues
        if args.force or not (args.output_dir / splits[source_id] / f"{source_id}.png").is_file()
    ]
    if not pending:
        logger.info("every figure already exists")
        return
    logger.info("%d figures to write", len(pending))

    candidate_ids = [image_id for source_id in pending for image_id, _key in queues[source_id]]
    logger.info("reading points for %d images", len(candidate_ids))
    points = load_points(manifest, candidate_ids, id2name)
    by_image: dict[tuple[int, str], pd.DataFrame] = {}
    for (source_id, image_id), frame in points.groupby(["source_id", "image_id"], sort=False):
        by_image[(int(source_id), str(image_id))] = frame

    download_jobs = [
        (source_id, image_id, key)
        for source_id in pending
        for image_id, key in queues[source_id]
        if (source_id, image_id) in by_image
    ]
    logger.info("fetching %d images with %d workers", len(download_jobs), args.workers)
    done = 0
    failed: set[tuple[int, str]] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_cache_image, key, args.image_cache): (source_id, image_id)
            for source_id, image_id, key in download_jobs
        }
        for future in as_completed(futures):
            source_id, image_id = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failed.add((source_id, image_id))
                logger.warning("source %s image %s failed: %s", source_id, image_id, exc)
            done += 1
            if done % 50 == 0 or done == len(futures):
                logger.info("fetched %d/%d images", done, len(futures))

    setup_style(False)
    failures: list[str] = []
    written = 0
    for source_id in pending:
        picked: list[tuple[str, str, pd.DataFrame]] = []
        for image_id, key in queues[source_id]:
            frame = by_image.get((source_id, image_id))
            if frame is None or frame.empty or (source_id, image_id) in failed:
                continue
            picked.append((image_id, key, frame))
            if len(picked) == N_IMAGES:
                break
        if len(picked) < N_IMAGES:
            message = f"{source_id}\t{splits[source_id]}\t{len(picked)} of {N_IMAGES}"
            failures.append(message)
            logger.error("source %s produced %s images", source_id, len(picked))
            continue

        used: list[str] = []
        frames = [frame for _image_id, _key, frame in picked]
        for frame in frames:
            used.extend(frame["source_label_name"].astype(str).tolist())
        order = (
            pd.Series(used)
            .value_counts()
            .rename_axis("source_label_name")
            .reset_index(name="n")
            .sort_values(["n", "source_label_name"], ascending=[False, True])
        )
        label_names = order["source_label_name"].astype(str).tolist()
        colors = assign_colors(label_names)
        overlays = []
        for _image_id, key, frame in picked:
            image, scale = open_cached(args.image_cache, key, args.max_edge)
            overlays.append(_overlay(image, frame, colors, scale))
        out = args.output_dir / splits[source_id] / f"{source_id}.png"
        compose_grid(overlays, used, colors, source_id, out)
        written += 1
        logger.info("wrote %s (%d labels)", out.relative_to(args.output_dir), len(label_names))

    if failures:
        report = args.output_dir / "failures.txt"
        report.write_text("\n".join(failures) + "\n")
        logger.error("%d sources failed; see %s", len(failures), report)
    logger.info("done: %d figures in %s", written, args.output_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--id2name", type=Path, default=DEFAULT_ID2NAME)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--sources", type=int, nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", choices=("val", "test"), default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-edge", type=int, default=MAX_EDGE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite figures that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    build(_parse_args(argv))


if __name__ == "__main__":
    main()
