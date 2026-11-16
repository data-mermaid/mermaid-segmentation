#!/usr/bin/env python3
"""Per-source figures for a ``scripts/evaluate.py --split train`` CoralNet run.

Each source in the trainval CoralNet whitelist gets ``<output>/<split>/<source_id>.png``:

* left, a 2x2 grid of bar charts from ``coralnet_train/per_unit.csv``: accuracy
  per taxonomic rank on all, living and live-coral points (bold black line is the
  mean over ranks), and F1 for bleached, anthropogenic and all binary concepts;
* right, six images from the evaluated sample in two rows of three, with the
  source's point labels drawn on and a legend of those labels.

``<split>`` is ``val`` for the validation holdouts in ``configs/data_config.yaml``
(folded into train by the trainval config) and ``train`` otherwise.

``<output>/histograms.png`` (and ``histograms_train.png``, ``histograms_val.png``)
show the distribution of each of those metrics over sources.

Usage:
    python scripts/make_coralnet_train_eval_figures.py \\
        ~/eval_out/768_15k/RUN_NAME/model_step15000_res768_train
    python scripts/make_coralnet_train_eval_figures.py RESULTS --sources 23 69 --force
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_appendix_data import as_rgb, dataset_rng, downscale  # noqa: E402
from make_appendix_figures import _sparse_legend, assign_colors, setup_style  # noqa: E402
from make_coralnet_check_figures import holdout_splits  # noqa: E402
from make_coralnet_source_examples import (  # noqa: E402
    BUCKET,
    DEFAULT_ID2NAME,
    DEFAULT_IMAGE_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT_CONFIG,
    ID2NAME_KEY,
    MANIFEST_KEY,
    _legend_kwargs,
    _overlay,
    _s3_client,
    ensure_cached,
    fit_legend,
    fitted_size,
    load_image_index,
    load_points,
)

logger = logging.getLogger("coralnet_train_eval_figures")

_REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_TRAINVAL_CONFIG = _REPO_ROOT / "configs" / "data_config_trainval.yaml"
DEFAULT_RESIZED_CACHE = _REPO_ROOT / ".cache" / "coralnet_train_eval_figures"

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
RANK_COLORS = ("#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377")
ACCURACY_PANELS = (
    ("acc_all", "Accuracy (all points)"),
    ("acc_living", "Accuracy (living)"),
    ("acc_coral", "Accuracy (live coral)"),
)
BINARY_BARS = (
    ("bleached_f1", "Bleached", "#EE7733"),
    ("anthropogenic_f1", "Anthropogenic", "#BBBBBB"),
    ("binary_macro_f1", "All binary\n(macro)", "black"),
)

N_COLS = 3
N_ROWS = 2
N_IMAGES = N_COLS * N_ROWS
SPARE_IMAGES = 4
MAX_EDGE = 800
SEED = 0
DPI = 110
LEGEND_WRAP = 34

FIG_WIDTH_IN = 20.0
MARGIN_IN = 0.25
TITLE_H_IN = 0.55
METRICS_W_IN = 7.6
METRICS_GAP_X_IN = 0.45
METRICS_GAP_Y_IN = 0.15
AXES_PAD_LEFT_IN = 0.5
AXES_PAD_BOTTOM_IN = 0.55
AXES_PAD_TOP_IN = 0.35
METRICS_TO_IMAGES_IN = 0.55
IMAGE_GAP_IN = 0.08
LEGEND_GAP_IN = 0.2


def trainval_sources(trainval_config: Path, split_config: Path) -> dict[int, str]:
    """Trainval CoralNet train whitelist, each tagged ``val`` (paper holdout) or ``train``."""
    import yaml

    coralnet = yaml.safe_load(trainval_config.read_text())["data"]["coralnet"]
    sources = [int(source) for source in coralnet["train"]["whitelist_sources"]]
    holdout = holdout_splits(split_config)
    return {
        source: "val" if holdout.get(source) == "val" else "train" for source in sorted(sources)
    }


def sample_ids(image_ids: Sequence[str], seed: int, source_id: int, n: int) -> list[str]:
    ids = sorted(str(image_id) for image_id in image_ids)
    order = dataset_rng(seed, "coralnet-train-eval", source_id).permutation(len(ids))
    return [ids[int(index)] for index in order[:n]]


def _resized_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    image = cache_dir / key
    return image, image.with_suffix(image.suffix + ".json")


def fetch_small(key: str, raw_cache: Path, resized_cache: Path, max_edge: int) -> None:
    """Store a ``max_edge`` copy of ``key`` and its point scale under ``resized_cache``.

    Reads the full-size JPEG from ``raw_cache`` when another script already downloaded it.
    """
    from PIL import Image

    image_path, meta_path = _resized_paths(resized_cache, key)
    if image_path.is_file() and meta_path.is_file():
        return
    raw_path = raw_cache / key
    if raw_path.is_file() and raw_path.stat().st_size > 0:
        raw = raw_path.read_bytes()
    else:
        last: Exception | None = None
        for _attempt in range(4):
            try:
                raw = _s3_client().get_object(Bucket=BUCKET, Key=key)["Body"].read()
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise RuntimeError(f"failed to download s3://{BUCKET}/{key}") from last
    with Image.open(io.BytesIO(raw)) as opened:
        small, scale = downscale(as_rgb(opened), max_edge)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    partial = image_path.with_suffix(image_path.suffix + ".partial")
    small.save(partial, format="JPEG", quality=92)
    partial.replace(image_path)
    meta_path.write_text(json.dumps({"scale": scale}))


def open_small(resized_cache: Path, key: str) -> tuple[Any, float]:
    from PIL import Image

    image_path, meta_path = _resized_paths(resized_cache, key)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    return image, float(json.loads(meta_path.read_text())["scale"])


def _bar_labels(axes: Any, xs: Sequence[float], values: Sequence[float]) -> None:
    for x, value in zip(xs, values, strict=True):
        if np.isfinite(value):
            axes.text(
                x,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.6},
            )
        else:
            axes.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=8, color="0.45")


def _style_axes(axes: Any, title: str) -> None:
    axes.set_ylim(0.0, 1.1)
    axes.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes.grid(axis="y", color="0.88", linewidth=0.6)
    axes.set_axisbelow(True)
    axes.set_title(title, fontsize=11, pad=4)


def plot_rank_accuracy(axes: Any, metrics: Mapping[str, float], suffix: str, title: str) -> None:
    values = [float(metrics.get(f"tax_{rank}_{suffix}", np.nan)) for rank in RANKS]
    xs = np.arange(len(RANKS))
    heights = [value if np.isfinite(value) else 0.0 for value in values]
    axes.bar(xs, heights, color=RANK_COLORS, width=0.72)
    _bar_labels(axes, xs, values)
    finite = [value for value in values if np.isfinite(value)]
    if finite:
        mean = float(np.mean(finite))
        axes.axhline(mean, color="black", linewidth=2.6, zorder=3)
        title = f"{title}: rank mean {mean:.3f}"
    axes.set_xticks(xs)
    axes.set_xticklabels([rank.capitalize() for rank in RANKS], rotation=30, ha="right")
    _style_axes(axes, title)


def plot_binary_f1(axes: Any, metrics: Mapping[str, float]) -> None:
    values = [float(metrics.get(column, np.nan)) for column, _label, _color in BINARY_BARS]
    xs = np.arange(len(BINARY_BARS))
    heights = [value if np.isfinite(value) else 0.0 for value in values]
    axes.bar(xs, heights, color=[color for *_rest, color in BINARY_BARS], width=0.6)
    _bar_labels(axes, xs, values)
    axes.set_xticks(xs)
    axes.set_xticklabels([label for _column, label, _color in BINARY_BARS])
    for tick in axes.get_xticklabels()[-1:]:
        tick.set_fontweight("bold")
    _style_axes(axes, "Binary concept F1")


def _legend_fontsize(n_labels: int) -> float:
    if n_labels <= 24:
        return 10.0
    if n_labels <= 48:
        return 8.5
    return 7.0


def compose_figure(task: Mapping[str, Any]) -> str:
    """Render one source. Runs in a worker process; returns the written path."""
    import matplotlib.pyplot as plt

    setup_style(False)
    plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

    source_id = int(task["source_id"])
    metrics: Mapping[str, float] = task["metrics"]
    resized_cache = Path(task["resized_cache"])
    picked: list[tuple[str, pd.DataFrame]] = task["picked"]

    used: list[str] = []
    for _key, frame in picked:
        used.extend(frame["source_label_name"].astype(str).tolist())
    order = pd.Series(used, dtype=str).value_counts()
    label_names = sorted(order.index, key=lambda name: (-int(order[name]), name))
    colors = assign_colors(label_names) if label_names else {}
    overlays = []
    for key, frame in picked:
        image, scale = open_small(resized_cache, key)
        overlays.append(_overlay(image, frame, colors, scale))

    images_x = MARGIN_IN + METRICS_W_IN + METRICS_TO_IMAGES_IN
    images_w = FIG_WIDTH_IN - MARGIN_IN - images_x
    cell_w = (images_w - (N_COLS - 1) * IMAGE_GAP_IN) / N_COLS
    cell_h = cell_w * 0.72
    block_h = N_ROWS * cell_h + (N_ROWS - 1) * IMAGE_GAP_IN

    handles = _sparse_legend(used, colors, wrap=LEGEND_WRAP) if used else []
    fontsize = _legend_fontsize(len(handles))
    ncol, legend_h = fit_legend(handles, images_w, fontsize) if handles else (1, 0.0)
    body_h = block_h + (LEGEND_GAP_IN + legend_h if handles else 0.0)
    fig_h = MARGIN_IN + TITLE_H_IN + body_h + MARGIN_IN

    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")
    title = (
        f"CoralNet source {source_id} ({task['split']})   |   "
        f"{task['n_images']:,} evaluated images, {int(metrics['num_points']):,} points   |   "
        f"class accuracy {float(metrics['class_accuracy']):.3f}"
    )
    fig.text(
        0.5,
        (fig_h - MARGIN_IN - TITLE_H_IN / 2) / fig_h,
        title,
        ha="center",
        va="center",
        fontsize=14,
    )

    top = fig_h - MARGIN_IN - TITLE_H_IN
    cell_mw = (METRICS_W_IN - METRICS_GAP_X_IN) / 2
    cell_mh = (block_h - METRICS_GAP_Y_IN) / 2
    for index in range(4):
        row, col = divmod(index, 2)
        x0 = MARGIN_IN + col * (cell_mw + METRICS_GAP_X_IN)
        y0 = top - (row + 1) * cell_mh - row * METRICS_GAP_Y_IN
        axes = fig.add_axes(
            [
                (x0 + AXES_PAD_LEFT_IN) / FIG_WIDTH_IN,
                (y0 + AXES_PAD_BOTTOM_IN) / fig_h,
                (cell_mw - AXES_PAD_LEFT_IN) / FIG_WIDTH_IN,
                (cell_mh - AXES_PAD_BOTTOM_IN - AXES_PAD_TOP_IN) / fig_h,
            ]
        )
        if index < len(ACCURACY_PANELS):
            suffix, panel_title = ACCURACY_PANELS[index]
            plot_rank_accuracy(axes, metrics, suffix, panel_title)
        else:
            plot_binary_f1(axes, metrics)

    for index in range(N_IMAGES):
        row, col = divmod(index, N_COLS)
        cx = images_x + col * (cell_w + IMAGE_GAP_IN)
        cy = top - (row + 1) * cell_h - row * IMAGE_GAP_IN
        if index >= len(overlays):
            axes = fig.add_axes([cx / FIG_WIDTH_IN, cy / fig_h, cell_w / FIG_WIDTH_IN, cell_h / fig_h])
            axes.set_axis_off()
            if index == len(overlays):
                axes.text(0.5, 0.5, "no more images", ha="center", va="center", color="0.5")
            continue
        image = overlays[index]
        disp_w, disp_h = fitted_size(image.size, cell_w, cell_h)
        x = cx + (cell_w - disp_w) / 2
        y = cy + (cell_h - disp_h) / 2
        axes = fig.add_axes([x / FIG_WIDTH_IN, y / fig_h, disp_w / FIG_WIDTH_IN, disp_h / fig_h])
        axes.imshow(np.asarray(image), interpolation="lanczos")
        axes.set_axis_off()

    if handles:
        legend_kwargs = _legend_kwargs(fontsize)
        legend_kwargs["loc"] = "upper center"
        fig.legend(
            handles=handles,
            ncol=ncol,
            bbox_to_anchor=(
                (images_x + images_w / 2) / FIG_WIDTH_IN,
                (top - block_h - LEGEND_GAP_IN) / fig_h,
            ),
            bbox_transform=fig.transFigure,
            **legend_kwargs,
        )

    out = Path(task["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return str(out)


def _histogram(axes: Any, values: pd.Series, color: str, title: str, *, bold: bool) -> None:
    finite = values[np.isfinite(values)]
    axes.hist(
        finite,
        bins=np.linspace(0.0, 1.0, 21),
        color=color,
        edgecolor="white" if not bold else "black",
        linewidth=0.5,
    )
    if len(finite):
        median = float(finite.median())
        axes.axvline(median, color="0.25" if not bold else "#CC3311", linestyle="--", linewidth=1.4)
        title = f"{title}\nmedian {median:.3f}, n={len(finite)}"
    else:
        title = f"{title}\nn=0"
    axes.set_title(title, fontsize=10, fontweight="bold" if bold else "normal", pad=4)
    axes.set_xlim(0.0, 1.0)
    axes.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes.grid(axis="y", color="0.88", linewidth=0.6)
    axes.set_axisbelow(True)


def plot_histograms(per_unit: pd.DataFrame, title: str, path: Path) -> None:
    """Distribution over sources of every per-source metric in the source figures.

    NaN values (no living or coral points, no positive labels) are left out of each panel.
    """
    import matplotlib.pyplot as plt

    setup_style(False)
    plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    n_cols = len(RANKS) + 1
    n_rows = len(ACCURACY_PANELS) + 1
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), dpi=DPI, squeeze=False
    )
    fig.patch.set_facecolor("white")
    for row, (suffix, panel_title) in enumerate(ACCURACY_PANELS):
        columns = [f"tax_{rank}_{suffix}" for rank in RANKS]
        for col, (rank, color) in enumerate(zip(RANKS, RANK_COLORS, strict=True)):
            _histogram(grid[row, col], per_unit[columns[col]], color, rank.capitalize(), bold=False)
        _histogram(
            grid[row, -1],
            per_unit[columns].mean(axis=1, skipna=True),
            "black",
            "Rank mean",
            bold=True,
        )
        grid[row, 0].set_ylabel(f"{panel_title}\n# sources", fontsize=10)
    binary_row = grid[-1]
    for col, (column, label, color) in enumerate(BINARY_BARS):
        _histogram(
            binary_row[col],
            per_unit[column],
            color,
            label.replace("\n", " ") + " F1",
            bold=column == "binary_macro_f1",
        )
    for axes in binary_row[len(BINARY_BARS) :]:
        axes.set_axis_off()
    binary_row[0].set_ylabel("Binary concept F1\n# sources", fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _numeric_metrics(row: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in row.items() if key != "unit"}


def build(args: argparse.Namespace) -> None:
    results = args.results_dir.expanduser()
    per_unit = pd.read_csv(results / "coralnet_train" / "per_unit.csv")
    per_unit["unit"] = per_unit["unit"].astype(int)
    metrics_by_source = {int(row["unit"]): _numeric_metrics(row) for _, row in per_unit.iterrows()}
    sample = pd.read_parquet(results / "coralnet_train_sample.parquet")
    sample["source_id"] = sample["source_id"].astype(int)
    sample["image_id"] = sample["image_id"].astype(str)
    sampled = {int(sid): frame["image_id"].tolist() for sid, frame in sample.groupby("source_id")}

    splits = trainval_sources(args.trainval_config, args.split_config)
    if args.sources:
        wanted = [int(source) for source in args.sources]
        unknown = [source for source in wanted if source not in splits]
        if unknown:
            raise RuntimeError(f"sources are not in the trainval CoralNet whitelist: {unknown}")
        splits = {source: splits[source] for source in wanted}
    output_dir = (args.output_dir or results / "source_figures").expanduser()

    not_evaluated = sorted(source for source in splits if source not in metrics_by_source)
    if not_evaluated:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "not_evaluated.txt").write_text(
            "\n".join(f"{source}\t{splits[source]}" for source in not_evaluated) + "\n"
        )
        logger.warning(
            "%d whitelisted sources have no eval results (no images after blacklist); "
            "listed in not_evaluated.txt",
            len(not_evaluated),
        )
    sources = [source for source in splits if source in metrics_by_source]
    evaluated = per_unit[per_unit["unit"].isin(sources)]
    for split in ("all", "train", "val"):
        rows = evaluated
        if split != "all":
            rows = evaluated[evaluated["unit"].map(splits) == split]
        if rows.empty:
            continue
        name = "histograms.png" if split == "all" else f"histograms_{split}.png"
        plot_histograms(
            rows,
            f"Per-source metrics over {len(rows)} CoralNet "
            f"{'train + val' if split == 'all' else split} sources",
            output_dir / name,
        )
        logger.info("wrote %s", output_dir / name)
    if args.histograms_only:
        return
    pending = [
        source
        for source in sources
        if args.force or not (output_dir / splits[source] / f"{source}.png").is_file()
    ]
    logger.info(
        "%d evaluated sources (%d train, %d val), %d figures to write",
        len(sources),
        sum(splits[source] == "train" for source in sources),
        sum(splits[source] == "val" for source in sources),
        len(pending),
    )
    if not pending:
        return

    manifest = ensure_cached(args.manifest, BUCKET, MANIFEST_KEY)
    id2name_path = ensure_cached(args.id2name, BUCKET, ID2NAME_KEY)
    id2name = {str(key): str(value) for key, value in json.loads(id2name_path.read_text()).items()}
    index = load_image_index(manifest)
    index = index[index["source_id"].isin(pending)]
    key_of = {
        (int(sid), str(iid)): str(key)
        for sid, iid, key in index[["source_id", "image_id", "image_s3_key"]].itertuples(
            index=False
        )
    }

    queues: dict[int, list[str]] = {}
    for source_id in pending:
        candidates = [iid for iid in sampled.get(source_id, []) if (source_id, iid) in key_of]
        queues[source_id] = sample_ids(candidates, args.seed, source_id, N_IMAGES + SPARE_IMAGES)

    logger.info("reading points for %d candidate images", sum(map(len, queues.values())))
    points = load_points(
        manifest, sorted({iid for queue in queues.values() for iid in queue}), id2name
    )
    by_image = {
        (int(sid), str(iid)): frame
        for (sid, iid), frame in points.groupby(["source_id", "image_id"], sort=False)
    }

    # Fetch the first six per source, then spares only for sources that came up short.
    ready: dict[int, list[str]] = {source: [] for source in pending}
    cursor = dict.fromkeys(pending, 0)
    while True:
        jobs = []
        for source_id in pending:
            need = N_IMAGES - len(ready[source_id])
            queue = queues[source_id]
            while need > 0 and cursor[source_id] < len(queue):
                image_id = queue[cursor[source_id]]
                cursor[source_id] += 1
                if (source_id, image_id) in by_image:
                    jobs.append((source_id, image_id))
                    need -= 1
        if not jobs:
            break
        logger.info("fetching %d images with %d threads", len(jobs), args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    fetch_small,
                    key_of[job],
                    args.image_cache,
                    args.resized_cache,
                    args.max_edge,
                ): job
                for job in jobs
            }
            for done, future in enumerate(as_completed(futures), start=1):
                source_id, image_id = futures[future]
                try:
                    future.result()
                    ready[source_id].append(image_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("source %s image %s failed: %s", source_id, image_id, exc)
                if done % 200 == 0 or done == len(futures):
                    logger.info("fetched %d/%d", done, len(futures))

    tasks = []
    for source_id in pending:
        chosen = [iid for iid in queues[source_id] if iid in set(ready[source_id])][:N_IMAGES]
        if len(chosen) < N_IMAGES:
            logger.warning("source %s: only %d images to show", source_id, len(chosen))
        tasks.append(
            {
                "source_id": source_id,
                "split": splits[source_id],
                "metrics": metrics_by_source[source_id],
                "n_images": len(sampled.get(source_id, [])),
                "picked": [
                    (key_of[(source_id, iid)], by_image[(source_id, iid)]) for iid in chosen
                ],
                "resized_cache": str(args.resized_cache),
                "out": str(output_dir / splits[source_id] / f"{source_id}.png"),
            }
        )

    logger.info("rendering %d figures with %d processes", len(tasks), args.jobs)
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compose_figure, task): task["source_id"] for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{futures[future]}\t{exc}")
                logger.error("source %s failed: %s", futures[future], exc)
            if done % 50 == 0 or done == len(futures):
                logger.info("rendered %d/%d", done, len(futures))
    if failures:
        (output_dir / "failures.txt").write_text("\n".join(failures) + "\n")
    logger.info("done: %d figures in %s", len(tasks) - len(failures), output_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_dir", type=Path, help="evaluate.py --split train output dir.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Default: RESULTS_DIR/source_figures."
    )
    parser.add_argument("--trainval-config", type=Path, default=DEFAULT_TRAINVAL_CONFIG)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--id2name", type=Path, default=DEFAULT_ID2NAME)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--resized-cache", type=Path, default=DEFAULT_RESIZED_CACHE)
    parser.add_argument("--sources", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-edge", type=int, default=MAX_EDGE)
    parser.add_argument("--workers", type=int, default=24, help="Download threads.")
    parser.add_argument("--jobs", type=int, default=8, help="Rendering processes.")
    parser.add_argument("--force", action="store_true", help="Rewrite existing figures.")
    parser.add_argument(
        "--histograms-only", action="store_true", help="Only write the histogram figures."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for name in ("fontTools", "botocore", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    build(_parse_args(argv))


if __name__ == "__main__":
    main()
