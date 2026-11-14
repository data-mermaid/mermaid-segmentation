#!/usr/bin/env python3
"""Export per-label counts and raw sample images for the dataset appendix figures.

Runs on the cluster against the same warmed cache as ``scripts/dataset_table_stats.py``
and writes a small bundle that ``scripts/make_appendix_figures.py`` renders locally:

    <output-dir>/
      label_counts.csv                    one row per (dataset, paper split, source label)
      samples/<dataset>/NN_image.jpg      dense: + NN_label.png (native uint8 ids) + classes.json
                                          sparse: + NN_points.csv (row, col, source_label_name)
      samples/<dataset>/samples.json      which images were exported
      samples/coralnet/s<id>/NN_*         CoralNet: one folder per random source
      samples/coralnet/sources.json

Counts use the definitions of the paper table. Sparse: ``images`` is the number of
images with the label, ``annotations`` and ``pixels`` are the number of points.
Dense: ``images`` is the number of masks containing the label, ``annotations`` the
number of 8-connected components, ``pixels`` the pixel count. Ignore labels are
dropped. ``mapped`` is True when the label survives the ``class_to_concepts.csv``
filter (a row exists and the target is not ``other``).

Samples are drawn with a fixed seed from all configured splits of a dataset and
keep every source label (mapped or not). Images whose longest edge exceeds
``--max-edge`` are downscaled; sparse point coordinates are scaled to match.
Benthos samples are full square tiles only; partial edge and corner tiles of
the orthomosaic are skipped.

Usage:
    python scripts/export_appendix_data.py --workers 16
    python scripts/export_appendix_data.py --datasets moorea_labeled_corals coralscapes_v2 \\
        --n-images 2 --skip-counts --output-dir /tmp/appendix_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import zlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dataset_table_stats import (  # noqa: E402
    _MASK_BATCH,
    _SPEC_BY_KEY,
    DEFAULT_CONCEPTS_CSV,
    DEFAULT_DATA_CONFIG,
    STRUCT8,
    _batched,
    _dense_id_to_name,
    _is_ignore_name,
    _iter_dense_masks,
    as_label_array,
    label_is_mapped,
    load_mapped_names,
    paper_column,
    strip_transforms,
)

logger = logging.getLogger("export_appendix_data")

_REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "appendix_data"

COUNT_COLUMNS: tuple[str, ...] = (
    "dataset_key",
    "split",
    "source_label",
    "mapped",
    "images",
    "annotations",
    "pixels",
)

# label -> [images, annotations (components), pixels]
LabelTally = dict[str, list[int]]


# ---------------------------------------------------------------------------
# Per-label counts
# ---------------------------------------------------------------------------


def sparse_label_counts(annotations: pd.DataFrame) -> pd.DataFrame:
    """Per-label image, annotation, and pixel counts for point annotations."""
    names = annotations["source_label_name"].astype(str)
    keep = ~names.map(_is_ignore_name)
    kept = pd.DataFrame(
        {"source_label": names[keep], "image_id": annotations.loc[keep, "image_id"]}
    )
    grouped = kept.groupby("source_label", sort=True)
    points = grouped.size()
    return pd.DataFrame(
        {
            "source_label": points.index.astype(str),
            "images": grouped["image_id"].nunique().reindex(points.index).to_numpy(dtype=np.int64),
            "annotations": points.to_numpy(dtype=np.int64),
            "pixels": points.to_numpy(dtype=np.int64),
        }
    )


def dense_mask_label_counts(mask: Any, id_to_name: Mapping[int, str]) -> LabelTally:
    """Per-label ``[1, components, pixels]`` for one dense mask.

    Matches ``count_dense_mask``: id 0 and ignore names are skipped, and a positive
    id with no class name is reported as ``id_<n>``.
    """
    array = as_label_array(mask)
    tally: LabelTally = {}
    for raw_id in np.unique(array).tolist():
        class_id = int(raw_id)
        if class_id <= 0:
            continue
        name = id_to_name.get(class_id, f"id_{class_id}")
        if _is_ignore_name(name):
            continue
        binary = array == class_id
        pixels = int(binary.sum())
        if pixels == 0:
            continue
        _labeled, n_components = ndimage.label(binary, structure=STRUCT8)
        entry = tally.setdefault(name, [0, 0, 0])
        entry[0] += 1
        entry[1] += int(n_components)
        entry[2] += pixels
    return tally


def merge_tally(total: LabelTally, part: Mapping[str, list[int]]) -> None:
    for name, values in part.items():
        entry = total.setdefault(name, [0, 0, 0])
        for i, value in enumerate(values):
            entry[i] += int(value)


def _count_label_batch(payload: tuple[list[np.ndarray], dict[int, str]]) -> LabelTally:
    masks, id_to_name = payload
    total: LabelTally = {}
    for mask in masks:
        merge_tally(total, dense_mask_label_counts(mask, id_to_name))
    return total


def reduce_label_masks(
    masks: Iterable[Any],
    id_to_name: Mapping[int, str],
    *,
    workers: int,
    pool: ProcessPoolExecutor | None,
) -> tuple[LabelTally, int]:
    """Sum per-label dense counts over masks. Returns ``(tally, n_masks)``."""
    id_table = {int(class_id): str(name) for class_id, name in id_to_name.items()}
    prepared = _batched((as_label_array(mask) for mask in masks), _MASK_BATCH)
    total: LabelTally = {}
    n_masks = 0
    if workers <= 1 or pool is None:
        for batch in prepared:
            n_masks += len(batch)
            merge_tally(total, _count_label_batch((batch, id_table)))
        return total, n_masks

    pending: list[tuple[Any, int]] = []
    max_pending = max(workers, 1) * 2
    for batch in prepared:
        pending.append((pool.submit(_count_label_batch, (batch, id_table)), len(batch)))
        if len(pending) >= max_pending:
            future, n_done = pending.pop(0)
            n_masks += n_done
            merge_tally(total, future.result())
            if n_masks % 200 < n_done:
                logger.info("dense masks counted: %d", n_masks)
    for future, n_done in pending:
        n_masks += n_done
        merge_tally(total, future.result())
    return total, n_masks


def tally_frame(tally: Mapping[str, list[int]]) -> pd.DataFrame:
    names = sorted(tally)
    return pd.DataFrame(
        {
            "source_label": names,
            "images": [tally[n][0] for n in names],
            "annotations": [tally[n][1] for n in names],
            "pixels": [tally[n][2] for n in names],
        }
    )


def finalize_counts(
    frames: list[pd.DataFrame], mapped_by_source: Mapping[str, set[str]]
) -> pd.DataFrame:
    """Concatenate per-split frames, sum duplicate rows, and add the ``mapped`` flag."""
    if not frames:
        return pd.DataFrame(columns=list(COUNT_COLUMNS))
    table = (
        pd.concat(frames, ignore_index=True)
        .groupby(["dataset_key", "split", "source_label"], as_index=False, sort=True)[
            ["images", "annotations", "pixels"]
        ]
        .sum()
    )
    table["mapped"] = [
        label_is_mapped(key, label, mapped_by_source)
        for key, label in zip(table["dataset_key"], table["source_label"], strict=True)
    ]
    return table[list(COUNT_COLUMNS)]


def count_labels(
    built: Mapping[tuple[str, str], Any],
    mapped_by_source: Mapping[str, set[str]],
    *,
    workers: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        for (dataset_key, yaml_split), dataset in built.items():
            if dataset_key not in _SPEC_BY_KEY:
                continue
            column = paper_column(dataset_key, yaml_split)
            if _SPEC_BY_KEY[dataset_key].kind == "sparse":
                frame = sparse_label_counts(dataset.df_annotations)
            else:
                tally, n_masks = reduce_label_masks(
                    _iter_dense_masks(dataset_key, dataset),
                    _dense_id_to_name(dataset_key, dataset),
                    workers=workers,
                    pool=pool,
                )
                frame = tally_frame(tally)
                logger.info("%s %s: %d masks", dataset_key, yaml_split, n_masks)
            frame.insert(0, "split", column)
            frame.insert(0, "dataset_key", dataset_key)
            frames.append(frame)
            logger.info(
                "%s %s -> %s: %d labels, %d annotations, %d pixels",
                dataset_key,
                yaml_split,
                column,
                len(frame),
                int(frame["annotations"].sum()),
                int(frame["pixels"].sum()),
            )
    finally:
        if pool is not None:
            pool.shutdown()
    return finalize_counts(frames, mapped_by_source)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def dataset_rng(seed: int, *keys: Any) -> np.random.Generator:
    """Deterministic generator per ``(seed, keys...)``, stable across Python runs."""
    entropy = [int(seed)] + [zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(entropy)


def as_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.ascontiguousarray(array[..., :3]).astype(np.uint8, copy=False)


def downscale(image: np.ndarray, max_edge: int, *, nearest: bool = False) -> tuple[Any, float]:
    """Return ``(PIL image, scale)`` with the longest edge clamped to ``max_edge``."""
    from PIL import Image

    pil = Image.fromarray(image)
    w, h = pil.size
    scale = min(1.0, max_edge / max(w, h)) if max_edge > 0 else 1.0
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        pil = pil.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    return pil, scale


def sparse_points(dataset: Any, image_id: Any) -> pd.DataFrame | None:
    positions = dataset._ann_positions.get(image_id)
    if positions is None or len(positions) == 0:
        return None
    return dataset.df_annotations.iloc[positions][["row", "col", "source_label_name"]].reset_index(
        drop=True
    )


def save_sparse_sample(image: np.ndarray, points: pd.DataFrame, stem: Path, max_edge: int) -> None:
    pil, scale = downscale(as_rgb(image), max_edge)
    pil.save(stem.with_name(f"{stem.name}_image.jpg"), quality=95)
    out = points.copy()
    out["row"] = out["row"].astype(float) * scale
    out["col"] = out["col"].astype(float) * scale
    out.to_csv(stem.with_name(f"{stem.name}_points.csv"), index=False)


def save_dense_sample(image: np.ndarray, label: np.ndarray, stem: Path, max_edge: int) -> None:
    if int(label.max()) > 255:
        raise ValueError(f"label id {int(label.max())} does not fit a uint8 PNG")
    pil, _ = downscale(as_rgb(image), max_edge)
    pil.save(stem.with_name(f"{stem.name}_image.jpg"), quality=95)
    lab, _ = downscale(label.astype(np.uint8), max_edge, nearest=True)
    lab.save(stem.with_name(f"{stem.name}_label.png"))


def _read_dense(dataset_key: str, dataset: Any, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
    if dataset_key == "benthos_yuval":
        row = dataset.df_images.loc[idx]
        image = dataset.read_image(image_id=row["image_id"], site=row["site"])
        label = dataset.read_label(image_id=row["image_id"], site=row["site"])
        return np.asarray(image), as_label_array(label), f"{row['site']}/{row['image_id']}"
    row = dataset.dataset[idx]
    image = np.asarray(row["image"].convert("RGB"))
    name = str(row.get("filename", idx))
    return image, as_label_array(row["label"]), name


def _has_labels(label: np.ndarray, id_to_name: Mapping[int, str]) -> bool:
    for raw_id in np.unique(label).tolist():
        if raw_id > 0 and not _is_ignore_name(id_to_name.get(int(raw_id), "")):
            return True
    return False


def _is_same_square(image: np.ndarray, label: np.ndarray) -> bool:
    """True when ``image`` and ``label`` are the same non-empty square."""
    if image.ndim < 2 or label.ndim < 2:
        return False
    height, width = int(image.shape[0]), int(image.shape[1])
    return height == width and height > 0 and tuple(label.shape[:2]) == (height, width)


def export_dense(
    dataset_key: str,
    splits: list[tuple[str, Any]],
    out_dir: Path,
    *,
    n_images: int,
    seed: int,
    max_edge: int,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_name = _dense_id_to_name(dataset_key, splits[0][1])
    (out_dir / "classes.json").write_text(
        json.dumps({str(k): v for k, v in sorted(id_to_name.items())}, indent=2)
    )
    candidates = [(split, ds, i) for split, ds in splits for i in range(len(ds))]
    order = dataset_rng(seed, dataset_key).permutation(len(candidates))
    records: list[dict[str, Any]] = []
    for pos in order:
        if len(records) >= n_images:
            break
        split, ds, idx = candidates[pos]
        try:
            image, label, name = _read_dense(dataset_key, ds, idx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s %s idx=%d failed: %s", dataset_key, split, idx, exc)
            continue
        if dataset_key == "benthos_yuval" and not _is_same_square(image, label):
            continue
        if not _has_labels(label, id_to_name):
            continue
        stem = out_dir / f"{len(records):02d}"
        save_dense_sample(image, label, stem, max_edge)
        records.append({"file": stem.name, "split": split, "index": int(idx), "image": name})
    return records


def export_sparse(
    dataset_key: str,
    splits: list[tuple[str, Any]],
    out_dir: Path,
    *,
    n_images: int,
    seed: int,
    max_edge: int,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(split, ds, i) for split, ds in splits for i in range(len(ds))]
    order = dataset_rng(seed, dataset_key).permutation(len(candidates))
    records: list[dict[str, Any]] = []
    failures = 0
    for pos in order:
        if len(records) >= n_images or failures >= 5 * max(n_images, 1):
            break
        split, ds, idx = candidates[pos]
        row = ds.df_images.loc[idx].to_dict()
        points = sparse_points(ds, row["image_id"])
        if points is None:
            continue
        try:
            image = np.asarray(ds.read_image(**row))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.warning("%s %s image %s failed: %s", dataset_key, split, row["image_id"], exc)
            continue
        stem = out_dir / f"{len(records):02d}"
        save_sparse_sample(image, points, stem, max_edge)
        records.append(
            {
                "file": stem.name,
                "split": split,
                "image_id": str(row["image_id"]),
                "n_points": int(len(points)),
            }
        )
    return records


def export_coralnet(
    splits: list[tuple[str, Any]],
    out_dir: Path,
    *,
    n_sources: int,
    n_images: int,
    seed: int,
    max_edge: int,
) -> list[dict[str, Any]]:
    """Export ``n_images`` random images from each of ``n_sources`` random sources."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[int, list[tuple[str, Any, int]]] = {}
    for split, ds in splits:
        images = ds.df_images
        has_key = images["image_key"].map(lambda k: isinstance(k, str) and bool(k))
        for idx in np.flatnonzero(has_key.to_numpy()):
            by_source.setdefault(int(images.loc[idx, "source_id"]), []).append((split, ds, idx))
    eligible = sorted(sid for sid, items in by_source.items() if len(items) >= n_images)
    if not eligible:
        raise RuntimeError(f"no CoralNet source has at least {n_images} images")
    rng = dataset_rng(seed, "coralnet")
    chosen = sorted(
        int(s) for s in rng.choice(eligible, size=min(n_sources, len(eligible)), replace=False)
    )
    logger.info("coralnet: %d eligible sources, chose %s", len(eligible), chosen)

    sources: list[dict[str, Any]] = []
    for sid in chosen:
        items = by_source[sid]
        src_dir = out_dir / f"s{sid}"
        src_dir.mkdir(parents=True, exist_ok=True)
        order = dataset_rng(seed, "coralnet", sid).permutation(len(items))
        image_ids: list[str] = []
        failures = 0
        for pos in order:
            if len(image_ids) >= n_images or failures >= 5 * max(n_images, 1):
                break
            split, ds, idx = items[pos]
            row = ds.df_images.loc[idx].to_dict()
            points = sparse_points(ds, row["image_id"])
            if points is None:
                continue
            try:
                image = np.asarray(ds.read_image(**row))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.warning("coralnet s%d image %s failed: %s", sid, row["image_id"], exc)
                continue
            save_sparse_sample(image, points, src_dir / f"{len(image_ids):02d}", max_edge)
            image_ids.append(str(row["image_id"]))
        sources.append(
            {
                "source_id": sid,
                "split": items[0][0],
                "n_images_in_source": len(items),
                "image_ids": image_ids,
                "url": f"https://coralnet.ucsd.edu/source/{sid}/",
            }
        )
        logger.info("coralnet s%d: exported %d images", sid, len(image_ids))
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    return sources


def export_samples(built: Mapping[tuple[str, str], Any], args: argparse.Namespace) -> None:
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for (dataset_key, yaml_split), dataset in built.items():
        if dataset_key in _SPEC_BY_KEY:
            grouped.setdefault(dataset_key, []).append((yaml_split, dataset))
    root = args.output_dir / "samples"
    for dataset_key, splits in grouped.items():
        out_dir = root / dataset_key
        if dataset_key == "coralnet":
            export_coralnet(
                splits,
                out_dir,
                n_sources=args.n_coralnet_sources,
                n_images=args.n_images,
                seed=args.seed,
                max_edge=args.max_edge,
            )
            continue
        export = export_sparse if _SPEC_BY_KEY[dataset_key].kind == "sparse" else export_dense
        records = export(
            dataset_key,
            splits,
            out_dir,
            n_images=args.n_images,
            seed=args.seed,
            max_edge=args.max_edge,
        )
        (out_dir / "samples.json").write_text(json.dumps(records, indent=2))
        logger.info("%s: exported %d samples to %s", dataset_key, len(records), out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset keys to load.")
    parser.add_argument("--n-images", type=int, default=10, help="Images per dataset / source.")
    parser.add_argument("--n-coralnet-sources", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-edge", type=int, default=2048, help="Longest edge of exported images (0 = native)."
    )
    parser.add_argument("--skip-counts", action="store_true")
    parser.add_argument("--skip-samples", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    from mermaidseg.datasets import build_datasets, setup_local_cache
    from mermaidseg.io import setup_config

    cfg = setup_config({"data": str(args.data_config)})
    data_cfg = strip_transforms(cfg.data)
    setup_local_cache(data_cfg)
    built = build_datasets(data_cfg, padding=None, names=args.datasets, verbose=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_counts:
        counts = count_labels(built, load_mapped_names(args.concepts_csv), workers=args.workers)
        path = args.output_dir / "label_counts.csv"
        counts.to_csv(path, index=False)
        logger.info("wrote %s: %d rows, %d mapped", path, len(counts), int(counts["mapped"].sum()))
    if not args.skip_samples:
        export_samples(built, args)
    logger.info("done: %s", args.output_dir)


if __name__ == "__main__":
    main()
