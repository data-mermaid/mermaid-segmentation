#!/usr/bin/env python3
"""Count images, annotations, and annotated pixels for the reconciled dataset table.

Reads the same cached data as the training jobs (``configs/data_config_all.yaml``
and the local S3 cache / Hugging Face cache). Crops, padding, and color
transforms are not applied, so counts are native annotations.

Two rows are written per dataset:

* ``source`` — labels as stored, before ``class_to_concepts.csv``.
* ``mapped`` — source labels that have a row in that CSV and whose
  ``target_label_class_name`` is not ``other``. Missing keys and ``other``
  targets are dropped. Class counts are surviving source labels, not collapsed
  target names. An image is counted only when it has at least one surviving
  label; images whose annotations are all unmapped or ``other`` are omitted.

Sparse annotations are point rows (one annotated pixel each). Dense annotations
are 8-connected components per class. Ignore pixels (id 0, and the Benthos /
UCSD names ``unlabeled`` and ``unlabeled / unidentified``) are not classes,
annotations, or pixels. The CoralscapesV2 class named ``background`` is kept.

Held-out yaml ``val`` splits for Pacific Labeled Corals and Benthos are reported
in the Test column. MERMAID has no val or test split in the training config.

The Slurm wrapper sets ``MERMAIDSEG_S3_OFFLINE=1`` and ``HF_HUB_OFFLINE=1`` so a
cold cache fails instead of downloading.

Usage:
    python scripts/dataset_table_stats.py
    python scripts/dataset_table_stats.py --workers 16 --output dataset_stats/dataset_table.csv
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

logger = logging.getLogger("dataset_table_stats")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = _REPO_ROOT / "configs" / "data_config_all.yaml"
DEFAULT_CONCEPTS_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "dataset_stats" / "dataset_table.csv"

PAPER_SPLITS: tuple[str, ...] = ("train", "val", "test")

# yaml split key -> paper column. Pacific and Benthos live under yaml ``val``
# because that is the held-out loader key; the table reports them as Test.
PAPER_COLUMNS: dict[str, dict[str, str]] = {
    "moorea_labeled_corals": {"train": "train"},
    "pacific_labeled_corals": {"val": "test"},
    "catlin_seaview": {"train": "train"},
    "benthos_yuval": {"val": "test"},
    "ucsd_mosaics": {"train": "train"},
    "coralscapes_v2": {"train": "train", "val": "val", "test": "test"},
    "mermaid": {"train": "train"},
    "coralnet": {"train": "train", "val": "val", "test": "test"},
}

# Pixel id 0 is unlabeled on every dense mask. These names are ignored even if
# a non-zero id carries them (Benthos ``unlabeled`` is id 0; UCSD's ignore name
# is included so a non-zero copy would still be dropped).
IGNORE_LABEL_NAMES: frozenset[str] = frozenset({"unlabeled", "unlabeled / unidentified"})

# Same 8-neighborhood as scripts/build_coralscapes_v2_probe_points.py.
STRUCT8 = np.ones((3, 3), dtype=bool)

_MASK_BATCH = 4
_RESERVED_DATA_KEYS = frozenset({"local_cache_dir", "local_cache_write_through"})

CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "dataset_key",
    "annotation_type",
    "image_type",
    "level",
    "n_classes",
    "images_train",
    "images_val",
    "images_test",
    "annotations_train",
    "annotations_val",
    "annotations_test",
    "pixels_train",
    "pixels_val",
    "pixels_test",
)

_COUNT_FIELDS: tuple[str, ...] = (
    "n_classes",
    "images_train",
    "images_val",
    "images_test",
    "annotations_train",
    "annotations_val",
    "annotations_test",
    "pixels_train",
    "pixels_val",
    "pixels_test",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    annotation_type: str
    image_type: str
    kind: str  # "sparse" or "dense"


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "moorea_labeled_corals", "Moorea Labeled Corals", "Sparse", "Photo quadrats", "sparse"
    ),
    DatasetSpec(
        "pacific_labeled_corals", "Pacific Labeled Corals", "Sparse", "Photo quadrats", "sparse"
    ),
    DatasetSpec("catlin_seaview", "Catlin Seaview", "Sparse", "Ortho-photos", "sparse"),
    DatasetSpec("benthos_yuval", "Benthos", "Dense", "Orthomosaics", "dense"),
    DatasetSpec("ucsd_mosaics", "UCSD Mosaics", "Dense", "Orthomosaics", "dense"),
    DatasetSpec("coralscapes_v2", "CoralscapesV2", "Dense", "General purpose", "dense"),
    DatasetSpec("mermaid", "MERMAID", "Sparse", "User-Uploaded", "sparse"),
    DatasetSpec("coralnet", "CoralNet Public", "Sparse", "User-Uploaded", "sparse"),
)

_SPEC_BY_KEY: dict[str, DatasetSpec] = {spec.key: spec for spec in DATASET_SPECS}


@dataclass
class DenseCounts:
    source_components: int = 0
    source_pixels: int = 0
    mapped_components: int = 0
    mapped_pixels: int = 0
    mapped_images: int = 0
    source_labels: set[str] = field(default_factory=set)
    mapped_labels: set[str] = field(default_factory=set)
    unknown_ids: set[int] = field(default_factory=set)

    def add(self, other: DenseCounts) -> None:
        self.source_components += other.source_components
        self.source_pixels += other.source_pixels
        self.mapped_components += other.mapped_components
        self.mapped_pixels += other.mapped_pixels
        self.mapped_images += other.mapped_images
        self.source_labels.update(other.source_labels)
        self.mapped_labels.update(other.mapped_labels)
        self.unknown_ids.update(other.unknown_ids)


@dataclass(frozen=True)
class SparseCounts:
    images: int
    mapped_images: int
    source_annotations: int
    source_pixels: int
    mapped_annotations: int
    mapped_pixels: int
    source_labels: set[str]
    mapped_labels: set[str]


def paper_column(dataset_key: str, yaml_split: str) -> str:
    """Return the Train/Val/Test column for one configured ``(dataset, split)``."""
    try:
        return PAPER_COLUMNS[dataset_key][yaml_split]
    except KeyError as exc:
        raise KeyError(
            f"No paper column for dataset {dataset_key!r} yaml split {yaml_split!r}"
        ) from exc


def empty_level() -> dict[str, Any]:
    """Zero image/annotation/pixel counts and an empty source-label set."""
    level: dict[str, Any] = {"labels": set()}
    for split in PAPER_SPLITS:
        level[f"images_{split}"] = 0
        level[f"annotations_{split}"] = 0
        level[f"pixels_{split}"] = 0
    return level


def add_split(
    level: dict[str, Any],
    paper_split: str,
    *,
    images: int,
    annotations: int,
    pixels: int,
    labels: set[str],
) -> None:
    """Add one split's counts into a dataset-level source or mapped record."""
    if paper_split not in PAPER_SPLITS:
        raise ValueError(f"paper split must be one of {PAPER_SPLITS}, got {paper_split!r}")
    level[f"images_{paper_split}"] += int(images)
    level[f"annotations_{paper_split}"] += int(annotations)
    level[f"pixels_{paper_split}"] += int(pixels)
    level["labels"].update(labels)


def mapped_names_by_source(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Source -> lowercased labels that have a CSV row and are not target ``other``.

    A label is dropped when it has no row, or when any row for that
    ``(source, label)`` pair has ``target_label_class_name`` equal to ``other``
    (case-insensitive). An empty target still counts as mapped.
    """
    required = {"source_dataset_source", "source_label_class_name", "target_label_class_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"class_to_concepts table is missing columns: {sorted(missing)}")

    source = frame["source_dataset_source"].astype(str).str.strip().str.lower()
    label = frame["source_label_class_name"].astype(str).str.strip().str.lower()
    is_other = (
        frame["target_label_class_name"].fillna("").astype(str).str.strip().str.lower().eq("other")
    )
    pairs = pd.DataFrame({"source": source, "label": label, "is_other": is_other})
    flagged = pairs.groupby(["source", "label"], sort=False)["is_other"].any()
    kept = flagged.index[~flagged.to_numpy()]
    mapped: dict[str, set[str]] = {}
    for src, lab in kept:
        mapped.setdefault(str(src), set()).add(str(lab))
    return mapped


def load_mapped_names(path: Path) -> dict[str, set[str]]:
    """Load mapped source labels from ``class_to_concepts.csv``."""
    return mapped_names_by_source(pd.read_csv(path))


def label_is_mapped(source: str, label: str, mapped: Mapping[str, set[str]]) -> bool:
    """True when ``label`` survives the class_to_concepts filter for ``source``."""
    return label.strip().lower() in mapped.get(source.strip().lower(), set())


def _is_ignore_name(name: str) -> bool:
    return name.strip().lower() in IGNORE_LABEL_NAMES


def count_sparse(
    annotations: pd.DataFrame,
    images: pd.DataFrame,
    mapped_names: set[str],
) -> SparseCounts:
    """Point-annotation counts. Each kept row is one annotation and one pixel.

    The source image count is ``len(images)``. The mapped image count includes
    an image only when it has at least one non-ignore point whose source label
    is in ``mapped_names``.
    """
    if "image_id" not in annotations.columns or "image_id" not in images.columns:
        raise ValueError("sparse counts require an image_id column on annotations and images")
    names = annotations["source_label_name"].astype(str)
    keep = ~names.map(_is_ignore_name)
    kept = annotations.loc[keep]
    kept_names = kept["source_label_name"].astype(str)
    lowered = kept_names.str.strip().str.lower()
    mapped_rows = kept.loc[lowered.isin(mapped_names)]
    mapped_ids = set(mapped_rows["image_id"])
    n_source = int(len(kept))
    n_mapped = int(len(mapped_rows))
    return SparseCounts(
        images=int(len(images)),
        mapped_images=int(images["image_id"].isin(mapped_ids).sum()),
        source_annotations=n_source,
        source_pixels=n_source,
        mapped_annotations=n_mapped,
        mapped_pixels=n_mapped,
        source_labels=set(kept_names.unique()),
        mapped_labels=set(mapped_rows["source_label_name"].astype(str).unique()),
    )


def as_label_array(mask: Any) -> np.ndarray:
    """Return a contiguous 2D integer label mask. A 3D mask keeps channel 0."""
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"label mask must be 2D, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.int32)
    return np.ascontiguousarray(array)


def count_dense_mask(
    mask: Any,
    id_to_name: Mapping[int, str],
    mapped_names: set[str],
) -> DenseCounts:
    """Components and pixels for one dense mask, in source and mapped views.

    Components are 8-connected and counted per class, then summed. Pixel id 0
    and ignore names are excluded. A positive id with no class name is kept in
    the source view as ``id_<n>`` and is never mapped. ``mapped_images`` is 1
    when the mask has at least one surviving label, otherwise 0.
    """
    array = as_label_array(mask)
    counts = DenseCounts()
    for raw_id in np.unique(array).tolist():
        class_id = int(raw_id)
        if class_id <= 0:
            continue
        name = id_to_name.get(class_id)
        if name is None:
            counts.unknown_ids.add(class_id)
            name = f"id_{class_id}"
        if _is_ignore_name(name):
            continue
        binary = array == class_id
        pixels = int(binary.sum())
        if pixels == 0:
            continue
        _labeled, n_components = ndimage.label(binary, structure=STRUCT8)
        n_components = int(n_components)
        counts.source_components += n_components
        counts.source_pixels += pixels
        counts.source_labels.add(name)
        if name.strip().lower() in mapped_names:
            counts.mapped_components += n_components
            counts.mapped_pixels += pixels
            counts.mapped_labels.add(name)
    counts.mapped_images = int(bool(counts.mapped_labels))
    return counts


def count_dense_mask_batch(
    masks: list[np.ndarray],
    id_to_name: Mapping[int, str],
    mapped_names: set[str],
) -> DenseCounts:
    total = DenseCounts()
    for mask in masks:
        total.add(count_dense_mask(mask, id_to_name, mapped_names))
    return total


def _dense_to_tuple(
    counts: DenseCounts,
) -> tuple[int, int, int, int, int, set[str], set[str], set[int]]:
    """Plain values so a spawned worker does not have to pickle ``DenseCounts``."""
    return (
        counts.source_components,
        counts.source_pixels,
        counts.mapped_components,
        counts.mapped_pixels,
        counts.mapped_images,
        counts.source_labels,
        counts.mapped_labels,
        counts.unknown_ids,
    )


def _dense_from_tuple(
    payload: tuple[int, int, int, int, int, set[str], set[str], set[int]],
) -> DenseCounts:
    (
        components,
        pixels,
        mapped_components,
        mapped_pixels,
        mapped_images,
        source_labels,
        mapped_labels,
        unknown,
    ) = payload
    return DenseCounts(
        source_components=components,
        source_pixels=pixels,
        mapped_components=mapped_components,
        mapped_pixels=mapped_pixels,
        mapped_images=mapped_images,
        source_labels=set(source_labels),
        mapped_labels=set(mapped_labels),
        unknown_ids=set(unknown),
    )


def _count_mask_batch(
    payload: tuple[list[np.ndarray], dict[int, str], set[str]],
) -> tuple[int, int, int, int, int, set[str], set[str], set[int]]:
    masks, id_to_name, mapped_names = payload
    return _dense_to_tuple(count_dense_mask_batch(masks, id_to_name, mapped_names))


def _batched(items: Iterable[np.ndarray], batch_size: int) -> Iterator[list[np.ndarray]]:
    batch: list[np.ndarray] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def reduce_masks(
    masks: Iterable[Any],
    id_to_name: Mapping[int, str],
    mapped_names: set[str],
    *,
    workers: int,
    pool: ProcessPoolExecutor | None,
) -> tuple[DenseCounts, int]:
    """Sum dense counts over masks. Returns ``(counts, n_masks)``."""
    id_table = {int(class_id): str(name) for class_id, name in id_to_name.items()}
    mapped = set(mapped_names)
    prepared = _batched((as_label_array(mask) for mask in masks), _MASK_BATCH)
    if workers <= 1 or pool is None:
        total = DenseCounts()
        n_masks = 0
        for batch in prepared:
            n_masks += len(batch)
            total.add(count_dense_mask_batch(batch, id_table, mapped))
        return total, n_masks

    total = DenseCounts()
    n_masks = 0
    pending: list[tuple[Any, int]] = []
    max_pending = max(workers, 1) * 2
    for batch in prepared:
        pending.append((pool.submit(_count_mask_batch, (batch, id_table, mapped)), len(batch)))
        if len(pending) >= max_pending:
            future, n_done = pending.pop(0)
            n_masks += n_done
            total.add(_dense_from_tuple(future.result()))
            if n_masks % 200 < n_done:
                logger.info("dense masks counted: %d", n_masks)
    for future, n_done in pending:
        n_masks += n_done
        total.add(_dense_from_tuple(future.result()))
    return total, n_masks


def _level_row(
    spec: DatasetSpec | None, level_name: str, level: Mapping[str, Any]
) -> dict[str, Any]:
    if spec is None:
        dataset = "Total"
        dataset_key = ""
        annotation_type = ""
        image_type = ""
        n_classes = int(level["n_classes"])
    else:
        dataset = spec.display
        dataset_key = spec.key
        annotation_type = spec.annotation_type
        image_type = spec.image_type
        n_classes = len(level["labels"])
    row: dict[str, Any] = {
        "dataset": dataset,
        "dataset_key": dataset_key,
        "annotation_type": annotation_type,
        "image_type": image_type,
        "level": level_name,
        "n_classes": n_classes,
    }
    for split in PAPER_SPLITS:
        row[f"images_{split}"] = int(level[f"images_{split}"])
        row[f"annotations_{split}"] = int(level[f"annotations_{split}"])
        row[f"pixels_{split}"] = int(level[f"pixels_{split}"])
    return row


def build_table(levels: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]) -> pd.DataFrame:
    """One source row and one mapped row per dataset, plus per-level totals.

    Datasets absent from ``levels`` are emitted as zeros. The class total is the
    sum of per-dataset class counts (label spaces are not united).
    """
    rows: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        source_level, mapped_level = levels.get(spec.key, (empty_level(), empty_level()))
        rows.append(_level_row(spec, "source", source_level))
        rows.append(_level_row(spec, "mapped", mapped_level))

    totals: list[dict[str, Any]] = []
    for level_name in ("source", "mapped"):
        chosen = [row for row in rows if row["level"] == level_name]
        total_level: dict[str, Any] = {"n_classes": sum(int(row["n_classes"]) for row in chosen)}
        for split in PAPER_SPLITS:
            for kind in ("images", "annotations", "pixels"):
                column = f"{kind}_{split}"
                total_level[column] = sum(int(row[column]) for row in chosen)
        totals.append(_level_row(None, level_name, total_level))
    rows.extend(totals)
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def apply_sparse(
    source: dict[str, Any], mapped: dict[str, Any], paper_split: str, counts: SparseCounts
) -> None:
    add_split(
        source,
        paper_split,
        images=counts.images,
        annotations=counts.source_annotations,
        pixels=counts.source_pixels,
        labels=counts.source_labels,
    )
    add_split(
        mapped,
        paper_split,
        images=counts.mapped_images,
        annotations=counts.mapped_annotations,
        pixels=counts.mapped_pixels,
        labels=counts.mapped_labels,
    )


def apply_dense(
    source: dict[str, Any],
    mapped: dict[str, Any],
    paper_split: str,
    *,
    images: int,
    counts: DenseCounts,
) -> None:
    add_split(
        source,
        paper_split,
        images=images,
        annotations=counts.source_components,
        pixels=counts.source_pixels,
        labels=counts.source_labels,
    )
    add_split(
        mapped,
        paper_split,
        images=counts.mapped_images,
        annotations=counts.mapped_components,
        pixels=counts.mapped_pixels,
        labels=counts.mapped_labels,
    )


def strip_transforms(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Drop compiled augmentations so dataset construction does not resize labels.

    Rebuilds the split dicts instead of deep-copying them, so the albumentations
    objects compiled by ``preprocess_data_config`` are never copied.
    """
    stripped: dict[str, Any] = {}
    for name, section in data_cfg.items():
        if name in _RESERVED_DATA_KEYS or not isinstance(section, Mapping):
            stripped[name] = section
            continue
        new_section: dict[str, Any] = {}
        for split_name, split_cfg in section.items():
            if not isinstance(split_cfg, Mapping):
                new_section[split_name] = split_cfg
                continue
            new_section[split_name] = {
                key: value
                for key, value in split_cfg.items()
                if key not in {"transform", "augmentation"}
            }
        stripped[name] = new_section
    return stripped


def _dense_id_to_name(dataset_key: str, dataset: Any) -> dict[int, str]:
    if dataset_key == "benthos_yuval":
        return {int(class_id): str(name) for name, class_id in dataset._classes_global.items()}
    if dataset_key == "ucsd_mosaics":
        return {int(entry["id"]): str(entry["name"]) for entry in dataset.class_table}
    if dataset_key == "coralscapes_v2":
        from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import (
            CORALSCAPES_V2_ID2NAME,
        )

        return {int(class_id): str(name) for class_id, name in CORALSCAPES_V2_ID2NAME.items()}
    raise ValueError(f"{dataset_key} is not a dense dataset")


def _iter_dense_masks(dataset_key: str, dataset: Any) -> Iterator[Any]:
    if dataset_key == "benthos_yuval":
        images = dataset.df_images
        for idx in range(len(images)):
            yield dataset.read_label(
                image_id=images.loc[idx, "image_id"],
                site=images.loc[idx, "site"],
            )
        return
    if dataset_key in {"ucsd_mosaics", "coralscapes_v2"}:
        labels = dataset.dataset.select_columns(["label"])
        for idx in range(len(labels)):
            yield labels[idx]["label"]
        return
    raise ValueError(f"{dataset_key} is not a dense dataset")


def _n_images(dataset_key: str, dataset: Any) -> int:
    if dataset_key == "benthos_yuval":
        return int(len(dataset.df_images))
    return int(len(dataset))


def count_built_datasets(
    built: Mapping[tuple[str, str], Any],
    mapped_by_source: Mapping[str, set[str]],
    *,
    workers: int,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Aggregate paper-column counts for every dataset present in ``built``."""
    levels: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        spec.key: (empty_level(), empty_level()) for spec in DATASET_SPECS
    }
    pool: ProcessPoolExecutor | None = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        for (dataset_key, yaml_split), dataset in built.items():
            if dataset_key not in _SPEC_BY_KEY:
                logger.info("Skipping %s (not in the dataset table)", dataset_key)
                continue
            column = paper_column(dataset_key, yaml_split)
            source_level, mapped_level = levels[dataset_key]
            mapped_names = mapped_by_source.get(dataset_key, set())
            spec = _SPEC_BY_KEY[dataset_key]
            if spec.kind == "sparse":
                counts = count_sparse(dataset.df_annotations, dataset.df_images, mapped_names)
                apply_sparse(source_level, mapped_level, column, counts)
                logger.info(
                    "%s %s -> %s: images=%d mapped_images=%d source_points=%d mapped_points=%d",
                    dataset_key,
                    yaml_split,
                    column,
                    counts.images,
                    counts.mapped_images,
                    counts.source_annotations,
                    counts.mapped_annotations,
                )
                continue

            expected = _n_images(dataset_key, dataset)
            dense_counts, n_masks = reduce_masks(
                _iter_dense_masks(dataset_key, dataset),
                _dense_id_to_name(dataset_key, dataset),
                mapped_names,
                workers=workers,
                pool=pool,
            )
            if n_masks != expected:
                raise RuntimeError(
                    f"{dataset_key} {yaml_split}: counted {n_masks} masks, expected {expected}"
                )
            if dense_counts.unknown_ids:
                logger.warning(
                    "%s %s: label ids with no class name: %s",
                    dataset_key,
                    yaml_split,
                    sorted(dense_counts.unknown_ids),
                )
            apply_dense(
                source_level,
                mapped_level,
                column,
                images=expected,
                counts=dense_counts,
            )
            logger.info(
                "%s %s -> %s: images=%d mapped_images=%d source_components=%d "
                "source_pixels=%d mapped_components=%d mapped_pixels=%d",
                dataset_key,
                yaml_split,
                column,
                expected,
                dense_counts.mapped_images,
                dense_counts.source_components,
                dense_counts.source_pixels,
                dense_counts.mapped_components,
                dense_counts.mapped_pixels,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    return levels


def format_level(table: pd.DataFrame, level: str) -> str:
    view = table.loc[table["level"] == level, list(CSV_COLUMNS)].drop(
        columns=["dataset_key", "level"]
    )
    return view.to_string(index=False)


def print_tables(table: pd.DataFrame) -> None:
    print("\nSource labels (before class_to_concepts mapping)")
    print(format_level(table, "source"))
    print(
        "\nMapped labels (class_to_concepts row whose target is not 'other'; "
        "images with no surviving label omitted)"
    )
    print(format_level(table, "mapped"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Process-pool size for dense connected components (1 = in process).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset keys to load. The CSV still lists every table row.",
    )
    parser.add_argument(
        "--check-pool",
        action="store_true",
        help="Count one synthetic mask in a spawn pool and exit. No data access.",
    )
    return parser.parse_args(argv)


def check_pool() -> None:
    """Fail fast if spawned workers cannot return dense counts."""
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        counts, n_masks = reduce_masks(
            [mask],
            {1: "coral"},
            {"coral"},
            workers=2,
            pool=pool,
        )
    if n_masks != 1 or counts.source_components != 1 or counts.source_pixels != 2:
        raise SystemExit(
            f"pool check failed: n_masks={n_masks} components={counts.source_components}"
        )
    if counts.mapped_pixels != 2 or counts.mapped_images != 1:
        raise SystemExit(
            f"pool check failed: mapped_pixels={counts.mapped_pixels} "
            f"mapped_images={counts.mapped_images}"
        )
    print("pool ok")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.check_pool:
        check_pool()
        return
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    from mermaidseg.datasets import build_datasets, setup_local_cache
    from mermaidseg.io import setup_config

    cfg = setup_config({"data": str(args.data_config)})
    data_cfg = strip_transforms(cfg.data)
    setup_local_cache(data_cfg)
    mapped_by_source = load_mapped_names(args.concepts_csv)
    for spec in DATASET_SPECS:
        logger.info(
            "%s: %d mapped source labels in %s",
            spec.key,
            len(mapped_by_source.get(spec.key, ())),
            args.concepts_csv,
        )
    built = build_datasets(
        data_cfg,
        padding=None,
        names=args.datasets,
        verbose=False,
    )
    levels = count_built_datasets(built, mapped_by_source, workers=args.workers)
    table = build_table(levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print_tables(table)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
