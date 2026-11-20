"""Per-image source-label counts for the training splits of built datasets.

``count_train_labels`` returns one row per ``(dataset, image, source label)`` with the columns
``dataset``, ``image_id`` (str; ``<site>/<tile id>`` for ``benthos_yuval``, whose tile ids repeat across sites, and the
row position for other dense datasets), ``label`` (lowercased, stripped source label) and ``n`` (points for sparse
datasets, pixels for dense ones). Ignore labels (mask id 0, ``unlabeled``, ``unlabeled / unidentified``) are
dropped.

``scripts/build_concept_removal.py`` reads this table through ``--label-counts`` and, with a
``class_to_concepts.csv``, counts per concept the annotations and the training images in which it is
annotated. Datasets in ``DENSE_DATASETS`` are counted from their label masks; every other dataset is counted
from ``df_annotations``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

from mermaidseg.dataset_reconciliation.concept_removal import DENSE_DATASETS
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import CORALSCAPES_V2_ID2NAME

logger = logging.getLogger(__name__)

COLUMNS = ["dataset", "image_id", "label", "n"]
IGNORE_LABEL_NAMES: frozenset[str] = frozenset({"unlabeled", "unlabeled / unidentified"})


def _is_ignore_name(name: str) -> bool:
    return name.strip().lower() in IGNORE_LABEL_NAMES


def _sparse_rows(key: str, ds: Any) -> pd.DataFrame:
    ann = ds.df_annotations[["image_id", "source_label_name"]].copy()
    ann["label"] = ann.source_label_name.astype(str).str.strip().str.lower()
    ann = ann[~ann.label.map(_is_ignore_name)]
    out = ann.groupby(["image_id", "label"], observed=True).size().rename("n").reset_index()
    out.insert(0, "dataset", key)
    return out


def _dense_id_to_name(key: str, ds: Any) -> dict[int, str]:
    if key == "benthos_yuval":
        return {int(class_id): str(name) for name, class_id in ds._classes_global.items()}
    if key == "ucsd_mosaics":
        return {int(entry["id"]): str(entry["name"]) for entry in ds.class_table}
    return {int(class_id): str(name) for class_id, name in CORALSCAPES_V2_ID2NAME.items()}


def _dense_image_ids(key: str, ds: Any) -> list[str]:
    if key == "benthos_yuval":
        images = ds.df_images
        return [
            f"{site}/{image_id}"
            for site, image_id in zip(images.site, images.image_id, strict=True)
        ]
    return [str(i) for i in range(len(ds))]


def _iter_dense_masks(key: str, ds: Any) -> Iterator[Any]:
    if key == "benthos_yuval":
        images = ds.df_images
        for idx in range(len(images)):
            yield ds.read_label(image_id=images.loc[idx, "image_id"], site=images.loc[idx, "site"])
        return
    labels = ds.dataset.select_columns(["label"])
    for idx in range(len(labels)):
        yield labels[idx]["label"]


def _mask_rows(
    key: str, image_id: str, mask: Any, id2name: Mapping[int, str]
) -> list[tuple[str, str, str, int]]:
    ids, counts = np.unique(np.asarray(mask), return_counts=True)
    rows = []
    for class_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
        if class_id <= 0:
            continue
        name = id2name.get(int(class_id))
        if name is None:
            raise ValueError(f"{key}: image {image_id!r} has mask id {class_id} with no class name")
        if not _is_ignore_name(name):
            rows.append((key, image_id, name.strip().lower(), int(count)))
    return rows


def _dense_rows(key: str, ds: Any) -> pd.DataFrame:
    id2name = _dense_id_to_name(key, ds)
    image_ids = _dense_image_ids(key, ds)
    rows: list[tuple[str, str, str, int]] = []
    n_masks = 0
    for mask in _iter_dense_masks(key, ds):
        if n_masks < len(image_ids):
            rows.extend(_mask_rows(key, image_ids[n_masks], mask, id2name))
        n_masks += 1
        if n_masks % 500 == 0:
            logger.info("%s: %d masks", key, n_masks)
    if n_masks != len(image_ids):
        raise RuntimeError(f"{key}: read {n_masks} masks, expected {len(image_ids)} images")
    return pd.DataFrame(rows, columns=COLUMNS)


def count_train_labels(built: Mapping[tuple[str, str], Any]) -> pd.DataFrame:
    """Count source labels per image for every ``(dataset, split)`` in ``built``.

    Args:
        built: The mapping ``mermaidseg.datasets.build_datasets(..., splits=["train"])`` returns.
            Keys in ``DENSE_DATASETS`` are counted from their masks, all others from ``df_annotations``.

    Returns:
        One row per ``(dataset, image_id, label)`` with columns ``dataset``, ``image_id``, ``label`` and
        ``n``.

    Raises:
        ValueError: ``built`` is empty, or a positive mask id has no class name.
        RuntimeError: A dense dataset yields a different number of masks than images.
    """
    if not built:
        raise ValueError("no train splits to count")
    parts = []
    for (key, _split), ds in built.items():
        df = _dense_rows(key, ds) if key in DENSE_DATASETS else _sparse_rows(key, ds)
        df["image_id"] = df.image_id.astype(str)
        parts.append(df[COLUMNS])
    return pd.concat(parts, ignore_index=True)
