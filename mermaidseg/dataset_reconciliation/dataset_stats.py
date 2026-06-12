"""Pure helpers that compute per-run dataset statistics in target space.

These helpers operate on already-resolved annotation frames and a ``SourceLabelRegistry``. They
never touch MLflow or perform any I/O — the ``Logger.log_dataset_statistics`` orchestrator calls
them and uploads the resulting frames/dicts.

Class identity ("target" space) comes from the registry; per-source rows preserve dataset-level keys
(``region_*`` for mermaid, ``source_id`` for coralnet).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

# class_kind values (kept in sync with the spec).
KIND_BACKGROUND = "background"
KIND_TARGET = "target"
KIND_UNCLASSIFIED = "unclassified"
_UNCLASSIFIED_NAMES = {"other", "unknown", "unclassified"}


def classify_kind(target_id: int, target_name: str) -> str:
    if target_id == 0:
        return KIND_BACKGROUND
    if target_name.strip().lower() in _UNCLASSIFIED_NAMES:
        return KIND_UNCLASSIFIED
    return KIND_TARGET


def _source_name_to_target(dataset: Any, registry: Any) -> dict[str, int]:
    """Build ``source_label_name -> target_id`` for one dataset.

    ``target_id`` is 0 if the registry maps the corresponding global source id to background.
    """
    s2t = registry.source_to_target
    mapping: dict[str, int] = {}
    for local_id, name in dataset.source_id2name.items():
        global_id = int(local_id) + int(dataset.global_offset)
        if 0 <= global_id < int(s2t.shape[0]):
            mapping[name] = int(s2t[global_id].item())
        else:
            mapping[name] = 0
    return mapping


def resolve_split_annotations(split: Any, registry: Any):
    """Resolve a split into ``(df_annotations, df_images, source_name_to_target)``.

    See module docstring for shape handling. Never raises.
    """
    if isinstance(split, Subset):
        parent = split.dataset
        resolved = resolve_split_annotations(parent, registry)
        if resolved is None:
            return None
        parent_ann, parent_img, mapping = resolved
        df_images = parent_img.iloc[list(split.indices)].reset_index(drop=True)
        df_ann = parent_ann[parent_ann["image_id"].isin(df_images["image_id"])].copy()
        return df_ann, df_images, mapping

    children = getattr(split, "_datasets", None) or getattr(split, "datasets", None)
    if children is not None:
        resolved_children = [resolve_split_annotations(c, registry) for c in children]
        resolved_children = [r for r in resolved_children if r is not None]
        if not resolved_children:
            return None
        ann = pd.concat([r[0] for r in resolved_children], ignore_index=True)
        img = pd.concat([r[1] for r in resolved_children], ignore_index=True)
        merged: dict[str, int] = {}
        for _, _, mapping in resolved_children:
            merged.update(mapping)
        return ann, img, merged

    if (
        hasattr(split, "df_annotations")
        and hasattr(split, "df_images")
        and hasattr(split, "source_id2name")
        and hasattr(split, "global_offset")
    ):
        source_map = _source_name_to_target(split, registry)
        df_ann = split.df_annotations.copy()
        df_ann["target_id"] = df_ann["source_label_name"].map(source_map).fillna(0).astype(int)
        return df_ann, split.df_images.copy(), source_map

    logger.warning("Unsupported split shape: %s — skipping", type(split).__name__)
    return None


def compute_class_counts(resolved_splits: dict, registry: Any) -> pd.DataFrame:
    """Target-space per-class × split counts and fractions.

    ``resolved_splits`` maps split-name to the tuple returned by ``resolve_split_annotations``. The
    function reads ``target_id`` from each resolved annotation frame — no source-name lookups happen
    here.
    """
    all_classes: list[tuple[int, str]] = [(0, "background")]
    all_classes.extend(sorted(registry.target_id2label.items()))
    split_names = list(resolved_splits.keys())

    result = pd.DataFrame(
        [
            {
                "target_id": tid,
                "target_name": tname,
                "class_kind": classify_kind(tid, tname),
            }
            for tid, tname in all_classes
        ]
    )

    target_ids = result["target_id"].tolist()
    for split_name in split_names:
        df_ann, _df_img, _ = resolved_splits[split_name]
        per_class = (
            df_ann.groupby("target_id")
            .agg(annotations=("image_id", "size"), images=("image_id", "nunique"))
            .reindex(target_ids, fill_value=0)
        )
        result[f"{split_name}_annotations"] = per_class["annotations"].astype(int).to_numpy()
        result[f"{split_name}_images"] = per_class["images"].astype(int).to_numpy()

    for split_name in split_names:
        col = f"{split_name}_annotations"
        total = result[col].sum()
        result[f"{split_name}_fraction"] = result[col] / total if total else 0.0

    ordered = ["target_id", "target_name", "class_kind"]
    ordered += [f"{s}_annotations" for s in split_names]
    ordered += [f"{s}_images" for s in split_names]
    ordered += [f"{s}_fraction" for s in split_names]
    return result[ordered]


def _source_columns(df_annotations: pd.DataFrame) -> tuple[str, str] | None:
    """Return ``(source_type, key_column)`` based on which dataset columns are present.

    Returns ``None`` if neither ``region_id`` nor ``source_id`` is in the frame.
    """
    if "region_id" in df_annotations.columns:
        return "region", "region_name"
    if "source_id" in df_annotations.columns:
        return "source", "source_id"
    return None


def compute_class_by_source(resolved_splits: dict, registry: Any) -> pd.DataFrame:
    """Long-format source × class × split frame in target space.

    Zero-count rows are omitted. Background (``target_id == 0``) is excluded — annotation rows never
    carry the background class. Rows whose ``target_id`` is 0 after registry mapping (e.g. unmapped
    source labels) are also dropped.
    """
    target_id2label = registry.target_id2label
    cols = [
        "source_key",
        "source_type",
        "target_id",
        "target_name",
        "split",
        "annotations",
        "images",
    ]

    frames: list[pd.DataFrame] = []
    for split_name, (df_ann, _df_img, _) in resolved_splits.items():
        src_cols = _source_columns(df_ann)
        if src_cols is None or df_ann.empty:
            continue
        source_type, key_col = src_cols
        grouped = (
            df_ann.assign(source_key=df_ann[key_col].astype(str))
            .groupby(["source_key", "target_id"])
            .agg(annotations=("image_id", "size"), images=("image_id", "nunique"))
            .reset_index()
        )
        grouped = grouped[grouped["target_id"] != 0]
        grouped = grouped[grouped["target_id"].isin(target_id2label)]
        if grouped.empty:
            continue
        grouped["source_type"] = source_type
        grouped["split"] = split_name
        grouped["target_name"] = grouped["target_id"].map(target_id2label)
        frames.append(grouped[cols])

    if not frames:
        return pd.DataFrame(columns=cols)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["source_type", "source_key", "target_id", "split"])
        .reset_index(drop=True)
    )


def compute_train_summary(resolved_splits: dict, registry: Any) -> dict:
    """Top-level summary dict serialized to ``train_summary.yaml``.

    Class-balance metrics (``top1/3/5_share``, ``effective_num_classes``) are computed over the
    training split only, eligible classes only (i.e. excluding background and unclassified).
    """
    target_id2label = registry.target_id2label
    eligible_ids = [
        tid for tid, name in target_id2label.items() if classify_kind(tid, name) == KIND_TARGET
    ]

    summary: dict = {
        "total_images": 0,
        "total_annotations": 0,
        "splits": {},
        "num_target_classes": int(registry.num_target_classes),
        "eligible_num_classes": len(eligible_ids),
    }
    annotations_per_image: dict[str, dict] = {}

    for split_name, (df_ann, df_img, _) in resolved_splits.items():
        n_images = int(len(df_img))
        n_annotations = int(len(df_ann))
        summary["splits"][split_name] = {"images": n_images, "annotations": n_annotations}
        summary["total_images"] += n_images
        summary["total_annotations"] += n_annotations

        if n_images:
            counts = (
                df_ann.groupby("image_id")
                .size()
                .reindex(df_img["image_id"], fill_value=0)
                .astype(int)
            )
            annotations_per_image[split_name] = {
                "mean": float(counts.mean()),
                "median": float(counts.median()),
                "p10": float(counts.quantile(0.10)),
                "p90": float(counts.quantile(0.90)),
                "min": int(counts.min()),
                "max": int(counts.max()),
            }
        else:
            annotations_per_image[split_name] = {
                "mean": 0.0,
                "median": 0.0,
                "p10": 0.0,
                "p90": 0.0,
                "min": 0,
                "max": 0,
            }

    summary["annotations_per_image"] = annotations_per_image

    train = resolved_splits.get("train")
    if train is not None:
        df_ann, _df_img, _ = train
        counts = (
            df_ann[df_ann["target_id"].isin(eligible_ids)]["target_id"]
            .value_counts()
            .reindex(eligible_ids, fill_value=0)
            .sort_values(ascending=False)
        )
        total = int(counts.sum())

        if total == 0:
            summary["top1_share"] = 0.0
            summary["top3_share"] = 0.0
            summary["top5_share"] = 0.0
            summary["effective_num_classes"] = 0.0
        else:
            summary["top1_share"] = float(counts.head(1).sum() / total)
            summary["top3_share"] = float(counts.head(3).sum() / total)
            summary["top5_share"] = float(counts.head(5).sum() / total)
            probs = (counts / total).to_numpy()
            probs = probs[probs > 0]
            entropy = float(-(probs * np.log(probs)).sum())
            summary["effective_num_classes"] = float(math.exp(entropy))

    return summary


def compute_source_stats(resolved_splits: dict) -> pd.DataFrame:
    """Per-region/source x split image and annotation counts.

    Mermaid (``region_*``) and CoralNet (``source_id``) rows coexist in the same frame,
    distinguished by ``source_type``. ``source_key`` is always a string.
    """
    split_names = list(resolved_splits.keys())
    rows_by_key: dict[tuple[str, str], dict] = {}

    for split_name in split_names:
        df_ann, df_img, _ = resolved_splits[split_name]
        cols = _source_columns(df_ann)
        if cols is None:
            continue
        source_type, key_col = cols
        img_per_source = df_img[key_col].astype(str).value_counts().to_dict()
        ann_per_source = df_ann[key_col].astype(str).value_counts().to_dict()
        for source_key in set(img_per_source) | set(ann_per_source):
            row = rows_by_key.setdefault(
                (source_type, source_key),
                {"source_key": source_key, "source_type": source_type},
            )
            row[f"{split_name}_images"] = img_per_source.get(source_key, 0)
            row[f"{split_name}_annotations"] = ann_per_source.get(source_key, 0)

    for row in rows_by_key.values():
        for split_name in split_names:
            row.setdefault(f"{split_name}_images", 0)
            row.setdefault(f"{split_name}_annotations", 0)

    cols_order = ["source_key", "source_type"]
    cols_order += [f"{s}_images" for s in split_names]
    cols_order += [f"{s}_annotations" for s in split_names]

    if not rows_by_key:
        return pd.DataFrame(columns=cols_order)
    return (
        pd.DataFrame(list(rows_by_key.values()))[cols_order]
        .sort_values(["source_type", "source_key"])
        .reset_index(drop=True)
    )
