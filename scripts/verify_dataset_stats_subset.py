#!/usr/bin/env python3
"""Smoke check for dataset stats on a small MERMAID slice (or in-memory stubs).

Reads MERMAID confirmed annotations from Parquet (local path or ``s3://``),
keeps the first ``--max-images`` unique images (after an optional shuffle
seed), builds a :class:`SourceLabelRegistry` with identity MERMAID→target
mapping, runs the same helpers as training-time stats logging, and prints
compact summaries (no MLflow).

From the repository root::

    # Default Parquet URI matches :class:`~mermaidseg.datasets.mermaid.mermaid_dataset.MermaidDataset`
    # (``annotations_path`` on S3; needs AWS credentials + network)
    uv run python scripts/verify_dataset_stats_subset.py --max-images 40

    # Override only when you use a different Parquet than the class default
    uv run python scripts/verify_dataset_stats_subset.py \\
        --parquet s3://other-bucket/path/to/annotations.parquet --max-images 25

    # Offline / CI-style check (synthetic stub)
    uv run python scripts/verify_dataset_stats_subset.py --stub

Environment:

    ``MERMAID_ANNOTATIONS_PARQUET`` overrides the default ``annotations_path``
    on :class:`~mermaidseg.datasets.mermaid.mermaid_dataset.MermaidDataset` when
    ``--parquet`` is omitted.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from torch.utils.data import Subset

from mermaidseg.dataset_reconciliation.dataset_stats import (
    compute_class_by_source,
    compute_class_counts,
    compute_source_stats,
    compute_train_summary,
    resolve_split_annotations,
)
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.mermaid.mermaid_dataset import MermaidDataset
from tests._dataset_stubs import make_mermaid_stub, make_registry_stub

_ap = inspect.signature(MermaidDataset.__init__).parameters["annotations_path"]
if _ap.default is inspect.Parameter.empty:
    raise RuntimeError("MermaidDataset.annotations_path must have a default for this script")
DEFAULT_MERMAID_PARQUET: str = str(_ap.default)


class MermaidParquetSubsetDataset(BaseCoralDataset):
    """MERMAID rows already in ``BaseCoralDataset`` schema; stats-only (no image I/O)."""

    SOURCE_NAME = "mermaid"

    def read_image(self, **kwargs) -> NDArray[np.uint8]:
        raise NotImplementedError("verify_dataset_stats_subset only uses annotation frames")


def _normalize_mermaid_annotation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_label_name" not in out.columns:
        if "benthic_attribute_name" in out.columns:
            out = out.rename(columns={"benthic_attribute_name": "source_label_name"})
        else:
            raise ValueError(
                "Expected 'source_label_name' or 'benthic_attribute_name' in MERMAID Parquet"
            )
    required = {"image_id", "row", "col", "source_label_name", "region_id", "region_name"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"MERMAID Parquet missing columns: {sorted(missing)}")
    return out


def _subset_by_max_images(
    df_annotations: pd.DataFrame, max_images: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_ids = df_annotations["image_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    keep = set(unique_ids[: max(1, min(max_images, len(unique_ids)))])
    ann = df_annotations[df_annotations["image_id"].isin(keep)].reset_index(drop=True)
    img = (
        ann[["image_id", "region_id", "region_name"]]
        .drop_duplicates(subset=["image_id"])
        .reset_index(drop=True)
    )
    return ann, img


def _split_indices(n: int, seed: int) -> dict[str, list[int]]:
    if n <= 0:
        return {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(n)
    rng.shuffle(idx)
    if n == 1:
        return {"train": [int(idx[0])], "val": [int(idx[0])], "test": [int(idx[0])]}
    if n == 2:
        return {
            "train": [int(idx[0])],
            "val": [int(idx[1])],
            "test": [int(idx[1])],
        }
    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    train = idx[:n_train].tolist()
    val = idx[n_train : n_train + n_val].tolist()
    test = idx[n_train + n_val :].tolist()
    if not test:
        test = [int(idx[-1])]
    return {"train": train, "val": val, "test": test}


def _run_resolved(resolved: dict, registry, expected_images: int | None) -> int:
    class_counts = compute_class_counts(resolved, registry)
    source_stats = compute_source_stats(resolved)
    class_by_source = compute_class_by_source(resolved, registry)
    train_summary = compute_train_summary(resolved, registry)

    print("class_counts:", class_counts.shape[0], "rows,", class_counts.shape[1], "cols")
    print("source_stats:", source_stats.shape[0], "rows,", source_stats.shape[1], "cols")
    print("class_by_source:", class_by_source.shape[0], "rows")
    print("train_summary (YAML):")
    print(yaml.safe_dump(train_summary, sort_keys=False).rstrip())

    if class_counts.empty:
        print("class_counts is empty", file=sys.stderr)
        return 1
    if source_stats.empty:
        print("source_stats is empty", file=sys.stderr)
        return 1
    if expected_images is not None and train_summary.get("total_images") != expected_images:
        print(
            f"expected total_images={expected_images}, got {train_summary.get('total_images')}",
            file=sys.stderr,
        )
        return 1
    return 0


def run_stub() -> int:
    stub = make_mermaid_stub(
        image_to_classes={
            "img-1": ["Acropora"],
            "img-2": ["Porites"],
            "img-3": ["Acropora", "Porites"],
        },
        image_to_region={"img-1": "A", "img-2": "B", "img-3": "B"},
        source_id2name={1: "Acropora", 2: "Porites"},
        global_offset=0,
    )
    registry = make_registry_stub(
        target_id2label={1: "Acropora", 2: "Porites"},
        source_to_target_pairs=[(1, 1), (2, 2)],
    )
    splits = {
        "train": Subset(stub, [0]),
        "val": Subset(stub, [1]),
        "test": Subset(stub, [2]),
    }
    resolved: dict = {}
    for name, ds in splits.items():
        r = resolve_split_annotations(ds, registry)
        if r is None:
            print(f"resolve_split_annotations failed for split={name!r}", file=sys.stderr)
            return 1
        resolved[name] = r
    return _run_resolved(resolved, registry, expected_images=3)


def run_mermaid_parquet(parquet_path: str, max_images: int, seed: int) -> int:
    print(f"Loading MERMAID annotations from {parquet_path!r} …", file=sys.stderr)
    df_raw = pd.read_parquet(parquet_path)
    df_ann = _normalize_mermaid_annotation_df(df_raw)
    df_ann, df_img = _subset_by_max_images(df_ann, max_images=max_images, seed=seed)
    if df_ann.empty or df_img.empty:
        print("Subset is empty after filtering by image_id", file=sys.stderr)
        return 1

    dataset = MermaidParquetSubsetDataset(df_ann, df_img)
    registry = SourceLabelRegistry([dataset], fetch_remote=False)

    split_map = _split_indices(len(dataset), seed=seed)
    splits = {name: Subset(dataset, indices) for name, indices in split_map.items()}

    resolved: dict = {}
    for name, ds in splits.items():
        r = resolve_split_annotations(ds, registry)
        if r is None:
            print(f"resolve_split_annotations failed for split={name!r}", file=sys.stderr)
            return 1
        resolved[name] = r

    n_img = int(len(dataset))
    return _run_resolved(resolved, registry, expected_images=n_img)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use in-memory stubs (no Parquet / S3).",
    )
    p.add_argument(
        "--parquet",
        default=os.environ.get("MERMAID_ANNOTATIONS_PARQUET"),
        help=(
            "Parquet path (default: env MERMAID_ANNOTATIONS_PARQUET, else same as "
            f"MermaidDataset.annotations_path: {DEFAULT_MERMAID_PARQUET!r})"
        ),
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=30,
        metavar="N",
        help="Maximum number of distinct image_id values to keep (default: 30).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for image subsampling and split shuffling.",
    )
    args = p.parse_args(argv)

    if args.stub:
        return run_stub()

    path = args.parquet or DEFAULT_MERMAID_PARQUET
    return run_mermaid_parquet(path, max_images=args.max_images, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
