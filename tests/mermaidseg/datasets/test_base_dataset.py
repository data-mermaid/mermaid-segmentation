"""BaseCoralDataset behaviour, plus DataLoader worker init reaching spawn-started workers."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import mermaidseg.datasets.local_cache
from mermaidseg.datasets import make_worker_init_fn
from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.local_cache import CacheStatsHandles, LocalS3Cache, create_cache_stats

BUCKET = "bucket"
KEY = "image.jpg"


def read_cached_object_in_worker(init_fn: Callable[[int], None]) -> None:
    """A worker body: run the init, then read one object the local cache already holds."""
    init_fn(0)
    LocalS3Cache.get().read_bytes(BUCKET, KEY)


@pytest.fixture
def spawn_cache_stats(monkeypatch) -> CacheStatsHandles:
    """Cache stats whose locks a spawn-started process can inherit, whatever the default."""
    # A fork-context SemLock cannot be passed to a spawn-context process.
    monkeypatch.setattr(mermaidseg.datasets.local_cache, "mp", mp.get_context("spawn"))
    return create_cache_stats()


def run_in_spawned_worker(init_fn: Callable[[int], None], cache_root: Path, monkeypatch) -> None:
    """Run ``read_cached_object_in_worker`` in a spawn-started process over ``cache_root``."""
    (cache_root / BUCKET).mkdir(parents=True)
    (cache_root / BUCKET / KEY).write_bytes(b"pixels")
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(cache_root))
    worker = mp.get_context("spawn").Process(target=read_cached_object_in_worker, args=(init_fn,))
    worker.start()
    try:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()


def test_worker_init_counts_a_spawned_workers_cache_hit_in_the_parent(
    monkeypatch, tmp_path, spawn_cache_stats
):
    init_fn = make_worker_init_fn(spawn_cache_stats)
    run_in_spawned_worker(init_fn, tmp_path / "cache", monkeypatch)
    assert spawn_cache_stats.local_hits.value == 1


@pytest.fixture
def minimal_dataset() -> BaseCoralDataset:
    """Smallest valid BaseCoralDataset — no real images needed."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1", "img2"],
            "region_id": [1, 2],
            "region_name": ["r1", "r2"],
            "source_label_name": ["Coral", "Sand"],
            "row": [10, 20],
            "col": [10, 20],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return BaseCoralDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral", "Sand"],
    )


@pytest.fixture
def single_image_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-image annotations and images DataFrames."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1"],
            "region_id": [1],
            "region_name": ["r1"],
            "source_label_name": ["Coral"],
            "row": [5],
            "col": [5],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df_annotations, df_images


class _AlwaysFailDataset(BaseCoralDataset):
    """Minimal subclass whose read_image always raises."""

    def read_image(self, **row_kwargs) -> Any:
        raise RuntimeError("simulated read failure")


class _SyntheticDataset(BaseCoralDataset):
    """Minimal subclass that returns a constant RGB image."""

    def read_image(self, **row_kwargs) -> np.ndarray:
        return np.zeros((20, 20, 3), dtype=np.uint8)


def _make_two_image_annotations(
    *,
    img0_labels: list[str | None],
    img1_labels: list[str | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[int] = []
    cols: list[int] = []
    labels: list[str | None] = []
    image_ids: list[str] = []

    for image_id, label_list in [("img0", img0_labels), ("img1", img1_labels)]:
        for row, col, label in zip([5, 10], [5, 10], label_list, strict=False):
            if label is None:
                continue
            image_ids.append(image_id)
            rows.append(row)
            cols.append(col)
            labels.append(label)

    df_annotations = pd.DataFrame(
        {
            "image_id": image_ids,
            "region_id": [1 if iid == "img0" else 2 for iid in image_ids],
            "region_name": ["r0" if iid == "img0" else "r1" for iid in image_ids],
            "source_label_name": labels,
            "row": rows,
            "col": cols,
        }
    )
    df_images = pd.DataFrame(
        {
            "image_id": ["img0", "img1"],
            "region_id": [1, 2],
            "region_name": ["r0", "r1"],
        }
    )
    return df_annotations, df_images


class _CropOutDataset(_SyntheticDataset):
    """Simulates RandomResizedCrop wiping labels on idx 0 only."""

    def _load_item(self, idx: int) -> tuple[Any, Any]:
        image, mask = super()._load_item(idx)
        if idx == 0:
            mask = np.zeros_like(mask)
        return image, mask


# --- BaseCoralDataset basic API ---


def test_base_dataset_exposes_source_label_attributes(minimal_dataset):
    assert minimal_dataset.source_id2name == {1: "Coral", 2: "Sand"}
    assert minimal_dataset.source_name2id == {"Coral": 1, "Sand": 2}
    assert minimal_dataset.num_source_classes == 3  # background + 2
    assert minimal_dataset.global_offset == 0


def test_base_dataset_set_global_offset_validates_negative(minimal_dataset):
    with pytest.raises(ValueError):
        minimal_dataset.set_global_offset(-1)


def test_base_dataset_set_global_offset_shifts_emitted_mask():
    """Folded offset in ``_load_item`` must match the previous ``np.where`` post-pass."""
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=["Coral"],
        img1_labels=["Coral"],
    )
    ds = _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
        split="val",
    )
    _, local_mask = ds[0]
    ds.set_global_offset(10)
    _, global_mask = ds[0]
    expected = np.where(local_mask > 0, local_mask + 10, local_mask)
    np.testing.assert_array_equal(global_mask, expected)


def test_base_dataset_ann_positions_match_boolean_scan(minimal_dataset):
    """Precomputed groupby indices must select the same rows (and order) as a boolean scan."""
    for image_id in minimal_dataset.df_images["image_id"]:
        positions = minimal_dataset._ann_positions[image_id]
        indexed = minimal_dataset.df_annotations.iloc[positions][
            ["row", "col", "source_label_name"]
        ]
        scanned = minimal_dataset.df_annotations.loc[
            minimal_dataset.df_annotations["image_id"] == image_id,
            ["row", "col", "source_label_name"],
        ]
        pd.testing.assert_frame_equal(
            indexed.reset_index(drop=True), scanned.reset_index(drop=True)
        )


# --- BaseCoralDataset.__getitem__ ---


def test_base_dataset_getitem_raises_and_logs_when_all_items_fail(single_image_annotations, caplog):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    # With a single-image dataset, recurse-on-failure exhausts every index and
    # raises rather than returning a ``(None, None)`` placeholder.
    with (
        caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"),
        pytest.raises(RuntimeError),
    ):
        _ = ds[0]

    assert "img1" in caplog.text
    assert "RuntimeError" in caplog.text


def test_base_dataset_records_failure_context(single_image_annotations):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
        split="train",
    )

    with pytest.raises(RuntimeError):
        _ = ds[0]
    failures = pd.DataFrame(ds._load_failures)
    assert len(failures) == 1

    row = failures.iloc[0]
    assert row["dataset_class"] == "_AlwaysFailDataset"
    assert row["split"] == "train"
    assert row["image_id"] == "img1"
    assert row["region_name"] == "r1"
    assert row["annotation_count"] == 1
    assert row["annotation_labels"] == "Coral"
    assert not row["missing_annotations"]
    assert row["error_type"] == "RuntimeError"
    assert "simulated read failure" in row["error_message"]


# --- BaseCoralDataset empty-mask skip (train split only) ---


def test_train_skips_image_with_no_annotations(monkeypatch):
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=[],
        img1_labels=["Coral"],
    )
    ds = _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        split="train",
    )
    monkeypatch.setattr(np.random, "randint", lambda low, high: 0)

    _, mask = ds[0]

    assert np.any(mask > 0)


def test_train_skips_crop_that_removed_all_labels(monkeypatch):
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=["Coral"],
        img1_labels=["Coral"],
    )
    ds = _CropOutDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
        split="train",
    )
    monkeypatch.setattr(np.random, "randint", lambda low, high: 0)

    _, mask = ds[0]

    assert np.any(mask > 0)


def test_val_returns_empty_mask_without_recursion():
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=[],
        img1_labels=["Coral"],
    )
    # Do not pass class_subset: that filter re-derives df_images from remaining
    # annotations and would drop the unlabeled image we want to probe.
    ds = _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        split="val",
    )

    _, mask = ds[0]

    assert np.all(mask == 0)


def test_train_raises_when_all_items_are_empty():
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=[],
        img1_labels=[],
    )
    ds = _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        split="train",
    )

    with pytest.raises(RuntimeError, match="empty \\(no labels\\)"):
        _ = ds[0]
