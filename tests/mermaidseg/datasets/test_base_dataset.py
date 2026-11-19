"""BaseCoralDataset behaviour, plus DataLoader worker init reaching spawn-started workers."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets import make_worker_init_fn
from mermaidseg.datasets.base_dataset import BaseCoralDataset
from tests.mermaidseg.conftest import run_in_spawned_worker


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


def test_base_dataset_set_global_offset_accepts_zero_to_reset(minimal_dataset):
    minimal_dataset.set_global_offset(5)
    assert minimal_dataset.global_offset == 5

    minimal_dataset.set_global_offset(0)

    assert minimal_dataset.global_offset == 0


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


# --- BaseCoralDataset.__getitem__ ---


def test_base_dataset_all_items_failing_raises_logs_and_records_context(
    single_image_annotations, caplog
):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
        split="train",
    )

    # With a single-image dataset, recurse-on-failure exhausts every index and
    # raises rather than returning a ``(None, None)`` placeholder.
    with (
        caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"),
        pytest.raises(RuntimeError, match="all 1 items failed"),
    ):
        _ = ds[0]

    assert "img1" in caplog.text
    assert "RuntimeError" in caplog.text

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


# --- construction: source-label space, class_subset, image_keep ---


def _annotations(labels: list[str], **extra: Any) -> pd.DataFrame:
    """One point per label, each on its own image ``img<i>``."""
    n = len(labels)
    return pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(n)],
            "source_label_name": labels,
            "row": [5] * n,
            "col": [5] * n,
            **extra,
        }
    )


def test_source_ids_follow_label_frequency_not_alphabet_or_first_appearance():
    df_annotations = _annotations(["Rubble", "Sand", "Sand", "Coral", "Sand", "Rubble"])
    ds = _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_annotations[["image_id"]].drop_duplicates().reset_index(drop=True),
    )

    assert ds.source_id2name == {1: "Sand", 2: "Rubble", 3: "Coral"}
    assert ds.source_name2id == {"Sand": 1, "Rubble": 2, "Coral": 3}
    assert ds.num_source_classes == 4


def test_class_subset_drops_other_classes_and_their_images():
    df_annotations = _annotations(["Coral", "Sand", "Coral"], region_id=[1, 2, 3])
    df_annotations["region_name"] = ["r1", "r2", "r3"]
    df_images = df_annotations[["image_id", "region_id", "region_name"]]

    ds = _SyntheticDataset(
        df_annotations=df_annotations, df_images=df_images, class_subset=["Coral"]
    )

    assert ds.df_annotations["source_label_name"].tolist() == ["Coral", "Coral"]
    assert ds.df_images["image_id"].tolist() == ["img0", "img2"]
    assert len(ds) == 2
    assert ds.source_id2name == {1: "Coral"}
    assert ds.num_source_classes == 2


def test_class_subset_with_source_id_columns_rederives_images_by_source_and_image():
    df_annotations = _annotations(["Coral", "Sand", "Coral"], source_id=[10, 10, 20])
    df_images = df_annotations[["source_id", "image_id"]]

    ds = _SyntheticDataset(
        df_annotations=df_annotations, df_images=df_images, class_subset=["Sand"]
    )

    assert ds.df_images.to_dict("list") == {"source_id": [10], "image_id": ["img1"]}


def test_class_subset_without_a_known_images_schema_raises():
    df_annotations = _annotations(["Coral", "Sand"])

    with pytest.raises(ValueError, match="cannot auto-detect"):
        _SyntheticDataset(
            df_annotations=df_annotations,
            df_images=df_annotations[["image_id"]],
            class_subset=["Coral"],
        )


def test_missing_source_label_name_column_raises():
    df_annotations = _annotations(["Coral"]).rename(columns={"source_label_name": "label"})

    with pytest.raises(ValueError, match="source_label_name"):
        _SyntheticDataset(df_annotations=df_annotations, df_images=df_annotations[["image_id"]])


class _NamedDataset(_SyntheticDataset):
    SOURCE_NAME = "named"


def test_image_keep_restricts_annotations_and_images_to_the_allowlist(tmp_path):
    df_annotations = _annotations(["Coral", "Sand", "Coral"])
    keep = tmp_path / "keep.parquet"
    pd.DataFrame({"dataset": ["named", "named"], "image_id": ["img0", "img2"]}).to_parquet(keep)

    ds = _NamedDataset(
        df_annotations=df_annotations,
        df_images=df_annotations[["image_id"]],
        image_keep=keep,
    )

    assert ds.df_images["image_id"].tolist() == ["img0", "img2"]
    assert ds.df_annotations["image_id"].tolist() == ["img0", "img2"]
    assert ds.source_id2name == {1: "Coral"}  # Sand was only on the dropped image


def test_image_keep_for_another_dataset_raises(tmp_path):
    df_annotations = _annotations(["Coral"])
    keep = tmp_path / "keep.parquet"
    pd.DataFrame({"dataset": ["other"], "image_id": ["img0"]}).to_parquet(keep)

    with pytest.raises(ValueError, match="expected named"):
        _NamedDataset(
            df_annotations=df_annotations,
            df_images=df_annotations[["image_id"]],
            image_keep=keep,
        )


# --- empty dataset and source rows ---


def test_empty_dataset_raises_on_access():
    df_annotations = _annotations([])
    ds = _SyntheticDataset(df_annotations=df_annotations, df_images=df_annotations[["image_id"]])

    assert len(ds) == 0
    with pytest.raises(RuntimeError, match="dataset is empty"):
        _ = ds[0]


def test_set_source_rows_must_be_indexed_like_df_images():
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=["Coral"], img1_labels=["Coral"]
    )
    ds = _SyntheticDataset(df_annotations=df_annotations, df_images=df_images)

    with pytest.raises(ValueError, match="indexed like df_images"):
        ds.set_source_rows(pd.Series([1, 2], index=[5, 6]))


def test_source_rows_are_appended_to_each_sample():
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=["Coral"], img1_labels=["Coral"]
    )
    ds = _SyntheticDataset(df_annotations=df_annotations, df_images=df_images, split="val")

    ds.set_source_rows(pd.Series([4, 9], index=df_images.index))

    assert [ds[0][2], ds[1][2]] == [4, 9]


def test_set_source_rows_none_returns_row_zero_for_every_sample():
    df_annotations, df_images = _make_two_image_annotations(
        img0_labels=["Coral"], img1_labels=["Coral"]
    )
    ds = _SyntheticDataset(df_annotations=df_annotations, df_images=df_images, split="val")

    ds.set_source_rows(None)

    image, mask, row = ds[1]
    assert row == 0
    assert mask[5, 5] == 1


# --- _load_item: padding and transform ---


def _single_point_dataset(row: int, col: int, **kwargs: Any) -> _SyntheticDataset:
    df_annotations = _annotations(["Coral"]).assign(row=row, col=col)
    return _SyntheticDataset(
        df_annotations=df_annotations,
        df_images=df_annotations[["image_id"]],
        split="val",
        **kwargs,
    )


def test_padding_is_forwarded_to_the_mask():
    image, mask = _single_point_dataset(10, 10, padding=2)[0]

    assert image.shape == (20, 20, 3)
    expected = np.zeros((20, 20), dtype=np.int32)
    expected[8:12, 8:12] = 1
    np.testing.assert_array_equal(mask, expected)


def test_without_padding_only_the_annotated_pixel_is_set():
    _, mask = _single_point_dataset(10, 10)[0]

    assert mask.sum() == 1 and mask[10, 10] == 1


def test_transform_is_applied_to_image_and_mask_and_image_goes_channels_first():
    # A non-square, non-integer-flip resize: height x2, width x3, nearest for the mask.
    transform = A.Resize(height=40, width=60)

    image, mask = _single_point_dataset(5, 5, transform=transform)[0]

    assert image.shape == (3, 40, 60)
    assert mask.shape == (40, 60)
    expected = np.zeros((40, 60), dtype=np.int32)
    expected[10:12, 15:18] = 1
    np.testing.assert_array_equal(mask, expected)
