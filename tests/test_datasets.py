"""Unit tests for mermaidseg.datasets.utils and BaseCoralDataset error handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.utils import create_annotation_mask


def _make_annotations(rows: list, cols: list, labels: list) -> pd.DataFrame:
    """Build a minimal annotations DataFrame."""
    return pd.DataFrame({"row": rows, "col": cols, "source_label_name": labels})


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


class _SometimesFailDataset(BaseCoralDataset):
    """Subclass whose read_image fails for specific image_ids only.

    Used to verify that ``__getitem__`` recurses to the next index on failure
    and eventually returns a valid sample.
    """

    fail_for: set[str]

    def __init__(self, fail_for: set[str], **kwargs: Any):
        super().__init__(**kwargs)
        self.fail_for = fail_for

    def read_image(self, **row_kwargs) -> Any:
        if row_kwargs.get("image_id") in self.fail_for:
            raise RuntimeError(f"simulated failure for {row_kwargs['image_id']}")
        return np.zeros((8, 8, 3), dtype=np.uint8)


# --- create_annotation_mask ---


def test_create_annotation_mask_basic():
    annotations = _make_annotations([10, 20, 30], [5, 15, 25], ["Coral", "Sand", "Rubble"])
    source_name2id = {"Coral": 1, "Sand": 2, "Rubble": 3}
    mask = create_annotation_mask(annotations, (50, 50), source_name2id)

    assert mask[10, 5] == 1
    assert mask[20, 15] == 2
    assert mask[30, 25] == 3
    assert mask[0, 0] == 0


def test_create_annotation_mask_with_padding():
    annotations = _make_annotations([10], [10], ["Coral"])
    mask = create_annotation_mask(annotations, (50, 50), {"Coral": 1}, padding=2)

    assert np.all(mask[8:12, 8:12] == 1)
    assert mask[7, 10] == 0
    assert mask[12, 10] == 0


def test_create_annotation_mask_padding_bounds_clamped():
    """Large padding at image corners must clamp to bounds without raising IndexError."""
    annotations = _make_annotations([0, 19], [0, 19], ["Coral", "Coral"])
    mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1}, padding=5)

    assert np.all(mask[0:5, 0:5] == 1)
    assert np.all(mask[14:20, 14:20] == 1)


def test_create_annotation_mask_overlapping_padding():
    """When padding regions overlap, later annotations should overwrite earlier ones."""
    annotations = _make_annotations([10, 10], [10, 14], ["Coral", "Sand"])
    mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1, "Sand": 2}, padding=3)

    assert mask[10, 12] == 2
    assert mask[10, 13] == 2

    assert mask[10, 9] == 1
    assert mask[10, 16] == 2


def test_create_annotation_mask_unknown_label_skipped(caplog):
    annotations = _make_annotations([5, 10], [5, 10], ["Coral", "UnknownLabel"])

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"):
        mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1})

    assert mask[5, 5] == 1
    assert mask[10, 10] == 0
    assert "unknown label" in caplog.text.lower() or "UnknownLabel" in caplog.text


@pytest.mark.parametrize(
    "rows,cols,labels",
    [
        ([], [], []),
        ([5], [5], [None]),
    ],
    ids=["empty", "all_null_labels"],
)
def test_create_annotation_mask_produces_zero_mask(rows, cols, labels):
    annotations = _make_annotations(rows, cols, labels)
    mask = create_annotation_mask(annotations, (10, 10), {"Coral": 1})
    assert np.all(mask == 0)


# --- BaseCoralDataset basic API ---


def test_base_dataset_exposes_source_label_attributes(minimal_dataset):
    assert minimal_dataset.source_id2name == {1: "Coral", 2: "Sand"}
    assert minimal_dataset.source_name2id == {"Coral": 1, "Sand": 2}
    assert minimal_dataset.num_source_classes == 3  # background + 2
    assert minimal_dataset.global_offset == 0


def test_base_dataset_set_global_offset_validates_negative(minimal_dataset):
    with pytest.raises(ValueError):
        minimal_dataset.set_global_offset(-1)


def test_base_dataset_set_global_offset_shifts_mask_via_helper():
    """Verify offset arithmetic on the helper directly: offset=10 → values become 11/12."""
    minimal_mask = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    offset = 10
    shifted = np.where(minimal_mask > 0, minimal_mask + offset, minimal_mask)
    assert shifted[0, 0] == 0
    assert shifted[0, 1] == 11
    assert shifted[0, 2] == 12
    assert shifted[1, 1] == 0


# --- BaseCoralDataset.collate_fn ---


def test_collate_fn_filters_none_items(minimal_dataset, caplog):
    img = torch.zeros(3, 4, 4)
    msk = torch.zeros(4, 4, dtype=torch.long)
    batch = [(img, msk), (None, None), (img, msk), (None, None)]

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        images, masks = minimal_dataset.collate_fn(batch)

    assert images.shape[0] == 2
    assert masks.shape[0] == 2
    assert "skipped 2/4" in caplog.text


def test_collate_fn_all_none_returns_empty_tensors(minimal_dataset, caplog):
    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        images, masks = minimal_dataset.collate_fn([(None, None), (None, None)])

    assert images.numel() == 0
    assert masks.numel() == 0
    assert "entire batch" in caplog.text


# --- BaseCoralDataset.__getitem__ ---


def test_base_dataset_getitem_raises_when_all_items_fail(single_image_annotations, caplog):
    """If every item in the dataset fails to load, ``__getitem__`` must raise
    so the caller can't silently train on an empty stream."""
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with (
        caplog.at_level("WARNING", logger="mermaidseg.datasets.utils"),
        pytest.raises(RuntimeError, match="all 1 items failed to load"),
    ):
        _ = ds[0]

    assert "img1" in caplog.text
    assert "RuntimeError" in caplog.text


def test_base_dataset_getitem_recurses_to_next_index_on_failure(capsys):
    """``__getitem__`` should skip failing indices and return the first
    successful sample instead of returning ``(None, None)``."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["bad1", "bad2", "good1"],
            "region_id": [1, 2, 3],
            "region_name": ["r1", "r2", "r3"],
            "source_label_name": ["Coral", "Coral", "Coral"],
            "row": [0, 0, 0],
            "col": [0, 0, 0],
        }
    )
    df_images = (
        df_annotations[["image_id", "region_id", "region_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    ds = _SometimesFailDataset(
        fail_for={"bad1", "bad2"},
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    image, mask = ds[0]
    assert image is not None
    assert mask is not None
    assert image.shape == (8, 8, 3)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "bad1" in combined
    assert "bad2" in combined
    assert ds.num_load_failures() == 2


def test_base_dataset_emit_warning_writes_to_stdout_and_stderr(
    single_image_annotations, capsys
):
    """Skip warnings must be visible on both stdout and stderr so DataLoader
    workers (where ``logging`` is often unconfigured) don't drop them silently."""
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with pytest.raises(RuntimeError):
        _ = ds[0]

    captured = capsys.readouterr()
    assert "img1" in captured.out
    assert "img1" in captured.err
    assert "WARNING" in captured.out
    assert "WARNING" in captured.err


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
    failures = ds.load_failures_df()
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


def test_base_dataset_saves_failure_report_as_parquet(single_image_annotations, tmp_path):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with pytest.raises(RuntimeError):
        _ = ds[0]
    output_path = tmp_path / "load_failures.parquet"
    saved_path = ds.save_load_failures(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == 1
    assert saved_df.iloc[0]["image_id"] == "img1"
