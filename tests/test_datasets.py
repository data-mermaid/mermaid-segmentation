"""Unit tests for mermaidseg.datasets.utils and BaseCoralDataset error handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.datasets.dataset import BaseCoralDataset
from mermaidseg.datasets.utils import create_annotation_mask


def _make_annotations(rows: list, cols: list, labels: list) -> pd.DataFrame:
    """Build a minimal annotations DataFrame."""
    return pd.DataFrame({"row": rows, "col": cols, "benthic_attribute_name": labels})


@pytest.fixture
def minimal_dataset() -> BaseCoralDataset:
    """Smallest valid BaseCoralDataset — no real images needed."""
    df_annotations = pd.DataFrame(
        {
            "image_id": ["img1", "img2"],
            "region_id": [1, 2],
            "region_name": ["r1", "r2"],
            "benthic_attribute_name": ["Coral", "Sand"],
            "row": [10, 20],
            "col": [10, 20],
        }
    )
    df_images = df_annotations[["image_id", "region_id", "region_name"]].drop_duplicates().reset_index(drop=True)
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
            "benthic_attribute_name": ["Coral"],
            "row": [5],
            "col": [5],
        }
    )
    df_images = df_annotations[["image_id", "region_id", "region_name"]].drop_duplicates().reset_index(drop=True)
    return df_annotations, df_images


class _AlwaysFailDataset(BaseCoralDataset):
    """Minimal subclass whose read_image always raises."""

    def read_image(self, **row_kwargs) -> Any:
        raise RuntimeError("simulated read failure")


# --- create_annotation_mask ---


def test_create_annotation_mask_basic():
    annotations = _make_annotations([10, 20, 30], [5, 15, 25], ["Coral", "Sand", "Rubble"])
    label2id = {"Coral": 1, "Sand": 2, "Rubble": 3}
    mask = create_annotation_mask(annotations, (50, 50), label2id)

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
    # Two annotations close together: (10, 10) and (10, 14)
    # With padding=3, their regions will overlap at rows 7-13, cols 10-17
    annotations = _make_annotations([10, 10], [10, 14], ["Coral", "Sand"])
    mask = create_annotation_mask(annotations, (20, 20), {"Coral": 1, "Sand": 2}, padding=3)

    # The overlap region (cols 12-13) should have Sand (value 2) since Sand annotation is second
    assert mask[10, 12] == 2
    assert mask[10, 13] == 2

    # Non-overlapping parts should have their respective values
    assert mask[10, 9] == 1  # Coral only
    assert mask[10, 16] == 2  # Sand only


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


# --- BaseCoralDataset.collate_fn ---


def test_collate_fn_filters_none_items(minimal_dataset, caplog):
    img = torch.zeros(3, 4, 4)
    msk = torch.zeros(4, 4, dtype=torch.long)
    batch = [(img, msk), (None, None), (img, msk), (None, None)]

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        images, masks = minimal_dataset.collate_fn(batch)

    assert images.shape[0] == 2
    assert masks.shape[0] == 2
    assert "skipped 2/4" in caplog.text


def test_collate_fn_all_none_returns_empty_tensors(minimal_dataset, caplog):
    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        images, masks = minimal_dataset.collate_fn([(None, None), (None, None)])

    assert images.numel() == 0
    assert masks.numel() == 0
    assert "entire batch" in caplog.text


# --- BaseCoralDataset.__getitem__ ---


def test_base_dataset_getitem_skips_and_logs_on_read_failure(single_image_annotations, caplog):
    df_annotations, df_images = single_image_annotations
    ds = _AlwaysFailDataset(
        df_annotations=df_annotations,
        df_images=df_images,
        class_subset=["Coral"],
    )

    with caplog.at_level("WARNING", logger="mermaidseg.datasets.dataset"):
        result = ds[0]

    assert result == (None, None)
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

    _ = ds[0]
    output_path = tmp_path / "load_failures.parquet"
    saved_path = ds.save_load_failures(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == 1
    assert saved_df.iloc[0]["image_id"] == "img1"
