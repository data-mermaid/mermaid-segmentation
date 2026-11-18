"""Unit tests for mermaidseg.datasets.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mermaidseg.datasets.utils import create_annotation_mask


def _make_annotations(rows: list, cols: list, labels: list) -> pd.DataFrame:
    """Build a minimal annotations DataFrame."""
    return pd.DataFrame({"row": rows, "col": cols, "source_label_name": labels})


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


def test_create_annotation_mask_offset_matches_where_pass():
    """Folded offset must equal the previous ``np.where(mask > 0, mask + offset, mask)`` pass."""
    annotations = _make_annotations([10, 20], [5, 15], ["Coral", "Sand"])
    source_name2id = {"Coral": 1, "Sand": 2}
    offset = 10
    baseline = create_annotation_mask(annotations, (50, 50), source_name2id)
    shifted = np.where(baseline > 0, baseline + offset, baseline)
    folded = create_annotation_mask(annotations, (50, 50), source_name2id, offset=offset)
    np.testing.assert_array_equal(folded, shifted)
    assert folded.dtype == np.int32


def test_create_annotation_mask_dtype_is_int32():
    annotations = _make_annotations([1], [1], ["Coral"])
    mask = create_annotation_mask(annotations, (8, 8), {"Coral": 1})
    assert mask.dtype == np.int32


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
