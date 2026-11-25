"""Tests for the release helpers shared by every versioned-dataset builder."""

from __future__ import annotations

import pyarrow as pa

from segdata_etl.common import release


def _points(*rows: tuple[str, int, int, int]) -> pa.Table:
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "row": [r[1] for r in rows],
            "col": [r[2] for r in rows],
            "label_id": [r[3] for r in rows],
            "annotator": ["default"] * len(rows),
        }
    )


def test_count_labels_counts_points_per_image_and_label_sorted_by_image_then_label() -> None:
    points = _points(("b", 0, 0, 5), ("a", 0, 0, 7), ("a", 1, 1, 5), ("a", 2, 2, 5))
    assert release.count_labels(points).to_pylist() == [
        {"image_id": "a", "label_id": 5, "n": 2},
        {"image_id": "a", "label_id": 7, "n": 1},
        {"image_id": "b", "label_id": 5, "n": 1},
    ]
