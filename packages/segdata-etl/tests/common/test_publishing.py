"""Publish helpers shared by every versioned-dataset builder."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from segdata_etl.common import publishing
from segdata_etl.common.local_store import LocalStore
from segdata_spec.errors import PublishRefused


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
    assert publishing.count_labels(points).to_pylist() == [
        {"image_id": "a", "label_id": 5, "n": 2},
        {"image_id": "a", "label_id": 7, "n": 1},
        {"image_id": "b", "label_id": 5, "n": 1},
    ]


def _store_with_object(tmp_path: Path, key: str) -> LocalStore:
    store = LocalStore(tmp_path)
    store.put_object(Bucket="b", Key=key, Body=b"x")
    return store


def test_require_version_absent_refuses_a_folder_holding_an_object(tmp_path: Path) -> None:
    store = _store_with_object(tmp_path, "datasets/ds/20261123-abc1234/manifest.json")

    with pytest.raises(PublishRefused):
        publishing.require_version_absent(store, "b", "ds", "20261123-abc1234")


def test_require_version_absent_accepts_a_sibling_version_folder(tmp_path: Path) -> None:
    store = _store_with_object(tmp_path, "datasets/ds/20261123-abc1234/manifest.json")

    publishing.require_version_absent(store, "b", "ds", "20261123-abc1234-2")


def test_pick_version_suffixes_a_version_already_listed_for_the_dataset(tmp_path: Path) -> None:
    store = _store_with_object(tmp_path, "datasets/ds/20261123-abc1234/manifest.json")

    version = publishing.pick_version(
        store, "b", "ds", git_sha="abc1234def5678", today=date(2026, 11, 23)
    )

    assert version == "20261123-abc1234-2"


def test_pick_version_ignores_versions_of_another_dataset(tmp_path: Path) -> None:
    store = _store_with_object(tmp_path, "datasets/other/20261123-abc1234/manifest.json")

    version = publishing.pick_version(
        store, "b", "ds", git_sha="abc1234def5678", today=date(2026, 11, 23)
    )

    assert version == "20261123-abc1234"
