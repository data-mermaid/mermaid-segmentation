from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import mermaidseg.data.parity as parity_module
import mermaidseg.dataset_reconciliation.registry as registry_module
from mermaidseg.data.parity import (
    ParityReport,
    compare,
    format_report,
    new_points,
    old_points,
    run_parity,
)
from tests.mermaidseg.data.conftest import ImageSpec, open_written, publish_points

_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]


def _frame(rows: Sequence[tuple[Any, int, int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS)


_BASE = [("a", 1, 1, "x", "hard"), ("a", 2, 2, "y", "soft"), ("b", 3, 3, "x", "hard")]
_GROUPS = {"a": "g1", "b": "g1"}


def test_compare_equal_frames_is_clean() -> None:
    report = compare(_frame(_BASE), _frame(_BASE), _GROUPS)

    assert report.ok
    assert report == ParityReport()


@pytest.mark.parametrize(
    ("new_rows", "field", "expected"),
    [
        (
            [("a", 1, 9, "x", "hard"), *_BASE[1:]],
            "old_only_points",
            Counter({("a", 1, 1, "hard"): 1}),
        ),
        (
            [("a", 1, 9, "x", "hard"), *_BASE[1:]],
            "new_only_points",
            Counter({("a", 1, 9, "hard"): 1}),
        ),
        (
            [("a", 1, 1, "x", "other"), *_BASE[1:]],
            "new_only_points",
            Counter({("a", 1, 1, "other"): 1}),
        ),
        (
            [*_BASE, ("a", 1, 1, "x", "hard")],
            "new_only_points",
            Counter({("a", 1, 1, "hard"): 1}),
        ),
        (_BASE[:2], "old_only_images", ["b"]),
        ([*_BASE, ("c", 5, 5, "x", "hard")], "new_only_images", ["c"]),
    ],
    ids=["moved-old", "moved-new", "relabel", "duplicate", "old-only-image", "new-only-image"],
)
def test_compare_reports_each_difference_in_its_field(
    new_rows: list[tuple[Any, int, int, str, str | None]], field: str, expected: object
) -> None:
    groups = {**_GROUPS, "c": "g1"}

    report = compare(_frame(_BASE), _frame(new_rows), groups)

    assert not report.ok
    assert getattr(report, field) == expected


def test_compare_ignores_new_groups_without_old_images() -> None:
    groups = {"a": "g1", "b": "g1", "extra": "g1", "z": "g2"}
    new = _frame([*_BASE, ("extra", 7, 7, "x", "hard"), ("z", 8, 8, "x", "hard")])

    report = compare(_frame(_BASE), new, groups)

    assert report.new_only_images == ["extra"]


def test_compare_reports_label_count_diffs() -> None:
    old = _frame([("a", i, i, "x", "m") for i in range(3)])
    new = _frame([("a", i, i, "x", "m") for i in range(2)])

    report = compare(old, new, {"a": "g1"})

    assert report.native_diff == {"x": (3, 2)}
    assert report.mapped_diff == {"m": (3, 2)}


def test_format_report_names_the_verdict() -> None:
    assert format_report(ParityReport()).endswith("PARITY OK")
    failed = format_report(ParityReport(old_only_images=["a"], native_diff={"x": (2, 1)}))
    assert failed.endswith("PARITY FAILED")
    assert "old-only images: 1" in failed
    assert "x: old 2, new 1" in failed


class _StubOld:
    SOURCE_NAME = "catlin_seaview"

    def __init__(self, annotations: pd.DataFrame | None = None) -> None:
        self.df_annotations = (
            annotations
            if annotations is not None
            else pd.DataFrame(
                {
                    "image_id": [10, 10, 11],
                    "row": [1, 2, 3],
                    "col": [4, 5, 6],
                    "source_label_name": ["Hard", "Soft", "Hard"],
                }
            )
        )
        self.source_id2name = {1: "Hard", 2: "Soft"}
        self.source_name2id = {"Hard": 1, "Soft": 2}
        self.offset: int | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def test_old_points_maps_through_the_registry_and_leaves_unmapped_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS,
        "catlin_seaview",
        lambda: {"Hard": "Hard Coral", "Soft": None},
    )

    points = old_points(_StubOld())

    assert list(points.columns) == _COLUMNS
    assert points["image_id"].tolist() == ["10", "10", "11"]
    assert points["native_name"].tolist() == ["Hard", "Soft", "Hard"]
    assert points["mapped"].tolist() == ["hard coral", None, "hard coral"]


_LABELS = [
    {"label_id": 1, "name": "Acropora", "mermaid_benthic_attribute": "Hard Coral"},
    {"label_id": 2, "name": "Rubble", "mermaid_benthic_attribute": None},
]
_IMAGES = [ImageSpec("i1"), ImageSpec("i2")]


def test_new_points_lowercases_the_attribute_and_keeps_null_none(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i1", 2, 2, 2), ("i2", 3, 3, 1)],
    )

    assert version.points is not None
    points = new_points(version, version.points.to_pandas())

    assert list(points.columns) == _COLUMNS
    assert points["native_name"].tolist() == ["Acropora", "Rubble", "Acropora"]
    assert points["mapped"].tolist() == ["hard coral", None, "hard coral"]
    assert points["image_id"].tolist() == ["i1", "i1", "i2"]


def _old_stub(rows: list[tuple[str, int, int]]) -> _StubOld:
    return _StubOld(
        pd.DataFrame(
            {
                "image_id": [r[0] for r in rows],
                "row": [r[1] for r in rows],
                "col": [r[2] for r in rows],
                "source_label_name": ["Acropora"] * len(rows),
            }
        )
    )


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, old_rows: list[tuple[str, int, int]]
) -> tuple[ParityReport, Path]:
    fake, pin = publish_points(
        tmp_path / "pub",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub(old_rows)
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    cache_dir = tmp_path / "cache"
    report = run_parity(
        {},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=fake,
        annotators=None,
    )
    return report, cache_dir


def test_run_parity_passes_on_matching_points_without_downloading_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, cache_dir = _run(monkeypatch, tmp_path, [("i1", 1, 1), ("i2", 3, 3)])

    assert report.ok
    assert {p.suffix for p in cache_dir.rglob("*") if p.is_file()} == {".json", ".parquet"}


def test_run_parity_reports_a_point_moved_by_one_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, _ = _run(monkeypatch, tmp_path, [("i1", 2, 1), ("i2", 3, 3)])

    assert not report.ok
    assert report.old_only_points == Counter({("i1", 2, 1, "hard coral"): 1})
    assert report.new_only_points == Counter({("i1", 1, 1, "hard coral"): 1})
