from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import mermaidseg.data.parity as parity_module
import mermaidseg.dataset_reconciliation.registry as registry_module
from mermaidseg.data.parity import (
    DenseParityReport,
    ParityReport,
    compare,
    compare_dense,
    format_report,
    new_dense_counts,
    new_points,
    old_dense_counts,
    old_points,
    run_parity,
)
from segdata_spec.errors import ChecksumMismatch, PinMismatch
from segdata_spec.manifest import Pin
from segdata_spec.read import open_version
from tests.mermaidseg.data.conftest import (
    BUCKET,
    IMAGE_BUCKET,
    IMAGE_PREFIX,
    ImageSpec,
    open_written,
    publish_points,
    write_version,
)

_COLUMNS = ["image_id", "row", "col", "native_name", "mapped"]


def _frame(rows: Sequence[tuple[Any, int, int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS)


_BASE = [("a", 1, 1, "x", "hard"), ("a", 2, 2, "y", "soft"), ("b", 3, 3, "x", "hard")]
_GROUPS = {"a": "g1", "b": "g1"}


def test_compare_equal_frames_is_clean() -> None:
    report = compare(_frame(_BASE), _frame(_BASE), _GROUPS)

    assert report.ok
    assert report == ParityReport(points_matched=3, groups_compared=1)


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


def test_compare_lists_new_groups_without_old_images_apart_from_the_diff() -> None:
    groups = {"a": "g1", "b": "g1", "z": "g2", "y": "g2", "w": "g3"}
    new = _frame(
        [*_BASE, ("z", 8, 8, "x", "hard"), ("z", 9, 9, "x", "hard"), ("y", 1, 1, "x", "soft")]
        + [("w", 2, 2, "y", "soft")]
    )

    report = compare(_frame(_BASE), new, groups)

    assert report.ok
    assert report.excluded_groups == {"g2": (2, 3), "g3": (1, 1)}
    assert report.groups_compared == 1


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


def test_format_report_states_what_was_compared_and_the_excluded_groups() -> None:
    text = format_report(
        ParityReport(points_matched=40, groups_compared=3, excluded_groups={"g9": (2, 7)})
    )

    lines = text.splitlines()
    assert "points matched: 40" in lines
    assert "groups compared: 3" in lines
    assert "new groups excluded: 1" in lines
    assert "  g9: 2 images, 7 points" in lines
    assert text.endswith("PARITY OK")


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
    assert isinstance(report, ParityReport)
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


def _stage_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Pin, _StubOld]:
    cache_dir, bucket, pin = write_version(
        tmp_path,
        kind="points",
        images=_IMAGES,
        labels=_LABELS,
        points=[("i1", 1, 1, 1), ("i2", 3, 3, 1)],
    )
    monkeypatch.setattr(parity_module, "DATASET_BUCKET", bucket)
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS, "catlin_seaview", lambda: {"Acropora": "Hard Coral"}
    )
    stub = _old_stub([("i1", 1, 1), ("i2", 3, 3)])
    stub.source_name2id = {"Acropora": 1}
    stub.source_id2name = {1: "Acropora"}
    return cache_dir, pin, stub


def test_run_parity_expands_the_default_block_before_building(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    cfg_seen: list[Any] = []

    def _build(cfg: Any, **_: Any) -> dict[tuple[str, str], Any]:
        cfg_seen.append(cfg)
        return {("ds", "train"): stub}

    monkeypatch.setattr(parity_module, "build_datasets", _build)

    run_parity(
        {"default": {"train": {"k": 1}}, "ds": {"val": {"v": 2}}},
        dataset="ds",
        split="train",
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        cache_dir=cache_dir,
        client=None,
        annotators=None,
        offline=True,
    )

    assert cfg_seen == [{"ds": {"train": {"k": 1}, "val": {"v": 2}}}]


def test_run_parity_offline_opens_the_staged_cache_without_a_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    args: dict[str, Any] = {
        "dataset": "ds",
        "split": "train",
        "version": pin.version,
        "cache_dir": cache_dir,
        "client": None,
        "annotators": None,
        "offline": True,
    }

    assert run_parity({}, manifest_sha256=pin.manifest_sha256, **args).ok
    with pytest.raises(PinMismatch):
        run_parity({}, manifest_sha256="0" * 64, **args)


def test_run_parity_online_without_a_client_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="client"):
        run_parity(
            {},
            dataset="ds",
            split="train",
            version="v",
            manifest_sha256="x",
            cache_dir=tmp_path,
            client=None,
            annotators=None,
        )


_TILE_MASK_A = np.array([[0, 1], [2, 2]], np.uint8)
_TILE_MASK_B = np.array([[3, 3], [3, 0]], np.uint8)
_DENSE_LABELS = [
    {"label_id": 1, "name": "Coral", "mermaid_benthic_attribute": "Hard Coral"},
    {"label_id": 2, "name": "Sand", "mermaid_benthic_attribute": None},
    {"label_id": 3, "name": "Turf", "mermaid_benthic_attribute": "Turf Algae"},
]
_BY_MAPPING = {"Coral": "Hard Coral", "Sand": None, "Turf": "Turf Algae"}


def _dense_version(tmp_path: Path) -> tuple[Path, str, Pin]:
    return write_version(
        tmp_path,
        kind="dense",
        images=[
            ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
            ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B, source_name="7_9.png"),
        ],
        labels=_DENSE_LABELS,
    )


def test_new_dense_counts_keys_tiles_by_group_and_source_name_stem(tmp_path: Path) -> None:
    cache_dir, bucket, pin = _dense_version(tmp_path)

    counts, mapped = new_dense_counts(open_version(cache_dir, bucket, pin))

    assert counts == {
        ("RS24", "0_5"): Counter({1: 1, 2: 2}),
        ("RS24", "7_9"): Counter({3: 3}),
    }
    assert mapped == {1: "hard coral", 2: None, 3: "turf algae"}


def test_new_dense_counts_rejects_a_tile_without_source_name(tmp_path: Path) -> None:
    version = open_written(
        tmp_path, kind="dense", images=[ImageSpec("t", mask=_TILE_MASK_A)], labels=_DENSE_LABELS
    )

    with pytest.raises(ValueError, match="source_name"):
        new_dense_counts(version)


class _StubDense:
    SOURCE_NAME = "benthos_yuval"

    def __init__(self, masks: dict[tuple[str, str], np.ndarray[Any, Any]]) -> None:
        self._masks = masks
        self.df_images = pd.DataFrame(
            {"site": [k[0] for k in masks], "image_id": [k[1] for k in masks]}
        )
        self._classes_global = {"Coral": 1, "Sand": 2, "Turf": 3, "Dropped": 4}
        self.source_id2name = {1: "Coral", 2: "Sand", 3: "Turf"}
        self.source_name2id = {"Coral": 1, "Sand": 2, "Turf": 3}

    def set_global_offset(self, offset: int) -> None:
        pass

    def read_label(self, image_id: str, site: str, **_: Any) -> np.ndarray[Any, Any]:
        return self._masks[(site, image_id)]


def test_old_dense_counts_drop_the_unlabeled_class_and_map_through_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    ds = _StubDense({("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B})

    counts, mapped = old_dense_counts(ds)

    assert counts == {
        ("RS24", "0_5"): Counter({1: 1, 2: 2}),
        ("RS24", "7_9"): Counter({3: 3}),
    }
    assert mapped == {1: "hard coral", 2: None, 3: "turf algae", 4: None}


_MAPPED = {1: "hard coral", 2: None}
_TILES = {("s", "0_1"): Counter({1: 4, 2: 6}), ("s", "2_3"): Counter({1: 5})}


def test_compare_dense_equal_counts_is_clean() -> None:
    report = compare_dense(_TILES, _TILES, _MAPPED, _MAPPED)

    assert report == DenseParityReport(tiles_matched=2, sites_compared=1)
    assert report.ok


def test_compare_dense_reports_a_tile_whose_label_count_differs() -> None:
    new = {**_TILES, ("s", "0_1"): Counter({1: 3, 2: 6})}

    report = compare_dense(_TILES, new, _MAPPED, _MAPPED)

    assert not report.ok
    assert report.tile_diff == {("s", "0_1"): {1: (4, 3)}}
    assert report.mapped_diff == {"hard coral": (9, 8)}
    assert report.old_only_tiles == report.new_only_tiles == []


def test_compare_dense_reports_tiles_on_one_side_and_ignores_unrelated_sites() -> None:
    new = {
        ("s", "0_1"): _TILES[("s", "0_1")],
        ("s", "9_9"): Counter({1: 1}),
        ("other", "0_1"): Counter({1: 7}),
    }

    report = compare_dense(_TILES, new, _MAPPED, _MAPPED)

    assert report.old_only_tiles == [("s", "2_3")]
    assert report.new_only_tiles == [("s", "9_9")]
    assert report.tile_diff == {}
    assert report.tiles_matched == 1
    assert report.excluded_sites == {"other": (1, 7)}


def test_compare_dense_maps_each_side_with_its_own_table() -> None:
    new_mapped = {1: "soft coral", 2: None}

    report = compare_dense(_TILES, _TILES, _MAPPED, new_mapped)

    assert report.tile_diff == {}
    assert report.mapped_diff == {"hard coral": (9, 0), "soft coral": (0, 9)}


def test_compare_dense_reports_swapped_label_mappings_whose_totals_agree() -> None:
    tiles = {("s", "0_1"): Counter({1: 5, 2: 5})}
    old_mapped = {1: "a", 2: "b", 3: "c"}
    new_mapped = {1: "b", 2: "a", 3: "d"}

    report = compare_dense(tiles, tiles, old_mapped, new_mapped)

    assert report.mapped_diff == {}
    assert report.label_map_diff == {1: ("a", "b"), 2: ("b", "a")}
    assert not report.ok


def test_format_dense_report_lists_ten_tile_diffs_then_the_remainder() -> None:
    old = {("s", f"{i:02d}"): Counter({1: 2}) for i in range(12)}
    new = {("s", f"{i:02d}"): Counter({1: 1}) for i in range(12)}

    text = format_report(compare_dense(old, new, _MAPPED, _MAPPED))

    assert text.endswith("PARITY FAILED")
    assert "s/00: label 1 old 2, new 1" in text
    assert "s/09: label 1 old 2, new 1" in text
    assert "s/10:" not in text
    assert "+2 more" in text
    assert format_report(DenseParityReport()).endswith("PARITY OK")


def test_run_parity_dispatches_a_dense_version_to_the_dense_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, bucket, pin = _dense_version(tmp_path)
    monkeypatch.setattr(parity_module, "DATASET_BUCKET", bucket)
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    masks = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubDense(masks)}
    )
    args: dict[str, Any] = {
        "dataset": "ds",
        "split": "val",
        "version": pin.version,
        "manifest_sha256": pin.manifest_sha256,
        "cache_dir": cache_dir,
        "client": None,
        "annotators": None,
        "offline": True,
    }

    assert run_parity({}, **args).ok

    masks[("RS24", "7_9")] = np.array([[3, 3], [3, 3]], np.uint8)
    report = run_parity({}, **args)
    assert isinstance(report, DenseParityReport)
    assert report.tile_diff == {("RS24", "7_9"): {3: (4, 3)}}


def test_format_dense_report_lists_every_finding_and_what_was_compared() -> None:
    report = DenseParityReport(
        old_only_tiles=[("s", "1_1")],
        new_only_tiles=[("s", "2_2")],
        mask_mismatches=[("s", "3_3")],
        label_count_mismatches={("t", "4_4"): {3: (3, 4)}},
        label_map_diff={1: ("a", None)},
        tiles_matched=12,
        sites_compared=2,
        excluded_sites={"t": (5, 900)},
    )

    lines = format_report(report).splitlines()

    for line in [
        "old-only tiles: 1",
        "  s/1_1",
        "new-only tiles: 1",
        "  s/2_2",
        "mask pixel mismatches: 1",
        "  s/3_3",
        "label_counts vs mask: 1 tiles",
        "  t/4_4: label 3 table 3, mask 4",
        "label id mapping:",
        "  1: old a, new None",
        "tiles matched: 12",
        "sites compared: 2",
        "new sites excluded: 1",
        "  t: 5 tiles, 900 pixels",
    ]:
        assert line in lines
    assert lines[-1] == "PARITY FAILED"


def test_new_dense_counts_ignores_a_label_zero_row_in_label_counts(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("t", group="g", mask=_TILE_MASK_B, source_name="7_9.png")],
        labels=_DENSE_LABELS,
        label_counts=[("t", 0, 1), ("t", 3, 3)],
    )

    counts, _ = new_dense_counts(version)

    assert counts == {("g", "7_9"): Counter({3: 3})}


def test_new_dense_counts_rejects_two_tiles_with_the_same_key(tmp_path: Path) -> None:
    version = open_written(
        tmp_path,
        kind="dense",
        images=[
            ImageSpec("a", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
            ImageSpec("b", group="RS24", mask=_TILE_MASK_B, source_name="0_5.jpg"),
        ],
        labels=_DENSE_LABELS,
    )

    with pytest.raises(ValueError, match=r"\('RS24', '0_5'\)"):
        new_dense_counts(version)


def _offline_args(cache_dir: Path, pin: Pin, split: str = "val") -> dict[str, Any]:
    return {
        "dataset": "ds",
        "split": split,
        "version": pin.version,
        "manifest_sha256": pin.manifest_sha256,
        "cache_dir": cache_dir,
        "client": None,
        "annotators": None,
        "offline": True,
    }


def _run_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    images: Sequence[ImageSpec],
    old_masks: dict[tuple[str, str], np.ndarray[Any, Any]],
    label_counts: Sequence[tuple[str, int, int]] | None = None,
    labels: Sequence[dict[str, Any]] = _DENSE_LABELS,
) -> DenseParityReport:
    cache_dir, bucket, pin = write_version(
        tmp_path, kind="dense", images=images, labels=labels, label_counts=label_counts
    )
    monkeypatch.setattr(parity_module, "DATASET_BUCKET", bucket)
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubDense(old_masks)}
    )
    report = run_parity({}, **_offline_args(cache_dir, pin))
    assert isinstance(report, DenseParityReport)
    return report


_TILE_MASK_B_MOVED = np.array([[3, 3], [0, 3]], np.uint8)


def test_run_parity_dense_reports_a_stored_mask_whose_pixels_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = [
        ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
        ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B_MOVED, source_name="7_9.png"),
    ]
    old = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}

    report = _run_dense(monkeypatch, tmp_path, images, old)

    assert report.tile_diff == {}
    assert report.mask_mismatches == [("RS24", "7_9")]
    assert not report.ok


def test_run_parity_dense_reports_label_counts_that_disagree_with_the_stored_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = [
        ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
        ImageSpec("X_7_9", group="X", mask=_TILE_MASK_B, source_name="7_9.png"),
    ]
    counts = [("RS24_0_5", 1, 1), ("RS24_0_5", 2, 2), ("X_7_9", 3, 4)]

    report = _run_dense(monkeypatch, tmp_path, images, {("RS24", "0_5"): _TILE_MASK_A}, counts)

    assert report.label_count_mismatches == {("X", "7_9"): {3: (4, 3)}}
    assert report.excluded_sites == {"X": (1, 4)}
    assert not report.ok


_PARITY_IMAGES = [
    ImageSpec("RS24_0_5", group="RS24", mask=_TILE_MASK_A, source_name="0_5.png"),
    ImageSpec("RS24_7_9", group="RS24", mask=_TILE_MASK_B, source_name="7_9.png"),
]
_PARITY_OLD_MASKS = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B}


def test_run_parity_dense_reports_label_names_swapped_under_the_same_attributes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    swapped = [
        {**_DENSE_LABELS[0], "name": "Turf"},
        _DENSE_LABELS[1],
        {**_DENSE_LABELS[2], "name": "Coral"},
    ]

    report = _run_dense(monkeypatch, tmp_path, _PARITY_IMAGES, _PARITY_OLD_MASKS, labels=swapped)

    assert report.native_name_diff == {1: ("coral", "turf"), 3: ("turf", "coral")}
    assert report.tile_diff == {}
    assert report.mapped_diff == {}
    assert not report.ok


def test_run_parity_dense_ignores_a_label_name_that_differs_only_in_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recased = [{**_DENSE_LABELS[0], "name": "coral"}, *_DENSE_LABELS[1:]]

    report = _run_dense(monkeypatch, tmp_path, _PARITY_IMAGES, _PARITY_OLD_MASKS, labels=recased)

    assert report.native_name_diff == {}
    assert report.ok


def test_dense_report_with_only_excluded_sites_is_ok() -> None:
    assert DenseParityReport(excluded_sites={"X": (1, 4)}).ok


def test_format_dense_report_lists_ten_label_name_diffs_then_the_remainder() -> None:
    diff = {i: (f"a{i}", f"b{i}") for i in range(1, 13)}

    text = format_report(DenseParityReport(native_name_diff=diff))

    assert "label id native names:" in text
    assert "  1: old a1, new b1" in text
    assert "  10: old a10, new b10" in text
    assert "  11:" not in text
    assert "+2 more" in text
    assert text.endswith("PARITY FAILED")


def _mask_file(cache_dir: Path, image_id: str) -> Path:
    return cache_dir / IMAGE_BUCKET / IMAGE_PREFIX / f"{image_id}.png"


def _stage_dense(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Pin]:
    cache_dir, bucket, pin = _dense_version(tmp_path)
    monkeypatch.setattr(parity_module, "DATASET_BUCKET", bucket)
    monkeypatch.setitem(registry_module._REMOTE_FETCHERS, "benthos_yuval", lambda: _BY_MAPPING)
    old = {("RS24", "0_5"): _TILE_MASK_A, ("RS24", "7_9"): _TILE_MASK_B_MOVED}
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubDense(old)}
    )
    return cache_dir, pin


def test_run_parity_offline_verifies_the_staged_masks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    Image.fromarray(_TILE_MASK_B_MOVED).save(_mask_file(cache_dir, "RS24_7_9"), "PNG")

    with pytest.raises(ChecksumMismatch, match="RS24_7_9.png"):
        run_parity({}, **_offline_args(cache_dir, pin))


def test_run_parity_offline_fails_on_a_mask_that_is_not_staged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    _mask_file(cache_dir, "RS24_7_9").unlink()

    with pytest.raises(FileNotFoundError, match="offline: .*RS24_7_9.png is not staged"):
        run_parity({}, **_offline_args(cache_dir, pin))


def test_run_parity_offline_verifies_the_staged_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    stub.df_annotations.loc[0, "row"] = 2
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})
    path = cache_dir / BUCKET / "datasets" / pin.name / pin.version / "points.parquet"
    table = pq.read_table(path)
    rows = table.column("row").to_pylist()
    rows[0] = 2
    row_index = table.schema.get_field_index("row")
    row_type = table.schema.field("row").type
    pq.write_table(table.set_column(row_index, "row", pa.array(rows, row_type)), path)

    with pytest.raises(ChecksumMismatch, match="points.parquet"):
        run_parity({}, **_offline_args(cache_dir, pin, split="train"))


def test_run_parity_rejects_an_old_split_without_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin, stub = _stage_points(monkeypatch, tmp_path)
    stub.df_annotations = stub.df_annotations.iloc[0:0]
    monkeypatch.setattr(parity_module, "build_datasets", lambda *a, **k: {("ds", "train"): stub})

    with pytest.raises(ValueError, match="no points"):
        run_parity({}, **_offline_args(cache_dir, pin, split="train"))


def test_run_parity_rejects_an_old_split_without_tiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir, pin = _stage_dense(monkeypatch, tmp_path)
    monkeypatch.setattr(
        parity_module, "build_datasets", lambda *a, **k: {("ds", "val"): _StubDense({})}
    )

    with pytest.raises(ValueError, match="no tiles"):
        run_parity({}, **_offline_args(cache_dir, pin))
