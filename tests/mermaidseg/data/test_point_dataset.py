from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mermaidseg.data import PointDataset
from mermaidseg.data.point_dataset import rasterize_points
from tests.mermaidseg.data.conftest import ImageSpec, open_written

LABELS = {5: "five", 7: "seven", 9: "nine"}


@pytest.mark.parametrize(
    ("annotators", "label_ids", "painted"),
    [
        # host labels (1, 1) as 5; archived alone labels (2, 2) as 7; 5 and 7 tie on one point.
        (["host", "archived"], {1: 5, 2: 7}, (1, 2, 0)),
        # archived labels both points as 7.
        (["archived", "host"], {1: 7}, (1, 1, 0)),
    ],
    ids=["host-first", "archived-first"],
)
def test_annotator_priority_falls_back_and_drops_unlisted(
    tmp_path: Path,
    annotators: list[str],
    label_ids: dict[int, int],
    painted: tuple[int, int, int],
) -> None:
    points = [
        ("i1", 1, 1, 5, "host"),
        ("i1", 1, 1, 7, "archived"),
        ("i1", 2, 2, 7, "archived"),
        ("i1", 3, 3, 9, "visitor1"),
    ]
    version = open_written(
        tmp_path, kind="points", images=[ImageSpec("i1")], labels=LABELS, points=points
    )
    ds = PointDataset(version, split="val", annotators=annotators)
    _, mask = ds[0]
    assert ds.source_id2label_id == label_ids
    assert (mask[1, 1], mask[2, 2], mask[3, 3]) == painted


@pytest.mark.parametrize(
    ("annotators_in_table", "expectation"),
    [
        (["default", "default"], nullcontext()),
        (["a", "b"], pytest.raises(ValueError, match="single annotator")),
    ],
    ids=["single", "several"],
)
def test_annotators_none_requires_a_single_annotator(
    tmp_path: Path, annotators_in_table: list[str], expectation: AbstractContextManager[None]
) -> None:
    points = [("i1", i, i, 5, who) for i, who in enumerate(annotators_in_table)]
    version = open_written(
        tmp_path, kind="points", images=[ImageSpec("i1")], labels=LABELS, points=points
    )
    with expectation:
        ds = PointDataset(version, split="val")
        assert ds[0][1].astype(bool).sum() == 2


def test_image_without_a_selected_point_is_dropped(tmp_path: Path) -> None:
    points = [("i1", 0, 0, 5, "host"), ("i2", 0, 0, 5, "visitor1")]
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1"), ImageSpec("i2")],
        labels=LABELS,
        points=points,
    )
    ds = PointDataset(version, split="val", annotators=["host"])
    assert list(ds.images["image_id"]) == ["i1"]


@pytest.mark.parametrize(
    ("row", "col", "raises"),
    [(8, 0, True), (0, 8, True), (-1, 0, True), (7, 7, False)],
    ids=["row=height", "col=width", "negative", "last-pixel"],
)
def test_points_must_lie_inside_the_image(tmp_path: Path, row: int, col: int, raises: bool) -> None:
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1", height=8, width=8)],
        labels=LABELS,
        points=[("i1", row, col, 5)],
    )
    expectation = pytest.raises(ValueError, match="outside") if raises else nullcontext()
    with expectation:
        assert len(PointDataset(version, split="val")) == 1


@pytest.mark.parametrize(
    ("listed", "named"),
    [(["Host", "archived"], ["Host"]), (["x", "y"], ["x", "y"])],
    ids=["one-typo", "all-wrong"],
)
def test_unknown_annotator_raises_naming_each(
    tmp_path: Path, listed: list[str], named: list[str]
) -> None:
    points = [("i1", 1, 1, 5, "host"), ("i1", 2, 2, 7, "archived")]
    version = open_written(
        tmp_path, kind="points", images=[ImageSpec("i1")], labels=LABELS, points=points
    )
    with pytest.raises(ValueError, match="annotators") as info:
        PointDataset(version, split="val", annotators=listed)
    assert all(name in str(info.value) for name in named)
    assert "'archived'" not in str(info.value)


def _selected_rows(ds: PointDataset) -> list[tuple[str, int, int, int, str]]:
    frame = ds.points
    return [
        (str(r.image_id), int(r.row), int(r.col), int(r.label_id), str(r.annotator))
        for r in frame.itertuples()
    ]


def test_points_holds_the_selected_rows(tmp_path: Path) -> None:
    points = [
        ("i1", 1, 1, 5, "host"),
        ("i1", 1, 1, 7, "archived"),
        ("i1", 2, 2, 7, "archived"),
        ("i1", 3, 3, 9, "visitor1"),
    ]
    version = open_written(
        tmp_path, kind="points", images=[ImageSpec("i1")], labels=LABELS, points=points
    )
    ds = PointDataset(version, split="val", annotators=["host", "archived"])
    assert list(ds.points.columns) == ["image_id", "row", "col", "label_id", "annotator"]
    assert sorted(_selected_rows(ds)) == [
        ("i1", 1, 1, 5, "host"),
        ("i1", 2, 2, 7, "archived"),
    ]


def test_points_omit_a_dropped_images_points(tmp_path: Path) -> None:
    points = [("i1", 0, 0, 5), ("i2", 1, 1, 7), ("i2", 2, 2, 7)]
    version = _write_points(tmp_path, points, ["i1", "i2"])
    version.image_path({"key": "i2.jpg"}).unlink()
    ds = PointDataset(version, split="val", on_missing_images="drop")
    assert _selected_rows(ds) == [("i1", 0, 0, 5, "default")]


def _write_points(tmp_path: Path, points: list[tuple[str, int, int, int]], ids: list[str]) -> Any:
    return open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec(i) for i in ids],
        labels=LABELS,
        points=points,
    )


def test_set_source_vocabulary_repaints_points_with_the_new_ids(tmp_path: Path) -> None:
    # Local ids are five=1 (two points) and seven=2; the new vocabulary reverses them.
    points = [("i1", 0, 0, 5), ("i1", 1, 1, 5), ("i1", 2, 2, 7)]
    ds = PointDataset(_write_points(tmp_path, points, ["i1"]), split="val")
    ds.set_source_vocabulary({1: "seven", 2: "five"}, {"seven": 1, "five": 2}, 3)
    _, mask = ds[0]
    assert (mask[0, 0], mask[1, 1], mask[2, 2], mask[3, 3]) == (2, 2, 1, 0)


@pytest.mark.parametrize(
    ("points", "padding", "offset", "expected"),
    [
        # No padding: one pixel per point.
        (
            [(0, 0, 1), (2, 3, 2)],
            None,
            0,
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # Padding 0 paints one pixel, and the offset shifts foreground ids.
        (
            [(0, 0, 1), (2, 3, 2)],
            0,
            10,
            [
                [11, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 12, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # Padding 2 paints [r-2 : r+2, c-2 : c+2], so the end is exclusive.
        (
            [(3, 3, 1)],
            2,
            0,
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # A point near the top-left clips its start at 0 instead of wrapping.
        (
            [(0, 1, 3)],
            2,
            0,
            [
                [3, 3, 3, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # The later point wins where two coincide.
        (
            [(5, 5, 1), (5, 5, 2)],
            None,
            0,
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
            ],
        ),
        # The later point wins on the overlapping part of two padded squares.
        (
            [(2, 2, 1), (3, 3, 2)],
            2,
            0,
            [
                [1, 1, 1, 1, 0, 0],
                [1, 2, 2, 2, 2, 0],
                [1, 2, 2, 2, 2, 0],
                [1, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # Padding 1 with a non-zero offset.
        (
            [(4, 4, 2)],
            1,
            10,
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 12, 12, 0],
                [0, 0, 0, 12, 12, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ),
        # Padding 2 at the far corner clips at the bottom and right edges.
        (
            [(5, 5, 1)],
            2,
            0,
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ],
        ),
    ],
    ids=[
        "no-padding",
        "padding-0-offset",
        "padding-2",
        "clip-at-row-0",
        "overlap-single-pixel",
        "overlap-padded",
        "padding-1-offset",
        "far-edge-clip",
    ],
)
def test_rasterize_points_paints_literal_mask(
    points: list[tuple[int, int, int]],
    padding: int | None,
    offset: int,
    expected: list[list[int]],
) -> None:
    rows, cols, ids = (np.array(axis, dtype=np.int64) for axis in zip(*points, strict=True))

    mask = rasterize_points(rows, cols, ids, (6, 6), padding, offset)

    assert mask.dtype == np.int32
    assert mask.tolist() == expected


def test_rasterize_points_returns_zeros_for_no_points() -> None:
    empty = np.array([], dtype=np.int64)

    mask = rasterize_points(empty, empty, empty, (6, 6), 2, 0)

    assert mask.dtype == np.int32
    assert mask.tolist() == [[0] * 6] * 6
