"""Unit tests for the CoralscapesV2 probe-point builder.

The builder script is not a package, so it is loaded by path. These tests use
synthetic masks only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_coralscapes_v2_probe_points.py"
_spec = importlib.util.spec_from_file_location("build_coralscapes_v2_probe_points", _SCRIPT)
assert _spec is not None and _spec.loader is not None
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _component(image_index: int, component_index: int, size: int, stem: str, site: str = "site"):
    return builder.Component(
        image_index=image_index,
        component_index=component_index,
        size=size,
        site=site,
        stem=stem,
    )


def test_assign_quotas_splits_remainder_onto_the_largest():
    # Caller orders components largest-first. 4 components, K=10 -> 3,3,2,2.
    assert builder.assign_quotas(4, 10) == [3, 3, 2, 2]
    assert builder.assign_quotas(1, 10) == [10]
    assert builder.assign_quotas(3, 10) == [4, 3, 3]
    assert sum(builder.assign_quotas(7, 10)) == 10


def test_assign_quotas_rejects_full_cover():
    with pytest.raises(ValueError, match="only for N < per_class"):
        builder.assign_quotas(10, 10)


def test_round_robin_spreads_across_images_largest_first():
    components = [
        _component(0, 1, 100, "a"),
        _component(0, 2, 50, "a"),
        _component(1, 1, 80, "b"),
        _component(2, 1, 70, "c"),
        _component(2, 2, 60, "c"),
        _component(2, 3, 40, "c"),
    ]
    picked = builder.select_components_round_robin(components, per_class=4)
    assert [component.size for component in picked] == [100, 80, 70, 50]

    picked5 = builder.select_components_round_robin(components, per_class=5)
    assert [component.size for component in picked5] == [100, 80, 70, 50, 60]


def test_select_class_assignments_one_point_when_enough_components():
    components = [_component(i, 1, 100 - i, f"s{i}") for i in range(12)]
    assignments = builder.select_class_assignments(components, per_class=10)
    assert len(assignments) == 10
    assert [quota for _component, quota in assignments] == [1] * 10
    assert len({component.image_index for component, _quota in assignments}) == 10


def test_select_class_assignments_even_split_when_fewer_components():
    components = [
        _component(0, 1, 10, "a"),
        _component(1, 2, 40, "b"),
        _component(2, 3, 30, "c"),
        _component(3, 4, 20, "d"),
    ]
    assignments = builder.select_class_assignments(components, per_class=10)
    # Largest first: 40, 30, 20, 10 with quotas 3, 3, 2, 2.
    assert [(component.size, quota) for component, quota in assignments] == [
        (40, 3),
        (30, 3),
        (20, 2),
        (10, 2),
    ]


def test_select_class_assignments_raises_when_no_components():
    with pytest.raises(builder.ProbeSelectionError, match="no connected components"):
        builder.select_class_assignments([], per_class=10)


def test_exclude_disk_removes_radius_16_inclusive():
    mask = np.ones((40, 40), dtype=bool)
    builder.exclude_disk(mask, row=20, col=20, radius=16)
    assert mask[20, 20] is np.False_
    assert mask[20, 20 + 16] is np.False_
    assert mask[20, 20 + 17] is np.True_


def test_pick_points_quota_one_is_the_distance_transform_maximum():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    points = builder.pick_points_in_component(mask, quota=1, exclusion_radius=16)
    assert points == [builder.interior_point(mask)]


def test_pick_points_quota_two_are_separated_by_the_disk():
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    points = builder.pick_points_in_component(mask, quota=2, exclusion_radius=16)
    assert len(points) == 2
    (row0, col0), (row1, col1) = points
    distance_sq = (row0 - row1) ** 2 + (col0 - col1) ** 2
    assert distance_sq > 16 * 16
    assert mask[row0, col0] and mask[row1, col1]


def test_pick_points_keeps_one_pixel_when_disk_does_not_fit():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    points = builder.pick_points_in_component(mask, quota=2, exclusion_radius=16)
    assert len(points) == 1
    row, col = points[0]
    assert mask[row, col]


def test_points_for_image_records_a_shortfall_and_keeps_the_pixel():
    label = np.zeros((40, 40), dtype=np.uint8)
    label[10:20, 10:20] = 75
    component = builder.Component(
        image_index=0,
        component_index=1,
        size=100,
        site="site30",
        stem="site30_000148_000236",
    )
    points, errors, shortfalls = builder._points_for_image(
        label, [(75, component, 3)], exclusion_radius=16
    )
    assert errors == []
    assert len(points) == 1
    assert points[0]["class_id"] == 75
    assert points[0]["component_quota"] == 3
    assert points[0]["point_index_in_component"] == 0
    assert shortfalls == [
        {
            "class_id": 75,
            "class_name": "seriatopora dead",
            "site": "site30",
            "stem": "site30_000148_000236",
            "component_index": 1,
            "component_size": 100,
            "quota": 3,
            "placed": 1,
        }
    ]


def test_pick_points_raises_when_the_component_has_fewer_pixels_than_the_quota():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    with pytest.raises(builder.ProbeSelectionError, match="fewer than quota"):
        builder.pick_points_in_component(mask, quota=5, exclusion_radius=16)
