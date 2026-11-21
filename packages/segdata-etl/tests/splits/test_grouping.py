from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from segdata_etl.splits.grouping import (
    components,
    haversine_km,
    load_labels,
    near_dup_edges,
    project_edges,
    shared_image_edges,
)


def _keys(rows: list[tuple[int, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source_id", "crc64nvme", "size"])


def test_two_sources_sharing_a_key_get_one_edge():
    keys = _keys([(1, "AAA=", 10), (2, "AAA=", 10), (3, "BBB=", 10)])
    edges, cross = shared_image_edges(keys, [1, 2, 3])
    assert edges == {(1, 2)}
    assert cross == {1, 2}


def test_same_crc_with_a_different_size_is_not_an_edge():
    keys = _keys([(1, "AAA=", 10), (2, "AAA=", 11)])
    edges, cross = shared_image_edges(keys, [1, 2])
    assert edges == set()
    assert cross == set()


def test_three_sources_sharing_one_key_form_one_component():
    keys = _keys([(5, "AAA=", 10), (3, "AAA=", 10), (9, "AAA=", 10)])
    edges, _ = shared_image_edges(keys, [3, 5, 7, 9])
    assert components([3, 5, 7, 9], edges) == [{3, 5, 9}, {7}]


@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "near_dup_labels.csv"
    path.write_text(
        "image_a,image_b,label,source_a,source_b\n"
        "a1,b1,dup,1,2\n"
        "a2,c1,unsure,1,3\n"
        "a3,d1,fp,1,4\n"
        "a4,a5,dup,1,1\n"
        "a6,z1,dup,1,99\n"
        # The later row for (b2, e1) overrides the earlier dup.
        "b2,e1,dup,2,5\n"
        "b2,e1,fp,2,5\n"
    )
    return path


def test_near_dup_dup_and_unsure_labels_link_sources(labels_csv):
    edges = near_dup_edges(load_labels(labels_csv), [1, 2, 3, 4, 5])
    assert edges == {(1, 2), (1, 3)}


def _meta(rows: list[tuple[int, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source_id", "admins", "latitude", "longitude"])


def test_shared_admin_within_the_id_gap_links_by_id():
    meta = _meta([(10, "ann", 0.0, 0.0), (40, "ann", 0.0, 90.0)])
    # Ranks 0 and 3 in the sorted universe: a gap of exactly max_id_gap.
    assert project_edges(meta, [10, 20, 30, 40], max_id_gap=3, max_km=50.0) == [(10, 40, "id")]


def test_shared_admin_beyond_the_id_gap_and_far_apart_is_not_linked():
    meta = _meta([(10, "ann", 0.0, 0.0), (50, "ann", 0.0, 90.0)])
    assert project_edges(meta, [10, 20, 30, 40, 50], max_id_gap=3, max_km=50.0) == []


def test_shared_admin_far_in_rank_but_close_in_location_links_by_location():
    # 0.1 degrees of longitude at the equator is about 11 km.
    meta = _meta([(10, "bob;ann", 0.0, 0.0), (50, "ann", 0.0, 0.1)])
    assert project_edges(meta, [10, 20, 30, 40, 50], max_id_gap=3, max_km=50.0) == [
        (10, 50, "location")
    ]


def test_nan_latitude_gives_no_location_edge():
    meta = _meta([(10, "ann", math.nan, 0.0), (50, "ann", 0.0, 0.0)])
    assert project_edges(meta, [10, 20, 30, 40, 50], max_id_gap=3, max_km=50.0) == []


def test_no_shared_admin_gives_no_edge():
    meta = _meta([(10, "ann", 0.0, 0.0), (20, "bob", 0.0, 0.0)])
    assert project_edges(meta, [10, 20], max_id_gap=3, max_km=50.0) == []


def test_haversine_one_degree_of_longitude_at_the_equator():
    # 6371 km * pi / 180 = 111.195 km.
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.001)
