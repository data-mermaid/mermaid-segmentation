"""Curation order (overrides, then blacklist) and the derived tables."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from segdata_etl.datasets.coralnet.curate.build import curate
from segdata_etl.datasets.coralnet.curate.overrides import OverrideStats
from segdata_spec.schema import IMAGES_SCHEMA, LABEL_COUNTS_SCHEMA, POINTS_SCHEMA

BIG = "1924169"  # 3486 x 3469, resized on load
SMALL = "42"
BLACK = "77"
GONE = "99"  # in the override file only


def _image_row(image_id: str, width: int, height: int) -> dict[str, object]:
    return {
        "image_id": image_id,
        "key": f"s1/{image_id}.jpg",
        "etag": "e",
        "size": 1,
        "crc64nvme": None,
        "media_type": "image/jpeg",
        "width": width,
        "height": height,
        "group": "s1",
        "mask_key": None,
        "mask_etag": None,
        "mask_size": None,
        "mask_crc64nvme": None,
        "source_fingerprint": "f",
        "source_name": f"{image_id}.jpg",
        "attrs": None,
    }


@pytest.fixture
def raw(tmp_path):
    images = pa.Table.from_pylist(
        [_image_row(BIG, 3486, 3469), _image_row(SMALL, 640, 480), _image_row(BLACK, 640, 480)],
        schema=IMAGES_SCHEMA,
    )
    points = pa.Table.from_pylist(
        [
            {"image_id": BIG, "row": 96, "col": 3158, "label_id": 141, "annotator": "default"},
            {"image_id": BIG, "row": 200, "col": 200, "label_id": 7, "annotator": "default"},
            {"image_id": SMALL, "row": 1, "col": 2, "label_id": 5, "annotator": "default"},
            {"image_id": SMALL, "row": 3, "col": 4, "label_id": 5, "annotator": "default"},
            {"image_id": SMALL, "row": 5, "col": 6, "label_id": 6, "annotator": "default"},
            {"image_id": BLACK, "row": 1, "col": 1, "label_id": 8, "annotator": "default"},
        ],
        schema=POINTS_SCHEMA,
    )
    overrides = pd.DataFrame(
        {
            "image_id": [BIG, BLACK, GONE],
            "row": pd.array([56, 10, 1], dtype="int16"),
            "col": pd.array([1854, 10, 1], dtype="int16"),
            "coralnet_id": [141, 9, 5],
            "extra": ["x", "y", "z"],
        }
    )
    blacklist = pd.DataFrame({"image_id": [int(BLACK), 12345], "reason": ["a", "b"]})
    overrides.to_parquet(tmp_path / "overrides.parquet")
    blacklist.to_parquet(tmp_path / "blacklist.parquet")
    version = SimpleNamespace(images=images, points=points)
    return version, tmp_path / "blacklist.parquet", tmp_path / "overrides.parquet"


def test_curate_applies_overrides_then_blacklist_and_derives_label_counts(raw) -> None:
    version, blacklist, overrides = raw
    out = curate(version, image_blacklist=blacklist, annotation_overrides=overrides)

    assert out.images.schema == IMAGES_SCHEMA
    assert out.images.to_pylist() == [
        r for r in version.images.to_pylist() if r["image_id"] != BLACK
    ]
    assert out.points.schema == POINTS_SCHEMA
    assert sorted(
        (r["image_id"], r["row"], r["col"], r["label_id"]) for r in out.points.to_pylist()
    ) == [(BIG, 96, 3158, 141), (SMALL, 1, 2, 5), (SMALL, 3, 4, 5), (SMALL, 5, 6, 6)]
    assert out.label_counts.schema == LABEL_COUNTS_SCHEMA
    assert sorted((r["image_id"], r["label_id"], r["n"]) for r in out.label_counts.to_pylist()) == [
        (BIG, 141, 1),
        (SMALL, 5, 2),
        (SMALL, 6, 1),
    ]
    assert out.dropped == {"blacklisted_images": 1, "override_images_not_in_raw": 1}
    assert out.override_stats == OverrideStats(
        images_overridden=2,
        points_exact=1,
        points_inverted=0,
        points_unscaled=1,
        override_images_not_in_raw=1,
    )
