"""Tests for the images-metadata pipeline + deterministic parquet writes."""

from __future__ import annotations

import threading

import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from segdata_etl.datasets.coralnet import images
from segdata_etl.datasets.coralnet.images import build_images
from segdata_etl.datasets.coralnet.io import write_parquet_deterministic
from segdata_etl.datasets.coralnet.schemas import (
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)


def test_build_images_handles_missing_objects(fake_s3, jpeg_small_bytes):
    ann_df = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["present", "missing"],
            "row": [0, 0],
            "col": [0, 0],
            "coralnet_id": [1, 1],
            "status": ["Confirmed", "Confirmed"],
        }
    )
    fake_s3.put("b", "p/s1/images/present.jpg", jpeg_small_bytes)
    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    statuses = dict(zip(img_df["image_id"], img_df["header_status"], strict=False))
    assert statuses["present"] == "ok"
    assert statuses["missing"] == "not_found"


def test_build_images_on_empty_annotations_writes_empty_parquet(fake_s3, tmp_path):
    ann_df = pd.DataFrame(
        {"source_id": pd.Series(dtype="int32"), "image_id": pd.Series(dtype="string")}
    )

    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert img_df.empty
    assert list(img_df.columns) == [f.name for f in IMAGES_SCHEMA]
    out = tmp_path / "images.parquet"
    write_parquet_deterministic(img_df, out, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    assert pq.read_schema(out).names == [f.name for f in IMAGES_SCHEMA]


def test_build_images_sequential_path_records_worker_exception(
    fake_s3, jpeg_small_bytes, monkeypatch
):
    ann_df = pd.DataFrame(
        {
            "source_id": [1, 1],
            "image_id": ["good", "bomb"],
            "row": [0, 0],
            "col": [0, 0],
            "coralnet_id": [1, 1],
            "status": ["Confirmed", "Confirmed"],
        }
    )
    fake_s3.put("b", "p/s1/images/good.jpg", jpeg_small_bytes)
    fake_s3.put("b", "p/s1/images/bomb.jpg", jpeg_small_bytes)
    real_get_object = fake_s3.get_object

    def get_object(*, Bucket, Key, Range=None):
        if Key.endswith("bomb.jpg"):
            raise Image.DecompressionBombError("too big")
        return real_get_object(Bucket=Bucket, Key=Key, Range=Range)

    monkeypatch.setattr(fake_s3, "get_object", get_object)

    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    by_id = img_df.set_index("image_id")
    assert len(img_df) == 2
    assert by_id.loc["bomb", "header_status"] == "worker_exception"
    assert by_id.loc["bomb", "error_message"] == "DecompressionBombError: too big"
    assert pd.isna(by_id.loc["bomb", "width"])
    assert pd.isna(by_id.loc["bomb", "height"])
    assert by_id.loc["good", "header_status"] == "ok"
    assert (by_id.loc["good", "width"], by_id.loc["good", "height"]) == (1024, 768)


@pytest.mark.parametrize(
    ("width", "height", "longest_edge", "needs_resize"),
    [
        (2048, 1536, 2048, False),
        (2049, 1536, 2049, True),
        (1536, 2049, 2049, True),
    ],
    ids=["landscape-at-2048", "landscape-2049", "portrait-2049"],
)
def test_build_images_longest_edge_and_resize_boundary(
    fake_s3, make_jpeg, width, height, longest_edge, needs_resize
):
    fake_s3.put("b", "p/s1/images/img.jpg", make_jpeg(width, height))
    ann_df = pd.DataFrame({"source_id": [1], "image_id": ["img"]})

    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    row = img_df.iloc[0]
    assert (row["width"], row["height"]) == (width, height)
    assert row["longest_edge"] == longest_edge
    assert bool(row["needs_resize"]) is needs_resize


class _ThreadRecordingS3:
    """Wraps a FakeS3, records the calling thread and fails for keys ending in ``bomb.jpg``."""

    def __init__(self, fake_s3):
        self._fake = fake_s3
        self.thread_ids: set[int] = set()

    def get_object(self, *, Bucket, Key, Range=None):
        self.thread_ids.add(threading.get_ident())
        if Key.endswith("bomb.jpg"):
            raise Image.DecompressionBombError("too big")
        return self._fake.get_object(Bucket=Bucket, Key=Key, Range=Range)


def test_build_images_threaded_path_records_worker_exception(fake_s3, make_jpeg, monkeypatch):
    ids = ["a", "b", "bomb", "c"]
    for image_id in ids:
        fake_s3.put("b", f"p/s1/images/{image_id}.jpg", make_jpeg(1024, 768))
    client = _ThreadRecordingS3(fake_s3)
    monkeypatch.setattr(images, "_get_thread_client", lambda: client)
    ann_df = pd.DataFrame({"source_id": [1] * 4, "image_id": ids})

    img_df = build_images(ann_df, bucket="b", prefix="p", workers=3)

    assert threading.get_ident() not in client.thread_ids
    assert img_df["image_id"].tolist() == ["a", "b", "bomb", "c"]
    assert img_df["header_status"].tolist() == ["ok", "ok", "worker_exception", "ok"]
    assert img_df["error_message"].tolist() == [
        pd.NA,
        pd.NA,
        "DecompressionBombError: too big",
        pd.NA,
    ]
    assert img_df["width"].tolist() == [1024, 1024, pd.NA, 1024]
