"""Tests for the images-metadata pipeline + deterministic parquet writes."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from coralnet_dataset.images import build_images
from coralnet_dataset.io import write_parquet_deterministic
from coralnet_dataset.schemas import (
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
)


def _seed_images(
    fake_s3, *, bucket: str, prefix: str, jpeg_small_bytes, jpeg_large_bytes
) -> pd.DataFrame:
    rows = [
        (1, "100"),
        (1, "101"),
        (1, "102"),
        (2, "200"),
        (2, "201"),
    ]
    for sid, iid in rows:
        body = jpeg_large_bytes if iid in {"100", "200"} else jpeg_small_bytes
        fake_s3.put(bucket, f"{prefix}/s{sid}/images/{iid}.jpg", body)
    return pd.DataFrame(
        {
            "source_id": [sid for sid, _ in rows],
            "image_id": [iid for _, iid in rows],
            "row": [0] * len(rows),
            "col": [0] * len(rows),
            "coralnet_id": [1] * len(rows),
            "status": ["Confirmed"] * len(rows),
        }
    )


def test_build_images_records_dimensions_and_resize_flag(
    fake_s3, jpeg_small_bytes, jpeg_large_bytes
):
    ann_df = _seed_images(
        fake_s3,
        bucket="b",
        prefix="p",
        jpeg_small_bytes=jpeg_small_bytes,
        jpeg_large_bytes=jpeg_large_bytes,
    )
    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert len(img_df) == 5
    assert set(img_df.columns) >= {field.name for field in IMAGES_SCHEMA}
    assert (img_df["header_status"] == "ok").all()
    longest_edges = dict(zip(img_df["image_id"], img_df["longest_edge"], strict=False))
    assert longest_edges["100"] == 4096
    assert longest_edges["101"] == 1024
    assert img_df.loc[img_df["image_id"] == "100", "needs_resize"].iloc[0]
    assert not img_df.loc[img_df["image_id"] == "101", "needs_resize"].iloc[0]


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
