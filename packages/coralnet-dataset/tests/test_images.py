"""Tests for the images-metadata pipeline + deterministic parquet writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mermaidseg.datasets.coralnet.etl.images import _next_part_index, build_images
from mermaidseg.datasets.coralnet.etl.io import write_parquet_deterministic
from mermaidseg.datasets.coralnet.etl.schemas import (
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


def test_build_images_deterministic_bytes(tmp_path, fake_s3, jpeg_small_bytes, jpeg_large_bytes):
    ann_df = _seed_images(
        fake_s3,
        bucket="b",
        prefix="p",
        jpeg_small_bytes=jpeg_small_bytes,
        jpeg_large_bytes=jpeg_large_bytes,
    )
    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    write_parquet_deterministic(img_df, p1, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)
    write_parquet_deterministic(img_df, p2, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)

    h1 = hashlib.sha256(Path(p1).read_bytes()).hexdigest()
    h2 = hashlib.sha256(Path(p2).read_bytes()).hexdigest()
    assert h1 == h2


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


def test_build_images_round_trips_through_parquet(
    tmp_path, fake_s3, jpeg_small_bytes, jpeg_large_bytes
):
    ann_df = _seed_images(
        fake_s3,
        bucket="b",
        prefix="p",
        jpeg_small_bytes=jpeg_small_bytes,
        jpeg_large_bytes=jpeg_large_bytes,
    )
    img_df = build_images(ann_df, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    out = tmp_path / "images.parquet"
    write_parquet_deterministic(img_df, out, IMAGES_SCHEMA, IMAGES_PRIMARY_KEY)

    schema = pq.read_schema(out)
    assert {f.name for f in schema} == {f.name for f in IMAGES_SCHEMA}
    round = pq.read_table(out).to_pandas()
    assert len(round) == len(img_df)


def test_next_part_index_uses_filename_max_not_loaded_count(tmp_path):
    cp = tmp_path / "images.parquet"
    (tmp_path / "images.part_00000.parquet").write_bytes(b"not parquet")
    (tmp_path / "images.part_00001.parquet").write_bytes(b"not parquet")
    assert _next_part_index(cp) == 2
