"""Tests for the annotations merge stage."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
from coralnet_dataset.annotations import (
    _derive_image_id,
    build_annotations,
    list_source_folders,
)
from coralnet_dataset.io import write_parquet_deterministic
from coralnet_dataset.schemas import ANNOTATIONS_PRIMARY_KEY, ANNOTATIONS_SCHEMA


def test_derive_image_id_strips_url_decorations():
    s = pd.Series(
        ["/image/12345/view/", "/image/9/view/", "https://coralnet.ucsd.edu/image/77/view/"]
    )
    out = _derive_image_id(s).tolist()
    assert out == ["12345", "9", "77"]


def test_build_annotations_merges_per_source(fake_s3, sample_csv_factory, populate):
    bucket, prefix = "b", "p"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=3),
    )
    populate(
        fake_s3,
        bucket,
        prefix,
        2,
        sample_csv_factory(source_id=2, n_images=1, n_points_per_image=4),
    )

    ann_df = build_annotations([1, 2], bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)

    # 2 imgs * 3 pts + 1 img * 4 pts
    assert len(ann_df) == 10
    assert set(ann_df.columns) == {"source_id", "image_id", "row", "col", "coralnet_id", "status"}
    # Status carried through (Confirmed in the fixture).
    assert (ann_df["status"] == "Confirmed").all()

    # image_id matches the URL-derived integer-as-string for each source.
    assert "10000" in set(ann_df.loc[ann_df["source_id"] == 1, "image_id"])
    assert "20000" in set(ann_df.loc[ann_df["source_id"] == 2, "image_id"])


def test_build_annotations_sorts_deterministically(fake_s3, sample_csv_factory, populate):
    populate(
        fake_s3, "b", "p", 5, sample_csv_factory(source_id=5, n_images=2, n_points_per_image=2)
    )
    a = build_annotations([5], bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    b = build_annotations([5], bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert a.equals(b)
    keys = list(zip(a["source_id"], a["image_id"], a["row"], a["col"], strict=False))
    assert keys == sorted(keys)


def test_build_annotations_strips_all_status_suffixes(fake_s3, populate):
    """Only Confirmed rows are kept in the merged parquet."""
    annotations = (
        "Name,Row,Column,Label ID,Status\n"
        "a.jpg,1,2,100,Confirmed\n"
        "b.jpg,3,4,101,Unconfirmed\n"
        "c.jpg,5,6,102,Unclassified\n"
    )
    image_list = (
        "Name,Image Page\n"
        "a.jpg - Confirmed,/image/1/view/\n"
        "b.jpg - Unconfirmed,/image/2/view/\n"
        "c.jpg - Unclassified,/image/3/view/\n"
    )
    files = {"annotations.csv": annotations.encode(), "image_list.csv": image_list.encode()}
    populate(fake_s3, "b", "p", 99, files)

    ann = build_annotations([99], bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert len(ann) == 1
    assert ann["image_id"].iloc[0] == "1"
    assert ann["status"].iloc[0] == "Confirmed"


def test_build_annotations_drops_unconfirmed_when_status_from_image_list_suffix(fake_s3, populate):
    """Status from image_list suffix: only Confirmed rows are kept."""
    annotations = "Name,Row,Column,Label ID\nok.jpg,1,2,100\nbad.jpg,3,4,101\n"
    image_list = (
        "Name,Image Page\n"
        "ok.jpg - Confirmed,/image/42/view/\n"
        "bad.jpg - Unconfirmed,/image/43/view/\n"
    )
    files = {"annotations.csv": annotations.encode(), "image_list.csv": image_list.encode()}
    populate(fake_s3, "b", "p", 50, files)

    ann = build_annotations([50], bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert len(ann) == 1
    assert ann["image_id"].iloc[0] == "42"
    assert ann["status"].iloc[0] == "Confirmed"


def test_build_annotations_skips_sources_missing_a_csv(fake_s3, sample_csv_factory, populate):
    bucket, prefix = "b", "p"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1),
    )
    files2 = sample_csv_factory(source_id=2)
    files2.pop("image_list.csv")
    populate(fake_s3, bucket, prefix, 2, files2)
    ann = build_annotations([1, 2], bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    assert set(ann["source_id"].unique()) == {1}


def test_list_source_folders_keeps_only_numeric_source_folders(fake_s3):
    bucket, prefix = "b", "coralnet-public-images"
    for folder in ("s12", "s3", "temporary", "resized", "sNotANumber"):
        fake_s3.put(bucket, f"{prefix}/{folder}/annotations.csv", b"a,b\n1,2\n")
    assert list_source_folders(fake_s3, bucket, prefix) == [3, 12]


def test_build_annotations_without_producing_sources_writes_empty_parquet(fake_s3, tmp_path):
    ann = build_annotations([404], bucket="b", prefix="p", workers=1, s3_client=fake_s3)

    assert ann.empty
    assert list(ann.columns) == [f.name for f in ANNOTATIONS_SCHEMA]
    out = tmp_path / "ann.parquet"
    write_parquet_deterministic(ann, out, ANNOTATIONS_SCHEMA, ANNOTATIONS_PRIMARY_KEY)
    assert pq.read_schema(out).names == [f.name for f in ANNOTATIONS_SCHEMA]
