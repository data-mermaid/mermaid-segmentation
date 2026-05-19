"""Tests for the annotations merge stage."""

from __future__ import annotations

import pandas as pd

from mermaidseg.datasets.coralnet.etl.annotations import (
    _derive_image_id,
    build_annotations,
)
from mermaidseg.datasets.coralnet.etl.audit import audit_sources


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

    audit_df = audit_sources(bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    ann_df = build_annotations(audit_df, bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)

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
    audit = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    a = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    b = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert a.equals(b)
    keys = list(zip(a["source_id"], a["image_id"], a["row"], a["col"], strict=False))
    assert keys == sorted(keys)


def test_build_annotations_strips_all_status_suffixes(fake_s3, populate):
    """Image_list Names can carry ' - Confirmed', ' - Unconfirmed', or ' - Unclassified'."""
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

    audit = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    ann = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert len(ann) == 3, "all three rows must survive the merge regardless of status suffix"
    assert set(ann["image_id"]) == {"1", "2", "3"}
    # annotations.csv has Status, so it wins precedence — but suffix capture
    # still has to work even when only one side carries the info.
    status_by_image = dict(zip(ann["image_id"], ann["status"], strict=False))
    assert status_by_image == {"1": "Confirmed", "2": "Unconfirmed", "3": "Unclassified"}


def test_build_annotations_falls_back_to_image_list_status_suffix(fake_s3, populate):
    """Older CoralNet exports lack a Status column; status comes from image_list Name suffix."""
    annotations = "Name,Row,Column,Label ID\nfoo.jpg,1,2,100\n"
    image_list = "Name,Image Page\nfoo.jpg - Unconfirmed,/image/42/view/\n"
    files = {"annotations.csv": annotations.encode(), "image_list.csv": image_list.encode()}
    populate(fake_s3, "b", "p", 50, files)

    audit = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    ann = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert len(ann) == 1
    assert ann["status"].iloc[0] == "Unconfirmed"


def test_build_annotations_skips_incomplete_sources(fake_s3, sample_csv_factory, populate):
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
    audit = audit_sources(bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    ann = build_annotations(audit, bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    assert set(ann["source_id"].unique()) == {1}
