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


def test_build_annotations_sorts_deterministically(fake_s3, populate):
    """Rows come back ordered by (source_id, image_id, row, col), not in file or audit order."""
    # Source 5: image 30 listed before image 4 (string order "30" < "4"), points out of order.
    populate(
        fake_s3,
        "b",
        "p",
        5,
        {
            "annotations.csv": (
                b"Name,Row,Column,Label ID,Status\n"
                b"z.jpg,9,1,100,Confirmed\n"
                b"z.jpg,2,7,100,Confirmed\n"
                b"z.jpg,2,3,100,Confirmed\n"
                b"y.jpg,5,5,100,Confirmed\n"
            ),
            "image_list.csv": (b"Name,Image Page\nz.jpg,/image/30/view/\ny.jpg,/image/4/view/\n"),
        },
    )
    populate(
        fake_s3,
        "b",
        "p",
        2,
        {
            "annotations.csv": b"Name,Row,Column,Label ID,Status\nq.jpg,1,1,100,Confirmed\n",
            "image_list.csv": b"Name,Image Page\nq.jpg,/image/8/view/\n",
        },
    )
    audit = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    # Feed the audit in reverse so the sources are read 5 then 2.
    audit = audit.iloc[::-1].reset_index(drop=True)
    assert audit["source_id"].tolist() == [5, 2]
    out = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert out[["source_id", "image_id", "row", "col"]].to_numpy().tolist() == [
        [2, "8", 1, 1],
        [5, "30", 2, 3],
        [5, "30", 2, 7],
        [5, "30", 9, 1],
        [5, "4", 5, 5],
    ]


def test_build_annotations_strips_all_status_suffixes(fake_s3, populate):
    """Every status suffix is stripped from image_list names so they join to annotations.csv.

    annotations.csv carries its own Status (which wins over the suffix), so a row survives only if
    its bare name matched the stripped image_list name.
    """
    annotations = (
        "Name,Row,Column,Label ID,Status\n"
        "a.jpg,1,2,100,Confirmed\n"
        "b.jpg,3,4,101,Confirmed\n"
        "c.jpg,5,6,102,Confirmed\n"
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
    assert ann["image_id"].tolist() == ["1", "2", "3"]
    assert ann["coralnet_id"].tolist() == [100, 101, 102]
    assert (ann["status"] == "Confirmed").all()


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

    audit = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    ann = build_annotations(audit, bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    assert len(ann) == 1
    assert ann["image_id"].iloc[0] == "42"
    assert ann["status"].iloc[0] == "Confirmed"


def test_build_annotations_skips_incomplete_sources(fake_s3, sample_csv_factory, populate):
    """Source 2 loads fine (rows, images, matching names) but has no images folder."""
    bucket, prefix = "b", "p"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1),
    )
    populate(
        fake_s3,
        bucket,
        prefix,
        2,
        sample_csv_factory(source_id=2, n_images=1, n_points_per_image=1),
    )
    fake_s3.delete_object(Bucket=bucket, Key=f"{prefix}/s2/images/img_marker.jpg")
    audit = audit_sources(bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    assert audit.set_index("source_id")["is_complete"].to_dict() == {1: True, 2: False}
    ann = build_annotations(audit, bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)
    assert ann["source_id"].tolist() == [1]
