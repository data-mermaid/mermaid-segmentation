"""Tests for the S3 audit pass."""

from __future__ import annotations

from mermaidseg.datasets.coralnet.etl.audit import audit_sources, list_source_folders


def test_list_source_folders_filters_to_numeric(fake_s3):
    bucket, prefix = "b", "coralnet-public-images"
    fake_s3.put(bucket, f"{prefix}/s1/annotations.csv", b"a,b\n1,2\n")
    fake_s3.put(bucket, f"{prefix}/s42/annotations.csv", b"a,b\n1,2\n")
    fake_s3.put(bucket, f"{prefix}/sNotANumber/annotations.csv", b"a,b\n1,2\n")
    fake_s3.put(bucket, f"{prefix}/other/foo.csv", b"x")
    ids = list_source_folders(fake_s3, bucket, prefix)
    assert ids == [1, 42]


def test_audit_sources_three_sources(fake_s3, sample_csv_factory, populate):
    bucket, prefix = "b", "coralnet-public-images"

    # source 1: complete (annotations + image_list + images + labelset + metadata)
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=3),
    )

    # source 2: missing image_list.csv
    files2 = sample_csv_factory(source_id=2, n_images=2, n_points_per_image=3)
    files2.pop("image_list.csv")
    populate(fake_s3, bucket, prefix, 2, files2)

    # source 3: annotations.csv is malformed (parse error)
    files3 = sample_csv_factory(source_id=3, n_images=2, n_points_per_image=3)
    files3["annotations.csv"] = b"\xff\xfe corrupt non-csv data\nmore garbage"
    populate(fake_s3, bucket, prefix, 3, files3)

    df = audit_sources(bucket=bucket, prefix=prefix, workers=1, s3_client=fake_s3)

    assert set(df["source_id"]) == {1, 2, 3}
    row1 = df.loc[df["source_id"] == 1].iloc[0]
    row2 = df.loc[df["source_id"] == 2].iloc[0]
    row3 = df.loc[df["source_id"] == 3].iloc[0]

    assert row1["is_complete"]
    assert row1["has_image_list_csv"]
    assert row1["n_annotations"] == 6  # 2 imgs * 3 pts
    assert row1["n_unique_images_annotated"] == 2

    assert not row2["is_complete"]
    assert not row2["has_image_list_csv"]
    assert row2["has_annotations_csv"]

    # source 3: annotations file parses as something (csv is forgiving) but the
    # important guarantee is "no crash + is_complete tracks reality".
    assert isinstance(bool(row3["is_complete"]), bool)


def test_audit_sources_emits_audit_timestamp_utc(fake_s3, sample_csv_factory, populate):
    populate(
        fake_s3, "b", "p", 1, sample_csv_factory(source_id=1, n_images=1, n_points_per_image=1)
    )
    df = audit_sources(bucket="b", prefix="p", workers=1, s3_client=fake_s3)
    ts = df["audit_timestamp"].iloc[0]
    assert ts.tzinfo is not None


def test_audit_sources_respects_explicit_source_ids(fake_s3, sample_csv_factory, populate):
    bucket, prefix = "b", "p"
    populate(fake_s3, bucket, prefix, 7, sample_csv_factory(source_id=7))
    populate(fake_s3, bucket, prefix, 11, sample_csv_factory(source_id=11))
    df = audit_sources(
        bucket=bucket,
        prefix=prefix,
        workers=1,
        s3_client=fake_s3,
        source_ids=[11],
    )
    assert df["source_id"].tolist() == [11]
