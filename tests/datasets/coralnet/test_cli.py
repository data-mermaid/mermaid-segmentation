"""Smoke tests for the ``coralnet-etl`` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from mermaidseg.datasets.coralnet.etl import __main__ as cli
from mermaidseg.datasets.coralnet.etl.schemas import (
    ANNOTATIONS_SCHEMA,
    AUDIT_SCHEMA,
    IMAGES_SCHEMA,
)


def test_build_parser_has_all_subcommands():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    assert actions, "expected a subparsers action"
    sub = actions[0]
    assert set(sub.choices.keys()) == {"audit", "build-annotations", "build-images", "all"}


def _patch_to_fake_s3(monkeypatch, fake_s3):
    """Redirect every S3/ibis factory in the ETL package at the in-memory FakeS3.

    The production code paths build a boto3 client (for paginator / head_object / images-folder
    counts) and an ibis DuckDB connection (for CSV reads). For tests we replace both with FakeS3 by
    patching the factory functions in every module that imports them, plus the ibis-reader factory
    which we redirect to a boto3-backed reader against FakeS3.
    """
    from mermaidseg.datasets.coralnet.etl.io import make_csv_reader_s3

    fake_reader_factory = lambda _con=None: make_csv_reader_s3(fake_s3)  # noqa: E731

    for mod in (
        cli,
        "mermaidseg.datasets.coralnet.etl.audit",
        "mermaidseg.datasets.coralnet.etl.annotations",
        "mermaidseg.datasets.coralnet.etl.images",
    ):
        monkeypatch.setattr(
            f"{mod if isinstance(mod, str) else mod.__name__}.make_s3_client",
            lambda: fake_s3,
            raising=False,
        )

    for mod in (
        "mermaidseg.datasets.coralnet.etl.audit",
        "mermaidseg.datasets.coralnet.etl.annotations",
    ):
        monkeypatch.setattr(f"{mod}.make_ibis_connection", lambda: object(), raising=False)
        monkeypatch.setattr(f"{mod}.make_csv_reader_ibis", fake_reader_factory, raising=False)

    monkeypatch.setattr(
        "mermaidseg.datasets.coralnet.etl.images._get_thread_client",
        lambda: fake_s3,
        raising=False,
    )


def test_audit_source_ids_exclusive():
    ns = argparse.Namespace(source_ids="1,2", source_ids_file=Path("x.parquet"))
    with pytest.raises(SystemExit):
        cli._audit_source_ids_from_args(ns)


def test_cli_audit_explicit_source_ids(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path
):
    bucket, prefix = "b", "coralnet-public-images"
    for sid in (1, 3):
        populate(
            fake_s3,
            bucket,
            prefix,
            sid,
            sample_csv_factory(source_id=sid, n_images=1, n_points_per_image=1),
        )

    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        [
            "audit",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--source-ids",
            "1,3",
            "--output-dir",
            str(tmp_path),
            "--workers",
            "1",
            "--version-tag",
            "subs",
        ]
    )
    assert rc == 0
    audit_parquet = Path(tmp_path) / "subs" / "coralnet_audit_subs.parquet"
    assert audit_parquet.exists()
    adf = pd.read_parquet(audit_parquet)
    assert sorted(adf["source_id"].astype(int).tolist()) == [1, 3]


def test_cli_audit_source_ids_file(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path):
    csv_path = tmp_path / "ids.csv"
    pd.DataFrame({"source_id": [2]}).to_csv(csv_path, index=False)

    bucket, prefix = "b", "coralnet-public-images"
    populate(
        fake_s3,
        bucket,
        prefix,
        2,
        sample_csv_factory(source_id=2, n_images=1, n_points_per_image=2),
    )
    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        [
            "audit",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--source-ids-file",
            str(csv_path),
            "--output-dir",
            str(tmp_path),
            "--workers",
            "1",
            "--version-tag",
            "from_csv",
        ]
    )
    assert rc == 0
    adf = pd.read_parquet(Path(tmp_path) / "from_csv" / "coralnet_audit_from_csv.parquet")
    assert adf["source_id"].tolist() == [2]


def test_cli_audit_round_trips(monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path):
    bucket, prefix = "b", "coralnet-public-images"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=2),
    )

    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        [
            "audit",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--output-dir",
            str(tmp_path),
            "--workers",
            "1",
            "--version-tag",
            "test",
        ]
    )
    assert rc == 0

    candidates = list(Path(tmp_path).rglob("coralnet_audit_test.parquet"))
    assert candidates, "audit parquet not written"
    schema = pq.read_schema(candidates[0])
    assert {f.name for f in schema} == {f.name for f in AUDIT_SCHEMA}


def test_cli_all_pipeline_end_to_end(
    monkeypatch, fake_s3, sample_csv_factory, populate, tmp_path, make_jpeg
):
    bucket, prefix = "b", "coralnet-public-images"
    populate(
        fake_s3,
        bucket,
        prefix,
        1,
        sample_csv_factory(source_id=1, n_images=2, n_points_per_image=2),
    )
    big = make_jpeg(4096, 3000)
    small = make_jpeg(800, 600)
    fake_s3.put(bucket, f"{prefix}/s1/images/10000.jpg", big)
    fake_s3.put(bucket, f"{prefix}/s1/images/10001.jpg", small)

    _patch_to_fake_s3(monkeypatch, fake_s3)

    rc = cli.main(
        [
            "all",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--output-dir",
            str(tmp_path),
            "--workers",
            "1",  # serial → forces s3_client injection path
            "--version-tag",
            "v1",
        ]
    )
    assert rc == 0

    out_dir = Path(tmp_path) / "v1"
    audit_path = out_dir / "coralnet_audit_v1.parquet"
    ann_path = out_dir / "coralnet_annotations_v1.parquet"
    img_path = out_dir / "coralnet_images_v1.parquet"
    assert audit_path.exists()
    assert ann_path.exists()
    assert img_path.exists()

    audit_df = pd.read_parquet(audit_path)
    ann_df = pd.read_parquet(ann_path)
    img_df = pd.read_parquet(img_path)

    assert int(audit_df["source_id"].iloc[0]) == 1
    assert audit_df["is_complete"].iloc[0]
    assert len(ann_df) == 4  # 2 imgs * 2 pts
    assert {f.name for f in ANNOTATIONS_SCHEMA}.issubset(set(ann_df.columns))
    assert {f.name for f in IMAGES_SCHEMA}.issubset(set(img_df.columns))
    assert len(img_df) == 2
    assert (img_df["header_status"] == "ok").all()
    assert img_df.loc[img_df["image_id"] == "10000", "needs_resize"].iloc[0]
    assert not img_df.loc[img_df["image_id"] == "10001", "needs_resize"].iloc[0]


@pytest.mark.parametrize(
    "subcmd,required",
    [
        ("build-annotations", "--audit"),
        ("build-images", "--annotations"),
    ],
)
def test_required_args_enforced(subcmd, required):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([subcmd])
