"""Parser / wiring tests for coralnet-remediate (no network)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from mermaidseg.datasets.coralnet.scraper.models import SourceProbe
from mermaidseg.datasets.coralnet.scraper.remediate_incomplete import (
    _join_path,
    _parent_path,
    build_parser,
    cmd_probe,
    cmd_redownload,
)


def test_s3_parent_and_join_path_helpers():
    audit = "s3://bucket/prefix/run/coralnet_audit_20260526_abc.parquet"
    parent = _parent_path(audit)
    assert parent == "s3://bucket/prefix/run"
    assert _join_path(parent, "incomplete_df.parquet") == (
        "s3://bucket/prefix/run/incomplete_df.parquet"
    )


_CREDS_TARGET = (
    "mermaidseg.datasets.coralnet.scraper.remediate_incomplete.load_coralnet_credentials"
)
_DL_TARGET = "mermaidseg.datasets.coralnet.scraper.remediate_incomplete.CoralNetDownloader"


def test_remediate_probe_writes_report(tmp_path):
    """Each source gets the action and gap its audit row and website probe imply.

    7: no S3 structure at all, site has 5 images        -> full_redownload, gap 5
    8: structure and CSVs fine, 3 of 5 images in S3     -> redownload_images, gap 2
    9: not in the audit at all, site has 4 images       -> manual_review, gap 4
    10: website not accessible                          -> skip_not_accessible, no gap
    """
    incomp = pd.DataFrame({"source_id": [10, 9, 8, 7]})
    audit = pd.DataFrame(
        {
            "source_id": [7, 8, 10],
            "has_images_folder": [False, True, True],
            "has_annotations_csv": [False, True, True],
            "has_image_list_csv": [False, True, True],
            "n_images_s3": [0, 3, 3],
            "n_images_csv": [0, 5, 5],
            "n_annotations": [0, 10, 10],
            "annotations_csv_read_failed": [False, False, False],
            "annotations_empty": [True, False, False],
            "image_count_match": [False, False, False],
        }
    )
    ipath = tmp_path / "incomplete.parquet"
    apath = tmp_path / "audit.parquet"
    rpath = tmp_path / "report.parquet"
    incomp.to_parquet(ipath)
    audit.to_parquet(apath)

    probes = {
        7: SourceProbe(7, "http://mock/7", True, None, 5, 5, 0, 0),
        8: SourceProbe(8, "http://mock/8", True, None, 5, 4, 1, 0),
        9: SourceProbe(9, "http://mock/9", True, None, 4, 4, 0, 0),
        10: SourceProbe(10, "http://mock/10", False, "403 forbidden", 0),
    }

    class _StubDL:
        def __init__(self, *_, **__) -> None:
            pass

        def login(self) -> bool:
            return True

        def probe_source(self, sid):  # noqa: ANN001
            return probes[int(sid)]

    with patch(_CREDS_TARGET, return_value=("u", "p")), patch(_DL_TARGET, _StubDL):
        ns = Namespace(
            incomplete_parquet=ipath,
            audit_parquet=apath,
            report_out=rpath,
            workers=1,
            delay_seconds=0.0,
        )
        assert cmd_probe(ns) == 0

    out = pd.read_parquet(rpath).set_index("source_id")
    assert out.index.tolist() == [7, 8, 9, 10]
    assert out["action"].to_dict() == {
        7: "full_redownload",
        8: "redownload_images",
        9: "manual_review",
        10: "skip_not_accessible",
    }
    assert out["probe_accessible"].to_dict() == {7: True, 8: True, 9: True, 10: False}
    assert out["images_gap"].iloc[:3].tolist() == [5, 2, 4]
    assert pd.isna(out.loc[10, "images_gap"])
    assert out["probe_error"].tolist() == [None, None, None, "403 forbidden"]
    assert out["n_confirmed_website"].iloc[:3].tolist() == [5, 4, 4]
    assert out["n_unconfirmed_website"].iloc[:3].tolist() == [0, 1, 0]
    assert out["source_url"].tolist() == [f"http://mock/{s}" for s in (7, 8, 9, 10)]


def test_remediate_verbose_flag_after_subcommand():
    args = build_parser().parse_args(
        ["redownload", "--report", "report.parquet", "--dry-run", "-v"]
    )
    assert args.command == "redownload"
    assert args.verbose is True
    assert args.dry_run is True


def test_remediate_redownload_dry_run(tmp_path, caplog):
    rep = pd.DataFrame(
        {
            "source_id": [1288, 307],
            "action": ["full_redownload", "redownload_csv"],
            "total_images_website": [31381, 5108],
            "images_gap": [31381, 0],
        }
    )
    rp = Path(tmp_path) / "report.parquet"
    rep.to_parquet(rp)

    mock_dl_cls = MagicMock()
    mock_inst = MagicMock()
    mock_inst.login.return_value = True
    mock_dl_cls.return_value = mock_inst

    ns = Namespace(
        report=rp,
        bucket="b",
        prefix="pr",
        dry_run=True,
        clean_prefix=False,
        scraper_workers=None,
    )
    with (
        patch(_CREDS_TARGET, return_value=("u", "p")),
        patch(_DL_TARGET, mock_dl_cls),
        caplog.at_level("INFO"),
    ):
        rc = cmd_redownload(ns)

    assert rc == 0
    mock_inst.download_source.assert_not_called()
    assert "Redownload queue: 2 source(s)" in caplog.text
    assert "Redownload 1/2: source 307" in caplog.text
    assert "Redownload 2/2: source 1288" in caplog.text


def test_cmd_probe_requires_source_id_column(tmp_path):
    incomp = pd.DataFrame({"wrong": [1]})
    ipath = tmp_path / "inc.parquet"
    incomp.to_parquet(ipath)
    audit = pd.DataFrame({"source_id": [100]})
    apath = tmp_path / "aud.parquet"
    audit.to_parquet(apath)
    ns = Namespace(
        incomplete_parquet=ipath,
        audit_parquet=apath,
        report_out=tmp_path / "out.parquet",
        workers=1,
        delay_seconds=0.0,
    )
    assert cmd_probe(ns) == 2
