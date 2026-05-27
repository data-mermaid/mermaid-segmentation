"""Parser / wiring tests for coralnet-remediate (no network)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from mermaidseg.datasets.coralnet.scraper.models import SourceProbe
from mermaidseg.datasets.coralnet.scraper.remediate_incomplete import (
    build_parser,
    cmd_probe,
    cmd_redownload,
)


def test_remediate_parser_has_subcommands():
    p = build_parser()
    actions = next(a for a in p._actions if getattr(a, "choices", None))
    assert set(actions.choices) == {"probe", "redownload", "all"}


_CREDS_TARGET = (
    "mermaidseg.datasets.coralnet.scraper.remediate_incomplete.load_coralnet_credentials"
)
_DL_TARGET = "mermaidseg.datasets.coralnet.scraper.remediate_incomplete.CoralNetDownloader"


def test_remediate_probe_writes_report(tmp_path):
    incomp = pd.DataFrame({"source_id": [7]})
    audit = pd.DataFrame(
        {
            "source_id": [7],
            "has_images_folder": [False],
            "has_annotations_csv": [False],
            "has_image_list_csv": [False],
            "n_images_s3": [0],
            "n_images_csv": [0],
            "n_annotations": [0],
            "annotations_csv_read_failed": [False],
            "annotations_empty": [True],
            "image_count_match": [False],
        }
    )
    ipath = tmp_path / "incomplete.parquet"
    apath = tmp_path / "audit.parquet"
    rpath = tmp_path / "report.parquet"
    incomp.to_parquet(ipath)
    audit.to_parquet(apath)

    class _StubDL:
        def __init__(self, *_, **__) -> None:
            pass

        def login(self) -> bool:
            return True

        def probe_source(self, sid):  # noqa: ANN001
            return SourceProbe(
                int(sid),
                f"http://mock/{sid}",
                accessible=True,
                error=None,
                total_images_website=5,
            )

    with patch(_CREDS_TARGET, return_value=("u", "p")), patch(_DL_TARGET, _StubDL):
        ns = Namespace(
            incomplete_parquet=ipath,
            audit_parquet=apath,
            report_out=rpath,
            workers=1,
            delay_seconds=0.0,
        )
        assert cmd_probe(ns) == 0

    out = pd.read_parquet(rpath)
    assert len(out) == 1
    assert int(out.iloc[0]["source_id"]) == 7
    for col in ("n_confirmed_website", "n_unconfirmed_website", "n_unclassified_website"):
        assert col in out.columns


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
