"""Tests for the scrape_coralnet_s3 entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from coralnet_scraper import scrape_coralnet_s3 as cli

_MOD = "coralnet_scraper.scrape_coralnet_s3"


def test_main_raises_before_any_network_call_when_bucket_unset(monkeypatch):
    monkeypatch.delenv("MERMAID_CORALNET_BUCKET", raising=False)
    with (
        patch(f"{_MOD}.load_coralnet_credentials") as creds,
        patch(f"{_MOD}.CoralNetDownloader") as downloader_cls,
        pytest.raises(ValueError, match="MERMAID_CORALNET_BUCKET"),
    ):
        cli.main([])
    creds.assert_not_called()
    downloader_cls.assert_not_called()
    downloader_cls.return_value.login.assert_not_called()


def test_main_passes_configured_bucket_and_prefix(monkeypatch):
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", "my-prefix")
    downloader = MagicMock(max_workers=4)
    downloader.login.return_value = True
    with (
        patch(f"{_MOD}.load_coralnet_credentials", return_value=("u", "p")),
        patch(f"{_MOD}.CoralNetDownloader", return_value=downloader),
        patch(f"{_MOD}.boto3.client"),
        patch(f"{_MOD}.discover_public_source_ids", return_value=[1, 2]),
        patch(f"{_MOD}.run_batch_download") as run,
    ):
        assert cli.main([]) == 0
    kwargs = run.call_args.kwargs
    assert kwargs["bucket"] == "my-bucket"
    assert kwargs["s3_prefix"] == "my-prefix"
    assert run.call_args.args[0] == [1, 2]


def _run_main(monkeypatch, argv):
    """Run ``main`` with only the network pieces patched; return (downloader_cls, run_batch)."""
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")
    downloader = MagicMock(max_workers=4)
    downloader.login.return_value = True
    with (
        patch(f"{_MOD}.load_coralnet_credentials", return_value=("u", "p")),
        patch(f"{_MOD}.CoralNetDownloader", return_value=downloader) as downloader_cls,
        patch(f"{_MOD}.boto3.client"),
        patch(f"{_MOD}.discover_public_source_ids", return_value=[1]),
        patch(f"{_MOD}.run_batch_download") as run,
    ):
        assert cli.main(argv) == 0
    return downloader_cls, run


@pytest.mark.parametrize(("argv", "force"), [([], False), (["--force"], True)])
def test_main_passes_force_flag(monkeypatch, argv, force):
    _, run = _run_main(monkeypatch, argv)
    assert run.call_args.kwargs["force"] is force


def test_main_passes_delay_seconds(monkeypatch):
    _, run = _run_main(monkeypatch, ["--delay-seconds", "1.5"])
    assert run.call_args.kwargs["delay_seconds"] == 1.5


def test_main_passes_scraper_workers_to_downloader(monkeypatch):
    downloader_cls, _ = _run_main(monkeypatch, ["--scraper-workers", "5"])
    assert downloader_cls.call_args.kwargs["max_workers"] == 5
