"""Tests for the scrape_coralnet_s3 entry point."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from coralnet_scraper import scrape_coralnet_s3 as cli
from coralnet_scraper.batch_download import BatchOutcome
from coralnet_scraper.models import DownloadResult, ImageFailures

_MOD = "coralnet_scraper.scrape_coralnet_s3"

_TRANSPORT = {
    "requests": 12,
    "mean_rate_per_s": 1.5,
    "retries": {"429": 2},
    "final_status": {"200": 12},
}


class _FakeDownloader:
    """Downloader double that keeps its constructor settings and reports fixed transport stats."""

    def __init__(
        self,
        user: str,
        password: str,
        *,
        max_workers: int | None,
        export_workers: int | None,
        requests_per_second: float | None,
        login_ok: bool = True,
    ) -> None:
        self.max_workers = max_workers if max_workers is not None else 4
        self.export_workers = export_workers if export_workers is not None else 3
        self.requests_per_second = requests_per_second
        self.session = object()
        self._login_ok = login_ok

    def login(self) -> bool:
        return self._login_ok

    def transport_stats(self) -> dict[str, Any]:
        return _TRANSPORT


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


def _run_main(
    monkeypatch,
    argv,
    results=None,
    expected_code=0,
    *,
    stopped=None,
    discovered=(1,),
    fake_s3=None,
):
    """Run ``main`` with the network pieces faked; return (downloader_cls, run_batch, discover)."""
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", "my-prefix")
    outcome = BatchOutcome(results=results or {}, stopped=stopped)
    with (
        patch(f"{_MOD}.load_coralnet_credentials", return_value=("u", "p")),
        patch(f"{_MOD}.CoralNetDownloader", side_effect=_FakeDownloader) as downloader_cls,
        patch(f"{_MOD}.boto3.client", return_value=fake_s3 or MagicMock()),
        patch(f"{_MOD}.discover_public_source_ids", return_value=list(discovered)) as discover,
        patch(f"{_MOD}.run_batch_download", return_value=outcome) as run,
    ):
        assert cli.main(argv) == expected_code
    return downloader_cls, run, discover


def test_main_passes_configured_bucket_prefix_and_discovered_ids(monkeypatch, fake_s3):
    downloader_cls, run, discover = _run_main(monkeypatch, [], discovered=[1, 2], fake_s3=fake_s3)

    kwargs = run.call_args.kwargs
    assert kwargs["bucket"] == "my-bucket"
    assert kwargs["s3_prefix"] == "my-prefix"
    assert kwargs["s3_client"] is fake_s3
    assert run.call_args.args[0] == [1, 2]
    assert discover.call_args.args[0] is kwargs["downloader"].session


@pytest.mark.parametrize(("argv", "force"), [([], False), (["--force"], True)])
def test_main_passes_force_flag(monkeypatch, argv, force):
    _, run, _ = _run_main(monkeypatch, argv)
    assert run.call_args.kwargs["force"] is force


def test_main_passes_delay_seconds(monkeypatch):
    _, run, _ = _run_main(monkeypatch, ["--delay-seconds", "1.5"])
    assert run.call_args.kwargs["delay_seconds"] == 1.5


def test_main_passes_scraper_workers_to_downloader(monkeypatch):
    downloader_cls, _, _ = _run_main(monkeypatch, ["--scraper-workers", "5"])
    assert downloader_cls.call_args.kwargs["max_workers"] == 5


_MIXED_RESULTS = {
    101: DownloadResult(ok=True),
    202: DownloadResult(
        ok=False,
        errors=["images_missing_url"],
        image_failures=ImageFailures(missing_url=14),
    ),
    303: DownloadResult(
        ok=False,
        errors=["images_missing_url", "images_upload_failed"],
        image_failures=ImageFailures(missing_url=2, upload_failed=5),
    ),
    404: DownloadResult(ok=False, errors=["ValueError"]),
}
_MIXED_LINES = [
    "Failed to download 3 source(s)",
    "ValueError: [404]",
    "images_missing_url: 202 (14 images), 303 (2 images)",
    "images_upload_failed: 303 (5 images)",
]


def test_summarize_failures_groups_by_error_kind():
    assert cli._summarize_failures(_MIXED_RESULTS) == _MIXED_LINES


def test_summarize_failures_lists_failure_without_error_code_as_no_error_code():
    results = {
        7: DownloadResult(ok=False),
        3: DownloadResult(ok=False, errors=["annotations_failed"]),
    }
    assert cli._summarize_failures(results) == [
        "Failed to download 2 source(s)",
        "annotations_failed: [3]",
        "no_error_code: [7]",
    ]


def test_summarize_failures_is_empty_when_nothing_failed():
    results = {
        1: DownloadResult(ok=True),
        2: DownloadResult(ok=True, skipped_no_confirmed=True),
    }
    assert cli._summarize_failures(results) == []


def test_main_logs_failure_summary_and_returns_1(monkeypatch, caplog):
    _run_main(monkeypatch, [], results=_MIXED_RESULTS, expected_code=1)
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == _MIXED_LINES


def test_main_returns_1_without_discovery_or_report_when_login_fails(monkeypatch, fake_s3):
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")

    def failing_login(*args, **kwargs):
        return _FakeDownloader(*args, **kwargs, login_ok=False)

    with (
        patch(f"{_MOD}.load_coralnet_credentials", return_value=("u", "p")),
        patch(f"{_MOD}.CoralNetDownloader", side_effect=failing_login),
        patch(f"{_MOD}.boto3.client", return_value=fake_s3),
        patch(f"{_MOD}.discover_public_source_ids") as discover,
        patch(f"{_MOD}.run_batch_download") as run,
    ):
        assert cli.main([]) == 1
    discover.assert_not_called()
    run.assert_not_called()
    assert fake_s3.objects == {}


def _run_report(fake_s3) -> dict:
    (key,) = [k for (_b, k) in fake_s3.objects]
    assert key.startswith("my-prefix/_runs/")
    assert key.endswith(".json")
    return json.loads(fake_s3.objects[("my-bucket", key)])


def test_main_reports_a_circuit_breaker_stop(monkeypatch, caplog, fake_s3):
    results = {5: DownloadResult(ok=False, errors=["annotations_failed"])}
    argv = ["--scraper-workers", "6", "--export-workers", "2", "--requests-per-second", "1.5"]

    _run_main(monkeypatch, argv, results, 1, stopped="circuit_open", fake_s3=fake_s3)

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors[0] == "Run stopped early: circuit_open"
    report = _run_report(fake_s3)
    assert report["stopped"] == "circuit_open"
    assert report["settings"] == {
        "scraper_workers": 6,
        "export_workers": 2,
        "requests_per_second": 1.5,
        "source_ids": None,
    }
    assert report["transport"] == _TRANSPORT
    assert report["sources"] == {"5": {"ok": False, "errors": ["annotations_failed"]}}


def test_main_logs_the_summary_and_returns_1_when_the_report_write_fails(
    monkeypatch, caplog, fake_s3
):
    def failing_put(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    monkeypatch.setattr(fake_s3, "put_object", failing_put)
    caplog.set_level("INFO")

    _run_main(monkeypatch, [], _MIXED_RESULTS, 1, fake_s3=fake_s3)

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors[:-1] == _MIXED_LINES
    assert errors[-1].startswith("Could not write run report s3://my-bucket/my-prefix/_runs/")
    assert any(r.getMessage().startswith("Transport: ") for r in caplog.records)


def test_main_returns_1_when_the_report_write_fails_after_an_all_ok_run(
    monkeypatch, caplog, fake_s3
):
    def failing_put(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    monkeypatch.setattr(fake_s3, "put_object", failing_put)

    _run_main(monkeypatch, [], {1: DownloadResult(ok=True)}, 1, fake_s3=fake_s3)

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any(e.startswith("Could not write run report") for e in errors)


def test_main_reports_a_finished_run_as_not_stopped(monkeypatch, fake_s3):
    _run_main(monkeypatch, [], {1: DownloadResult(ok=True)}, 0, fake_s3=fake_s3)

    report = _run_report(fake_s3)
    assert report["stopped"] is None
    assert report["sources"] == {"1": {"ok": True, "errors": []}}


@pytest.mark.parametrize(
    ("flag", "discovered", "expected"),
    [("3,1", [1, 2, 3], [3, 1]), ("9", [1, 2], [9])],
)
def test_main_processes_only_the_given_source_ids_in_order(
    monkeypatch, fake_s3, flag, discovered, expected
):
    _, run, _ = _run_main(
        monkeypatch, ["--source-ids", flag], discovered=discovered, fake_s3=fake_s3
    )

    assert run.call_args.args[0] == expected
    assert _run_report(fake_s3)["settings"]["source_ids"] == expected


def test_main_rejects_a_non_integer_source_id(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--source-ids", "1,x"])
