"""Tests for the scrape_coralnet_s3 entry point."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from conftest import FakeS3

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
        self.totals: dict[int, int | None] = {}
        self.verify_results: dict[int, list[str] | Exception] = {}
        self.verified: list[int] = []
        self.login_calls = 0

    def login(self) -> bool:
        self.login_calls += 1
        return self._login_ok

    def transport_stats(self) -> dict[str, Any]:
        return _TRANSPORT

    def fetch_total_images(self, source_id: int) -> int | None:
        return self.totals[source_id]

    def verify_existing(self, source_id: int, bucket_name: str, s3_prefix: str) -> list[str]:
        self.verified.append(source_id)
        result = self.verify_results[source_id]
        if isinstance(result, Exception):
            raise result
        return result


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
        patch(f"{_MOD}.boto3.client", return_value=fake_s3 if fake_s3 is not None else FakeS3()),
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


@pytest.mark.parametrize(
    ("argv", "prune", "dry_run"),
    [([], False, False), (["--prune"], True, False), (["--prune", "--dry-run"], True, True)],
)
def test_main_passes_the_prune_and_dry_run_flags(monkeypatch, fake_s3, argv, prune, dry_run):
    _, run, _ = _run_main(monkeypatch, argv, {1: DownloadResult(ok=True)}, fake_s3=fake_s3)

    assert run.call_args.kwargs["prune"] is prune
    assert run.call_args.kwargs["dry_run"] is dry_run
    settings = _run_report(fake_s3)["settings"]
    assert (settings["prune"], settings["dry_run"]) == (prune, dry_run)


@pytest.mark.parametrize(("argv", "cooldown"), [([], 600.0), (["--breaker-cooldown", "30"], 30.0)])
def test_main_passes_the_breaker_cooldown_and_reports_it(monkeypatch, fake_s3, argv, cooldown):
    _, run, _ = _run_main(monkeypatch, argv, {1: DownloadResult(ok=True)}, fake_s3=fake_s3)

    assert run.call_args.kwargs["breaker_cooldown_s"] == cooldown
    assert _run_report(fake_s3)["settings"]["breaker_cooldown_s"] == cooldown


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


def test_summarize_failures_reports_skipped_images_under_images_outage():
    results = {
        5: DownloadResult(
            ok=False,
            errors=["images_outage", "images_missing_url"],
            image_failures=ImageFailures(missing_url=20, skipped=30),
        ),
    }
    assert cli._summarize_failures(results) == [
        "Failed to download 1 source(s)",
        "images_missing_url: 5 (20 images)",
        "images_outage: 5 (30 images)",
    ]


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
    (key,) = [k for (_b, k) in fake_s3.objects if "/_runs/" in k]
    assert key.startswith("my-prefix/_runs/")
    assert key.endswith(".json")
    return json.loads(fake_s3.objects[("my-bucket", key)])


def test_main_reports_a_circuit_breaker_stop(monkeypatch, caplog, fake_s3):
    results = {5: DownloadResult(ok=False, errors=["annotations_failed"])}
    argv = ["--scraper-workers", "6", "--export-workers", "2", "--requests-per-second", "1.5"]
    argv += ["--prune", "--allow-source-drop"]

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
        "prune": True,
        "dry_run": False,
        "allow_source_drop": True,
        "breaker_cooldown_s": 600.0,
    }
    assert report["transport"] == _TRANSPORT
    assert report["sources"] == {"5": {"ok": False, "errors": ["annotations_failed"]}}


def _put_failing_for_run_reports(fake_s3):
    real_put = fake_s3.put_object

    def put(**kwargs):
        if "/_runs/" in kwargs["Key"]:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")
        return real_put(**kwargs)

    return put


def test_main_logs_the_summary_and_returns_1_when_the_report_write_fails(
    monkeypatch, caplog, fake_s3
):
    monkeypatch.setattr(fake_s3, "put_object", _put_failing_for_run_reports(fake_s3))
    caplog.set_level("INFO")

    _run_main(monkeypatch, [], _MIXED_RESULTS, 1, fake_s3=fake_s3)

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors[:-1] == _MIXED_LINES
    assert errors[-1].startswith("Could not write run report s3://my-bucket/my-prefix/_runs/")
    assert any(r.getMessage().startswith("Transport: ") for r in caplog.records)


def test_main_returns_1_when_the_report_write_fails_after_an_all_ok_run(
    monkeypatch, caplog, fake_s3
):
    monkeypatch.setattr(fake_s3, "put_object", _put_failing_for_run_reports(fake_s3))

    _run_main(monkeypatch, [], {1: DownloadResult(ok=True)}, 1, fake_s3=fake_s3)

    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any(e.startswith("Could not write run report") for e in errors)


def test_main_reports_a_finished_run_as_not_stopped(monkeypatch, fake_s3):
    _run_main(monkeypatch, [], {1: DownloadResult(ok=True)}, 0, fake_s3=fake_s3)

    report = _run_report(fake_s3)
    assert report["stopped"] is None
    assert (report["settings"]["prune"], report["settings"]["allow_source_drop"]) == (False, False)
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


_LISTING_KEY = ("my-bucket", "my-prefix/_meta/discovered_source_ids.json")


@pytest.mark.parametrize(
    ("now", "argv", "code", "proceeds"),
    [
        (list(range(1, 19)), [], 1, False),
        (list(range(1, 20)), [], 0, True),
        (list(range(1, 19)), ["--allow-source-drop"], 0, True),
    ],
    ids=["drop-to-18-stops", "drop-to-19-runs", "drop-to-18-allowed"],
)
def test_discovery_guard_stops_a_run_whose_listing_shrank_below_95_percent(
    monkeypatch, fake_s3, now, argv, code, proceeds
):
    previous = json.dumps(list(range(1, 21))).encode()
    fake_s3.put(*_LISTING_KEY, previous)

    _, run, _ = _run_main(monkeypatch, argv, expected_code=code, discovered=now, fake_s3=fake_s3)

    assert run.called is proceeds
    expected_listing = now if proceeds else list(range(1, 21))
    assert json.loads(fake_s3.objects[_LISTING_KEY]) == expected_listing


@pytest.mark.parametrize("baseline", [None, [1, 2, 3]], ids=["no-baseline", "with-baseline"])
@pytest.mark.parametrize("argv", [[], ["--allow-source-drop"]], ids=["default", "allow-drop"])
def test_an_empty_discovery_stops_the_run_and_keeps_the_baseline(
    monkeypatch, fake_s3, caplog, baseline, argv
):
    if baseline is not None:
        fake_s3.put(*_LISTING_KEY, json.dumps(baseline).encode())

    _, run, _ = _run_main(monkeypatch, argv, expected_code=1, discovered=[], fake_s3=fake_s3)

    run.assert_not_called()
    if baseline is None:
        assert _LISTING_KEY not in fake_s3.objects
    else:
        assert json.loads(fake_s3.objects[_LISTING_KEY]) == baseline
    assert any(r.levelname == "ERROR" for r in caplog.records)


@pytest.mark.parametrize(
    "argv",
    [
        ["--prune", "verify-existing"],
        ["--allow-source-drop", "invalidate", "--all"],
        ["--prune", "invalidate", "--changed"],
        ["--dry-run", "verify-existing"],
        ["--breaker-cooldown", "30", "verify-existing"],
        ["--breaker-cooldown", "0", "invalidate", "--all"],
        ["--breaker-cooldown", "-1"],
    ],
)
def test_batch_only_flags_are_rejected_with_a_subcommand_or_a_negative_value(argv):
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(argv)

    assert exc.value.code == 2


def test_batch_dry_run_without_prune_is_rejected():
    with pytest.raises(SystemExit) as exc:
        cli._parse_args(["--dry-run"])

    assert exc.value.code == 2


def _legacy_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and "verify-existing" in r.getMessage()
    ]


def test_a_bucket_of_folders_without_markers_warns_once_and_the_run_proceeds(
    monkeypatch, fake_s3, caplog
):
    fake_s3.put("my-bucket", "my-prefix/s1/annotations.csv", b"Name\n")
    fake_s3.put("my-bucket", "my-prefix/s2/annotations.csv", b"Name\n")

    _, run, _ = _run_main(monkeypatch, [], discovered=[1, 2], fake_s3=fake_s3)

    (warning,) = _legacy_warnings(caplog)
    assert "2" in warning
    run.assert_called_once()


def test_no_legacy_warning_when_any_stored_source_has_a_marker(monkeypatch, fake_s3, caplog):
    _store_source(fake_s3, 1, 5)
    fake_s3.put("my-bucket", "my-prefix/s2/annotations.csv", b"Name\n")

    _run_main(monkeypatch, [], discovered=[1, 2], fake_s3=fake_s3)

    assert _legacy_warnings(caplog) == []


def test_no_legacy_warning_for_an_empty_bucket(monkeypatch, fake_s3, caplog):
    _run_main(monkeypatch, [], fake_s3=fake_s3)

    assert _legacy_warnings(caplog) == []


def test_main_reports_the_skipped_sources(monkeypatch, fake_s3):
    results = {
        1: DownloadResult(ok=True),
        2: DownloadResult(ok=True, skipped_no_confirmed=True),
        3: DownloadResult(ok=True, skipped_no_images=True),
        4: DownloadResult(ok=True, skipped_no_confirmed=True),
    }
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")
    outcome = BatchOutcome(results=results, stopped=None, skipped_complete=5)
    with (
        patch(f"{_MOD}.load_coralnet_credentials", return_value=("u", "p")),
        patch(f"{_MOD}.CoralNetDownloader", side_effect=_FakeDownloader),
        patch(f"{_MOD}.boto3.client", return_value=fake_s3),
        patch(f"{_MOD}.discover_public_source_ids", return_value=[1, 2, 3, 4]),
        patch(f"{_MOD}.run_batch_download", return_value=outcome),
    ):
        monkeypatch.setenv("MERMAID_CORALNET_PREFIX", "my-prefix")
        assert cli.main([]) == 0

    assert _run_report(fake_s3)["skipped"] == {
        "complete": 5,
        "no_confirmed": [2, 4],
        "no_images": [3],
    }


def _run_command(
    monkeypatch,
    fake_s3,
    argv,
    *,
    discovered=(),
    totals=None,
    verify_results=None,
) -> tuple[int, _FakeDownloader, MagicMock]:
    """Run a subcommand of ``main`` with no CoralNet credentials anywhere.

    Returns its exit code, the downloader and the batch mock.
    """
    monkeypatch.delenv("CORALNET_USERNAME", raising=False)
    monkeypatch.delenv("CORALNET_PASSWORD", raising=False)
    monkeypatch.setenv("MERMAID_CORALNET_BUCKET", "my-bucket")
    monkeypatch.setenv("MERMAID_CORALNET_PREFIX", "my-prefix")
    made: list[_FakeDownloader] = []

    def make(*args, **kwargs):
        dl = _FakeDownloader(*args, **kwargs)
        dl.totals = totals or {}
        dl.verify_results = verify_results or {}
        made.append(dl)
        return dl

    with (
        patch(f"{_MOD}.load_coralnet_credentials") as credentials,
        patch(f"{_MOD}.CoralNetDownloader", side_effect=make),
        patch(f"{_MOD}.boto3.client", return_value=fake_s3),
        patch(f"{_MOD}.discover_public_source_ids", return_value=list(discovered)),
        patch(f"{_MOD}.run_batch_download") as run,
    ):
        code = cli.main(argv)
    credentials.assert_not_called()
    return code, made[0], run


def _store_source(fake_s3, sid: int, total: int) -> None:
    """A complete source with leftover partial work."""
    fake_s3.put(
        "my-bucket",
        f"my-prefix/s{sid}/_complete.json",
        json.dumps({"total_images": total}).encode(),
    )
    fake_s3.put("my-bucket", f"my-prefix/s{sid}/_partial/state.json", b"{}")
    fake_s3.put("my-bucket", f"my-prefix/s{sid}/annotations.csv", b"Name\n")


def _remaining(fake_s3) -> list[str]:
    return sorted(k.removeprefix("my-prefix/") for (_b, k) in fake_s3.objects)


def test_invalidate_source_ids_deletes_only_their_markers_and_partial_state(monkeypatch, fake_s3):
    for sid in (1, 2, 3):
        _store_source(fake_s3, sid, 10)

    code, _, run = _run_command(monkeypatch, fake_s3, ["invalidate", "--source-ids", "1,2"])

    assert code == 0
    run.assert_not_called()
    assert _remaining(fake_s3) == [
        "s1/annotations.csv",
        "s2/annotations.csv",
        "s3/_complete.json",
        "s3/_partial/state.json",
        "s3/annotations.csv",
    ]


def test_invalidate_all_deletes_every_marker(monkeypatch, fake_s3):
    for sid in (1, 22):
        _store_source(fake_s3, sid, 10)
    fake_s3.put("my-bucket", "my-prefix/_meta/discovered_source_ids.json", b"[1, 22]")

    code, _, _ = _run_command(monkeypatch, fake_s3, ["invalidate", "--all", "--yes"])

    assert code == 0
    assert _remaining(fake_s3) == [
        "_meta/discovered_source_ids.json",
        "s1/annotations.csv",
        "s22/annotations.csv",
    ]


def test_invalidate_all_without_yes_or_dry_run_exits_2_and_deletes_nothing(monkeypatch, fake_s3):
    for sid in (1, 22):
        _store_source(fake_s3, sid, 10)
    before = _remaining(fake_s3)

    with pytest.raises(SystemExit) as exc:
        _run_command(monkeypatch, fake_s3, ["invalidate", "--all"])

    assert exc.value.code == 2
    assert _remaining(fake_s3) == before


@pytest.mark.parametrize(
    "selector", [["--source-ids", "1"], ["--all"], ["--changed"]], ids=["ids", "all", "changed"]
)
def test_invalidate_dry_run_deletes_nothing_and_names_the_sources(
    monkeypatch, fake_s3, caplog, selector
):
    caplog.set_level(logging.INFO)
    _store_source(fake_s3, 1, 5)

    code, _, _ = _run_command(
        monkeypatch, fake_s3, ["invalidate", *selector, "--dry-run"], totals={1: 6}
    )

    assert code == 0
    assert ("my-bucket", "my-prefix/s1/_complete.json") in fake_s3.objects
    assert ("my-bucket", "my-prefix/s1/_partial/state.json") in fake_s3.objects
    assert "Would invalidate 1 sources: [1]" in [r.getMessage() for r in caplog.records]


def test_invalidate_changed_deletes_the_marker_of_a_source_whose_total_changed(
    monkeypatch, fake_s3
):
    _store_source(fake_s3, 1, 5)
    _store_source(fake_s3, 2, 7)

    code, dl, _ = _run_command(
        monkeypatch, fake_s3, ["invalidate", "--changed"], totals={1: 5, 2: 8}
    )

    assert code == 0
    assert dl.login_calls == 0
    assert ("my-bucket", "my-prefix/s1/_complete.json") in fake_s3.objects
    assert ("my-bucket", "my-prefix/s2/_complete.json") not in fake_s3.objects


def test_verify_existing_skips_marked_sources_and_exits_1_listing_failures(
    monkeypatch, fake_s3, caplog
):
    _store_source(fake_s3, 1, 5)

    code, dl, run = _run_command(
        monkeypatch,
        fake_s3,
        ["verify-existing"],
        discovered=[1, 2, 3],
        verify_results={2: [], 3: ["annotations_missing", "image_list_missing"]},
    )

    assert code == 1
    assert dl.verified == [2, 3]
    run.assert_not_called()
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == ["Source 3 not verified: annotations_missing, image_list_missing"]


def test_verify_existing_exits_0_when_every_source_passes(monkeypatch, fake_s3):
    code, dl, _ = _run_command(
        monkeypatch, fake_s3, ["verify-existing"], discovered=[2, 3], verify_results={2: [], 3: []}
    )

    assert code == 0
    assert dl.verified == [2, 3]
    assert dl.login_calls == 0


def test_verify_existing_checks_only_the_given_source_ids(monkeypatch, fake_s3):
    code, dl, _ = _run_command(
        monkeypatch,
        fake_s3,
        ["verify-existing", "--source-ids", "1,2"],
        discovered=[1, 2, 3],
        verify_results={1: [], 2: [], 3: []},
    )

    assert code == 0
    assert dl.verified == [1, 2]


def test_invalidate_changed_continues_past_an_unreadable_source_and_exits_1(
    monkeypatch, fake_s3, caplog
):
    _store_source(fake_s3, 1, 5)
    _store_source(fake_s3, 10, 7)
    _store_source(fake_s3, 3, 2)
    fake_s3.put("my-bucket", "my-prefix/s2/annotations.csv", b"Name\n")
    fake_s3.put("my-bucket", "my-prefix/s2/_partial/state.json", b"{}")

    code, _, _ = _run_command(
        monkeypatch, fake_s3, ["invalidate", "--changed"], totals={1: 6, 3: 9}
    )

    assert code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any("Source 10" in e for e in errors)
    assert any("[10]" in e for e in errors)
    assert _remaining(fake_s3) == [
        "s1/annotations.csv",
        "s10/_complete.json",
        "s10/_partial/state.json",
        "s10/annotations.csv",
        "s2/_partial/state.json",
        "s2/annotations.csv",
        "s3/annotations.csv",
    ]


def test_verify_existing_fails_a_source_reporting_a_missing_overview_total(
    monkeypatch, fake_s3, caplog
):
    code, _, _ = _run_command(
        monkeypatch,
        fake_s3,
        ["verify-existing"],
        discovered=[2],
        verify_results={2: ["overview_total_missing"]},
    )

    assert code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == ["Source 2 not verified: overview_total_missing"]


def test_verify_existing_reports_a_raising_source_by_exception_name_and_goes_on(
    monkeypatch, fake_s3, caplog
):
    code, dl, _ = _run_command(
        monkeypatch,
        fake_s3,
        ["verify-existing"],
        discovered=[2, 3],
        verify_results={2: ValueError("boom"), 3: []},
    )

    assert code == 1
    assert dl.verified == [2, 3]
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert "Source 2 not verified: ValueError" in errors
