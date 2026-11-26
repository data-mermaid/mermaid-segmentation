"""Tests for batch download orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests
from conftest import Reply, ScriptedServer

from coralnet_scraper import batch_download as bd
from coralnet_scraper.http import SessionFactory
from coralnet_scraper.models import DownloadResult

_MARKER = "p/s{}/_complete.json"


def _mark(fake_s3, source_id: int) -> None:
    fake_s3.put_object(Bucket="b", Key=_MARKER.format(source_id), Body=b"{}")


def _skip(fake_s3, source_id: int) -> bool:
    return bd.should_skip_download(
        source_id=source_id, bucket="b", s3_prefix="p", s3_client=fake_s3
    )


def test_should_skip_only_a_source_with_a_marker(fake_s3):
    fake_s3.put_object(Bucket="b", Key="p/s1/annotations.csv", Body=b"a\n")
    _mark(fake_s3, 2)

    assert _skip(fake_s3, 1) is False
    assert _skip(fake_s3, 2) is True


@pytest.mark.parametrize(("prune", "dry_run"), [(False, False), (True, False), (True, True)])
def test_run_batch_download_downloads_only_unmarked_sources_and_counts_the_skipped(
    fake_s3, prune, dry_run
):
    _mark(fake_s3, 1)
    downloader = MagicMock()
    result = MagicMock(skipped_no_confirmed=False)
    downloader.download_source.return_value = result

    outcome = bd.run_batch_download(
        [1, 2],
        downloader=downloader,
        bucket="b",
        s3_prefix="p",
        s3_client=fake_s3,
        prune=prune,
        dry_run=dry_run,
        breaker_cooldown_s=0,
    )

    downloader.download_source.assert_called_once_with(
        2, bucket_name="b", s3_prefix="p", prune=prune, dry_run=dry_run
    )
    assert outcome.results == {2: result}
    assert outcome.skipped_complete == 1


def _run(fake_s3, downloader):
    return bd.run_batch_download(
        [1, 2, 3],
        downloader=downloader,
        bucket="b",
        s3_prefix="p",
        s3_client=fake_s3,
        breaker_cooldown_s=0,
    ).results


def test_run_batch_download_continues_after_a_source_raises(fake_s3):
    ok = MagicMock(skipped_no_confirmed=False)
    downloader = MagicMock()
    downloader.download_source.side_effect = [ok, ValueError("boom"), ok]

    results = _run(fake_s3, downloader)

    assert downloader.download_source.call_count == 3
    assert results[1] is ok
    assert results[3] is ok
    assert results[2].ok is False
    assert results[2].errors == ["ValueError"]


def test_run_batch_download_records_a_raising_skip_check(fake_s3):
    downloader = MagicMock()
    ok = MagicMock(skipped_no_confirmed=False)
    downloader.download_source.return_value = ok
    real = bd.should_skip_download

    def flaky(*, source_id, **kwargs):
        if source_id == 2:
            raise RuntimeError("s3 down")
        return real(source_id=source_id, **kwargs)

    with patch.object(bd, "should_skip_download", side_effect=flaky):
        results = _run(fake_s3, downloader)

    assert downloader.download_source.call_count == 2
    assert results[2].ok is False
    assert results[2].errors == ["RuntimeError"]


_LISTING_HTML = """\
<html><body>
<a href="/about/">About</a>
<a href="/source/new/">New Source</a>
<a href="/source/invites/">Source Invites</a>
<a href="/source/100/">Reef A</a>
<a href="https://coralnet.ucsd.edu/source/7/">Reef B</a>
<a href="/source/12/">Reef C</a>
<a href="/source/12/">Reef C again</a>
<a href="mailto:help@coralnet.org?subject=/source/55/x">Contact</a>
</body></html>"""


def _listing_session(
    http_server: ScriptedServer, monkeypatch: pytest.MonkeyPatch, *replies: Reply
) -> requests.Session:
    http_server.script("/source/about/", *replies)
    monkeypatch.setattr(bd, "_SOURCE_LISTING_URL", f"{http_server.url}/source/about/")
    return SessionFactory(pool_size=2, requests_per_second=None).coralnet()


def test_discover_public_source_ids_returns_sorted_unique_ids_from_source_links(
    http_server, monkeypatch
):
    session = _listing_session(http_server, monkeypatch, Reply(200, _LISTING_HTML.encode()))

    assert bd.discover_public_source_ids(session) == [7, 12, 100]


def test_discover_public_source_ids_raises_when_the_listing_keeps_failing(http_server, monkeypatch):
    session = _listing_session(http_server, monkeypatch, Reply(500))

    with pytest.raises(requests.HTTPError):
        bd.discover_public_source_ids(session)


def _no_prompt() -> tuple:
    return (
        patch("builtins.input", side_effect=AssertionError("prompted")),
        patch.object(bd.getpass, "getpass", side_effect=AssertionError("prompted")),
    )


def test_load_credentials_reads_env_and_strips_only_the_username(monkeypatch):
    monkeypatch.setenv("CORALNET_USERNAME", "  alice  ")
    monkeypatch.setenv("CORALNET_PASSWORD", " pw ")
    no_input, no_getpass = _no_prompt()
    with no_input, no_getpass:
        assert bd.load_coralnet_credentials() == ("alice", " pw ")


@pytest.mark.parametrize(
    ("env_user", "env_password"),
    [("alice", None), (None, "pw"), ("   ", "pw"), ("alice", "")],
)
def test_load_credentials_prompts_unless_both_env_values_are_set(
    monkeypatch, env_user, env_password
):
    for name, value in (("CORALNET_USERNAME", env_user), ("CORALNET_PASSWORD", env_password)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with (
        patch("builtins.input", return_value="  bob ") as prompt_user,
        patch.object(bd.getpass, "getpass", return_value="typed-pw") as prompt_pw,
    ):
        creds = bd.load_coralnet_credentials()

    assert creds == ("bob", "typed-pw")
    prompt_user.assert_called_once_with("CoralNet username: ")
    prompt_pw.assert_called_once_with("CoralNet password: ")


@pytest.mark.parametrize(
    ("delay", "expected_sleeps"),
    [(2.5, [call(2.5), call(2.5)]), (0.0, [])],
)
def test_run_batch_download_sleeps_after_each_download_not_after_a_skip(
    fake_s3, delay, expected_sleeps
):
    _mark(fake_s3, 1)  # source 1 is skipped
    downloader = MagicMock()
    downloader.download_source.side_effect = [
        MagicMock(skipped_no_confirmed=False),
        ValueError("boom"),  # a failed source is followed by the delay too
    ]

    with patch.object(bd.time, "sleep") as sleep:
        bd.run_batch_download(
            [1, 2, 3],
            downloader=downloader,
            bucket="b",
            s3_prefix="p",
            s3_client=fake_s3,
            delay_seconds=delay,
            breaker_cooldown_s=0,
        )

    assert sleep.call_args_list == expected_sleeps


class _ScriptedDownloader:
    """Downloader double answering each source from a script.

    A script value is True (ok), False (failed with ``annotations_failed``), one error code,
    a list of error codes (failed with exactly those) or an exception to raise.
    """

    def __init__(self, script: dict[int, bool | str | list[str] | Exception]) -> None:
        self.script = script
        self.attempted: list[int] = []

    def download_source(
        self, sid: int, *, bucket_name: str, s3_prefix: str, prune: bool, dry_run: bool
    ) -> DownloadResult:
        self.attempted.append(sid)
        outcome = self.script[sid]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return DownloadResult(ok=False, errors=[outcome])
        if isinstance(outcome, list):
            return DownloadResult(ok=False, errors=outcome)
        return DownloadResult(ok=outcome, errors=[] if outcome else ["annotations_failed"])


def _batch(
    fake_s3, script: dict[int, bool | str | list[str] | Exception], cooldown: float = 0
) -> tuple[bd.BatchOutcome, list[int]]:
    downloader = _ScriptedDownloader(script)
    outcome = bd.run_batch_download(
        list(script),
        downloader=downloader,
        bucket="b",
        s3_prefix="p",
        s3_client=fake_s3,
        breaker_cooldown_s=cooldown,
    )
    return outcome, downloader.attempted


def test_five_consecutive_failures_open_the_circuit_and_stop_the_run(fake_s3):
    outcome, attempted = _batch(
        fake_s3, {1: False, 2: False, 3: False, 4: False, 5: False, 6: True}
    )

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]
    assert sorted(outcome.results) == [1, 2, 3, 4, 5]


def test_images_outage_failures_open_the_circuit(fake_s3):
    errors = ["images_outage", "images_missing_url"]
    outcome, attempted = _batch(fake_s3, dict.fromkeys(range(1, 7), errors))

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


def test_a_failed_result_with_no_error_codes_counts_as_a_failure(fake_s3):
    outcome, attempted = _batch(fake_s3, {1: [], 2: [], 3: [], 4: [], 5: [], 6: []})

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


def test_a_success_resets_the_failure_streak(fake_s3):
    script = {i: i == 5 for i in range(1, 10)}  # 4 failures, 1 ok, 4 failures

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped is None
    assert attempted == list(range(1, 10))


def test_skipped_sources_neither_count_nor_reset_the_failure_streak(fake_s3):
    _mark(fake_s3, 2)
    script: dict[int, bool | str | list[str] | Exception] = {
        1: False,
        2: True,
        3: False,
        4: False,
        5: False,
    }
    script.update({6: False, 7: True})

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 3, 4, 5, 6]


def test_a_source_whose_download_raises_counts_as_a_failure(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = {
        1: False,
        2: False,
        3: RuntimeError("boom"),
    }
    script.update({4: False, 5: False, 6: True})

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


def test_a_raised_exception_with_a_lowercase_class_name_counts_as_a_failure(fake_s3):
    class gaierror(Exception):  # noqa: N801
        pass

    script: dict[int, bool | str | list[str] | Exception] = {
        i: gaierror("dns") for i in range(1, 7)
    }

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


_PER_IMAGE = ["images_missing_url"]


def test_sources_failing_only_with_per_image_codes_never_open_the_circuit(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = {
        i: _PER_IMAGE if i % 2 else ["images_upload_failed"] for i in range(1, 6)
    }

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped is None
    assert attempted == [1, 2, 3, 4, 5]


def test_a_per_image_only_failure_does_not_reset_the_outage_streak(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = {1: False, 2: False, 3: _PER_IMAGE}
    script.update({4: False, 5: False, 6: False, 7: True})

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5, 6]


def test_per_image_codes_mixed_with_another_error_still_count_as_a_failure(fake_s3):
    mixed = ["images_missing_url", "annotations_failed"]
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 7), mixed)

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


def test_sources_failing_verification_do_not_open_the_circuit(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(
        range(1, 7), "verification_failed"
    )
    script[7] = True

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped is None
    assert attempted == list(range(1, 8))


def test_a_verification_failure_does_not_reset_the_outage_streak(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = {
        1: "annotations_failed",
        2: "image_list_failed",
        3: "verification_failed",
        4: "login failed",
        5: "overview fetch failed: boom",
        6: "annotations_failed",
        7: True,
    }

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "error",
    [
        "cannot access source 1: Permission denied",
        "cannot access source 1: HTTP 404",
    ],
)
def test_sources_that_are_inaccessible_do_not_open_the_circuit(fake_s3, error):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 7), error)

    outcome, attempted = _batch(fake_s3, script)

    assert outcome.stopped is None
    assert attempted == list(range(1, 7))


@pytest.mark.parametrize("code", ["overview_forbidden", "overview_total_missing"])
def test_overview_blocks_and_missing_totals_open_the_circuit(fake_s3, code):
    outcome, attempted = _batch(fake_s3, dict.fromkeys(range(1, 8), code))

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]


def test_a_cooldown_then_a_success_resets_the_streak_and_the_run_continues(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 6), False)
    script.update({6: True, 7: True})

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped is None
    assert attempted == [1, 2, 3, 4, 5, 6, 7]
    assert sleep.call_args_list == [call(600)]


def test_a_cooldown_then_an_outage_stops_the_run(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 7), False)
    script[7] = True

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5, 6]
    assert sleep.call_args_list == [call(600)]


def test_a_raising_skip_check_while_a_trip_is_pending_stops_the_run(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 6), False)
    script.update({6: True, 7: True})
    real = bd.should_skip_download

    def flaky(*, source_id, **kwargs):
        if source_id == 6:
            raise RuntimeError("s3 down")
        return real(source_id=source_id, **kwargs)

    with (
        patch.object(bd, "should_skip_download", side_effect=flaky),
        patch.object(bd.time, "sleep") as sleep,
    ):
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]
    assert 7 not in outcome.results
    assert sleep.call_args_list == []


def test_each_trip_gets_its_own_cooldown(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 12), False)
    script.update({6: True, 12: True})

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped is None
    assert attempted == list(range(1, 13))
    assert sleep.call_args_list == [call(600), call(600)]


def test_a_trip_on_the_last_source_stops_without_a_cooldown(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 6), False)

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]
    assert sleep.call_args_list == []


def test_a_trip_followed_only_by_skipped_sources_stops_without_a_cooldown(fake_s3):
    _mark(fake_s3, 6)
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 6), False)
    script[6] = True

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5]
    assert sleep.call_args_list == []


def test_a_non_outage_failure_while_probing_keeps_probing(fake_s3):
    script: dict[int, bool | str | list[str] | Exception] = dict.fromkeys(range(1, 6), False)
    script.update({6: "verification_failed", 7: False, 8: True})

    with patch.object(bd.time, "sleep") as sleep:
        outcome, attempted = _batch(fake_s3, script, cooldown=600)

    assert outcome.stopped == "circuit_open"
    assert attempted == [1, 2, 3, 4, 5, 6, 7]
    assert sleep.call_args_list == [call(600)]
