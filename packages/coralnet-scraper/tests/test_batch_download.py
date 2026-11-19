"""Tests for batch download orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from coralnet_scraper import batch_download as bd


def _skip(fake_s3, source_id: int, *, force: bool) -> bool:
    return bd.should_skip_download(
        source_id=source_id,
        bucket="b",
        s3_prefix="p",
        s3_client=fake_s3,
        force=force,
    )


def test_should_skip_force_never_when_enabled(fake_s3):
    fake_s3.put_object(Bucket="b", Key="p/s1/annotations.csv", Body=b"a,b\n")
    assert _skip(fake_s3, 1, force=True) is False


def test_should_skip_when_annotations_exist(fake_s3):
    fake_s3.put_object(Bucket="b", Key="p/s99/annotations.csv", Body=b"a\n")
    assert _skip(fake_s3, 99, force=False) is True


def test_should_not_skip_when_annotations_missing(fake_s3):
    assert _skip(fake_s3, 7, force=False) is False


def test_run_batch_download_downloads_only_sources_without_annotations(fake_s3):
    fake_s3.put_object(Bucket="b", Key="p/s1/annotations.csv", Body=b"a\n")
    downloader = MagicMock()
    result = MagicMock(skipped_no_confirmed=False)
    downloader.download_source.return_value = result

    results = bd.run_batch_download(
        [1, 2], downloader=downloader, bucket="b", s3_prefix="p", s3_client=fake_s3
    )

    downloader.download_source.assert_called_once_with(2, bucket_name="b", s3_prefix="p")
    assert results == {2: result}


def _run(fake_s3, downloader):
    return bd.run_batch_download(
        [1, 2, 3], downloader=downloader, bucket="b", s3_prefix="p", s3_client=fake_s3
    )


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


def test_run_batch_download_force_downloads_a_source_with_existing_annotations(fake_s3):
    fake_s3.put_object(Bucket="b", Key="p/s1/annotations.csv", Body=b"a\n")
    downloader = MagicMock()
    result = MagicMock(skipped_no_confirmed=False)
    downloader.download_source.return_value = result

    results = bd.run_batch_download(
        [1], downloader=downloader, bucket="b", s3_prefix="p", s3_client=fake_s3, force=True
    )

    downloader.download_source.assert_called_once_with(1, bucket_name="b", s3_prefix="p")
    assert results == {1: result}


_LISTING_HTML = """\
<html><body>
<a href="/about/">About</a>
<a href="/source/100/">Reef A</a>
<a href="https://coralnet.ucsd.edu/source/7/">Reef B</a>
<a href="/source/12/">Reef C</a>
<a href="/source/12/">Reef C again</a>
<a href="mailto:help@coralnet.org?subject=/source/55/x">Contact</a>
</body></html>"""


def _listing_response(text: str, status: int = 200) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = text.encode()
    return resp


def test_discover_public_source_ids_returns_sorted_unique_ids_from_source_links():
    with patch.object(bd.requests, "get", return_value=_listing_response(_LISTING_HTML)) as get:
        ids = bd.discover_public_source_ids()

    assert ids == [7, 12, 100]
    assert get.call_args.args == ("https://coralnet.ucsd.edu/source/about/",)


def test_discover_public_source_ids_raises_on_error_status():
    with (
        patch.object(bd.requests, "get", return_value=_listing_response("", status=503)),
        pytest.raises(requests.HTTPError),
    ):
        bd.discover_public_source_ids()


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
    fake_s3.put_object(Bucket="b", Key="p/s1/annotations.csv", Body=b"a\n")  # source 1 is skipped
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
        )

    assert sleep.call_args_list == expected_sleeps
