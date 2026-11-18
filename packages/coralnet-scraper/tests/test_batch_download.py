"""Tests for batch download orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

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
