"""Tests for the local-first S3 cache layer."""

from __future__ import annotations

import threading
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from mermaidseg.datasets.local_cache import (
    LocalS3Cache,
    create_cache_stats,
    parse_storage_ref,
)


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    LocalS3Cache.reset_singleton()
    yield
    LocalS3Cache.reset_singleton()


def _mock_s3(data: bytes = b"image-bytes") -> MagicMock:
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = data
    client.get_object.return_value = {"Body": body}
    return client


def test_parse_storage_ref_s3_uri():
    assert parse_storage_ref(
        "s3://my-bucket/path/to/file.parquet", "default"
    ) == ("my-bucket", "path/to/file.parquet")


def test_parse_storage_ref_relative_key():
    assert parse_storage_ref("mermaid/foo.parquet", "coral-reef-training") == (
        "coral-reef-training",
        "mermaid/foo.parquet",
    )


def test_parse_storage_ref_absolute_local():
    assert parse_storage_ref("/tmp/local.parquet", "bucket") is None


def test_local_hit_skips_s3(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)
    local = cache.local_path("bucket", "img.png")
    local.parent.mkdir(parents=True)
    local.write_bytes(b"cached")

    mock_s3 = _mock_s3()
    cache.set_s3_client(mock_s3)

    result = cache.read_bytes("bucket", "img.png")
    assert result == b"cached"
    mock_s3.get_object.assert_not_called()

    snapshot = cache.snapshot_stats()
    assert snapshot.local_hits == 1
    assert snapshot.s3_fetches == 0


def test_s3_miss_write_through(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, write_through=True, stats=stats)
    mock_s3 = _mock_s3(b"from-s3")
    cache.set_s3_client(mock_s3)

    assert cache.read_bytes("bucket", "img.png") == b"from-s3"
    mock_s3.get_object.assert_called_once_with(Bucket="bucket", Key="img.png")
    assert cache.local_path("bucket", "img.png").read_bytes() == b"from-s3"

    mock_s3.get_object.reset_mock()
    assert cache.read_bytes("bucket", "img.png") == b"from-s3"
    mock_s3.get_object.assert_not_called()

    snapshot = cache.snapshot_stats()
    assert snapshot.s3_fetches == 1
    assert snapshot.local_hits == 1


def test_concurrent_miss_single_s3_fetch(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, write_through=True, stats=stats)
    call_count = 0
    count_lock = threading.Lock()

    def slow_get_object(**_kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        body = MagicMock()
        body.read.return_value = b"synced"
        return {"Body": body}

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = slow_get_object
    cache.set_s3_client(mock_s3)

    results: list[bytes] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(cache.read_bytes("bucket", "race.png"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert sorted(results) == [b"synced", b"synced"]
    assert call_count == 1


def test_snapshot_stats_resets_counters(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)
    local = cache.local_path("bucket", "a.png")
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    cache.read_bytes("bucket", "a.png")

    first = cache.snapshot_stats()
    assert first.local_hits == 1

    cache.read_bytes("bucket", "a.png")
    second = cache.snapshot_stats()
    assert second.local_hits == 1

    third = cache.snapshot_stats()
    assert third.local_hits == 0
    assert third.s3_fetches == 0


def test_read_parquet_round_trip(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, write_through=True, stats=stats)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    mock_s3 = _mock_s3(parquet_bytes)
    cache.set_s3_client(mock_s3)

    loaded = cache.read_parquet("bucket", "data.parquet")
    pd.testing.assert_frame_equal(loaded, df)
    assert cache.local_path("bucket", "data.parquet").is_file()

    mock_s3.get_object.reset_mock()
    loaded_again = cache.read_parquet("bucket", "data.parquet")
    pd.testing.assert_frame_equal(loaded_again, df)
    mock_s3.get_object.assert_not_called()


def test_read_parquet_ref_local_absolute(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)
    path = tmp_path / "direct.parquet"
    df = pd.DataFrame({"x": [1]})
    df.to_parquet(path, index=False)

    loaded = cache.read_parquet_ref(str(path), "ignored-bucket")
    pd.testing.assert_frame_equal(loaded, df)


def test_disabled_cache_fetches_s3_without_writing(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(None, stats=stats)
    mock_s3 = _mock_s3(b"remote")
    cache.set_s3_client(mock_s3)

    assert cache.read_bytes("bucket", "key") == b"remote"
    mock_s3.get_object.assert_called_once()
    assert not cache.enabled

    snapshot = cache.snapshot_stats()
    assert snapshot.s3_fetches == 1


def test_s3_error_propagates(tmp_path):
    cache = LocalS3Cache.configure(tmp_path, write_through=True)
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
        "GetObject",
    )
    cache.set_s3_client(mock_s3)

    with pytest.raises(ClientError):
        cache.read_bytes("bucket", "missing.png")
