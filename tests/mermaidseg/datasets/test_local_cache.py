"""Tests for the local-first S3 cache layer."""

from __future__ import annotations

import os
import threading
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from mermaidseg.datasets import local_cache as local_cache_module
from mermaidseg.datasets.local_cache import (
    DataLoadError,
    LocalS3Cache,
    _env_flag,
    create_cache_stats,
    parse_storage_ref,
    setup_local_cache,
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("  ", False),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        (" Off ", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_env_flag_is_false_for_unset_blank_and_falsy_words(monkeypatch, value, expected):
    monkeypatch.delenv("MERMAIDSEG_TEST_FLAG", raising=False)
    if value is not None:
        monkeypatch.setenv("MERMAIDSEG_TEST_FLAG", value)

    assert _env_flag("MERMAIDSEG_TEST_FLAG") is expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("s3://my-bucket/path/to/file.parquet", ("my-bucket", "path/to/file.parquet")),
        ("mermaid/foo.parquet", ("coral-reef-training", "mermaid/foo.parquet")),
        ("/leading/slash/is/absolute", None),
    ],
)
def test_parse_storage_ref_splits_s3_uris_keys_and_local_paths(ref, expected):
    assert parse_storage_ref(ref, "coral-reef-training") == expected


@pytest.mark.parametrize("ref", ["s3://bucket-only", "s3://bucket/", "s3:///key-only", "s3://"])
def test_parse_storage_ref_rejects_an_s3_uri_missing_bucket_or_key(ref):
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        parse_storage_ref(ref, "default")


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


def test_load_stage_stats_flush_and_snapshot(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)
    cache.record_stage("image_load", 0.25)
    cache.record_stage("mask", 0.05)
    cache.record_retries(2)
    cache._flush_every = 1
    cache.record_sample()

    # The periodic flush alone reaches the shared counters, before any snapshot.
    assert stats.samples.value == 1
    assert stats.retries.value == 2
    assert stats.stage_sec["image_load"].value == pytest.approx(0.25)

    snapshot = cache.snapshot_stats()
    assert snapshot.samples == 1
    assert snapshot.retries == 2
    assert snapshot.stage_sec["image_load"] == pytest.approx(0.25)
    assert snapshot.stage_sec["mask"] == pytest.approx(0.05)

    reset = cache.snapshot_stats()
    assert reset.samples == 0
    assert reset.retries == 0
    assert reset.stage_sec["image_load"] == 0.0


def test_record_sample_flushes_only_once_flush_every_samples_are_counted(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)
    cache._flush_every = 3

    cache.record_sample()
    cache.record_sample()
    assert stats.samples.value == 0

    cache.record_sample()
    assert stats.samples.value == 3


def test_record_retries_ignores_zero_and_negative_counts(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats)

    cache.record_retries(0)
    cache.record_retries(-3)

    assert cache.snapshot_stats().retries == 0


def test_local_path_without_a_root_raises():
    cache = LocalS3Cache.configure(None)

    with pytest.raises(RuntimeError, match="not configured"):
        cache.local_path("bucket", "key")


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (3, 2), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_read_pil_image_decodes_the_object(tmp_path):
    cache = LocalS3Cache.configure(tmp_path)
    mock_s3 = _mock_s3(_png_bytes())
    cache.set_s3_client(mock_s3)

    image = cache.read_pil_image("bucket", "dir/a.png")

    mock_s3.get_object.assert_called_once_with(Bucket="bucket", Key="dir/a.png")
    assert image.size == (3, 2)


def test_read_pil_image_thumbnail_rewrites_the_png_key(tmp_path):
    cache = LocalS3Cache.configure(tmp_path)
    mock_s3 = _mock_s3(_png_bytes())
    cache.set_s3_client(mock_s3)

    cache.read_pil_image("bucket", "dir/a.png", thumbnail=True)

    mock_s3.get_object.assert_called_once_with(Bucket="bucket", Key="dir/a_thumbnail.png")


def test_read_pil_image_wraps_a_client_error_in_data_load_error(tmp_path):
    cache = LocalS3Cache.configure(tmp_path)
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "gone"}}, "GetObject"
    )
    cache.set_s3_client(mock_s3)

    with pytest.raises(DataLoadError, match="NoSuchKey") as excinfo:
        cache.read_pil_image("bucket", "a.png")

    assert isinstance(excinfo.value.__cause__, ClientError)


def test_read_pil_image_wraps_corrupt_bytes_in_data_load_error(tmp_path):
    cache = LocalS3Cache.configure(tmp_path)
    cache.set_s3_client(_mock_s3(b"not an image"))

    with pytest.raises(DataLoadError, match="cannot open image"):
        cache.read_pil_image("bucket", "a.png")


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


def test_offline_local_hit_skips_s3(tmp_path):
    stats = create_cache_stats()
    cache = LocalS3Cache.configure(tmp_path, stats=stats, offline=True)
    local = cache.local_path("bucket", "img.png")
    local.parent.mkdir(parents=True)
    local.write_bytes(b"cached")

    mock_s3 = _mock_s3()
    cache.set_s3_client(mock_s3)

    assert cache.read_bytes("bucket", "img.png") == b"cached"
    mock_s3.get_object.assert_not_called()
    assert cache.offline


def test_offline_miss_raises_without_s3_client_or_lock(tmp_path):
    cache = LocalS3Cache.configure(tmp_path, write_through=True, offline=True)

    with pytest.raises(DataLoadError, match="offline mode"):
        cache.read_bytes("bucket", "sub/missing.png")

    # No S3 client should ever be constructed in offline mode.
    assert cache._s3_client is None
    assert not (tmp_path / "bucket" / "sub" / "missing.png.lock").exists()
    assert not (tmp_path / "bucket").exists()


def test_offline_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMAIDSEG_S3_OFFLINE", "1")
    cache = LocalS3Cache.configure(tmp_path)
    assert cache.offline is True

    with pytest.raises(DataLoadError, match="offline mode"):
        cache.read_bytes("bucket", "missing.png")


def test_offline_explicit_flag_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MERMAIDSEG_S3_OFFLINE", "1")
    cache = LocalS3Cache.configure(tmp_path, offline=False)
    assert cache.offline is False


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


def test_setup_local_cache_config_root_beats_env(tmp_path, monkeypatch):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(dir_a))

    setup_local_cache({"local_cache_dir": str(dir_b), "local_cache_write_through": True})

    assert LocalS3Cache.get()._root == dir_b.resolve()
    assert os.environ["MERMAIDSEG_LOCAL_CACHE_DIR"] == str(dir_b.resolve())


@pytest.mark.parametrize(
    "data_cfg",
    [
        {"local_cache_write_through": True},
        {"local_cache_dir": "", "local_cache_write_through": True},
    ],
)
def test_setup_local_cache_without_config_root_ignores_and_clears_env(
    data_cfg, tmp_path, monkeypatch
):
    monkeypatch.setenv("MERMAIDSEG_LOCAL_CACHE_DIR", str(tmp_path / "a"))

    setup_local_cache(data_cfg)

    assert LocalS3Cache.get().enabled is False
    assert "MERMAIDSEG_LOCAL_CACHE_DIR" not in os.environ


def test_read_parquet_with_cache_disabled_offline_raises_without_reading(monkeypatch):
    cache = LocalS3Cache.configure(None, offline=True)
    monkeypatch.setattr(
        local_cache_module.pd, "read_parquet", MagicMock(side_effect=AssertionError("read"))
    )

    with pytest.raises(DataLoadError, match="local cache is disabled"):
        cache.read_parquet("bucket", "data.parquet")


def test_read_parquet_with_cache_disabled_online_reads_the_s3_uri(monkeypatch):
    cache = LocalS3Cache.configure(None, offline=False)
    frame = pd.DataFrame({"a": [1]})
    reader = MagicMock(return_value=frame)
    monkeypatch.setattr(local_cache_module.pd, "read_parquet", reader)

    assert cache.read_parquet("bucket", "dir/data.parquet") is frame
    reader.assert_called_once_with("s3://bucket/dir/data.parquet")
