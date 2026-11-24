from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from s3_fake import FakeS3, quoted_etag

from segdata_etl.common import stage
from segdata_etl.common.s3_objects import put_new
from segdata_etl.common.stage import LocalDirS3, next_version, refuse_unpublishable
from segdata_spec._s3 import is_not_found, is_precondition_failed
from segdata_spec.publish import GitState, PublishRefused


def test_local_put_round_trips_to_get_and_head(tmp_path):
    client = LocalDirS3(tmp_path)
    client.put_object(Bucket="bkt", Key="a/b.bin", Body=b"hello", ContentType="text/plain")
    assert (tmp_path / "bkt" / "a" / "b.bin").read_bytes() == b"hello"
    assert client.get_object(Bucket="bkt", Key="a/b.bin")["Body"].read() == b"hello"
    head = client.head_object(Bucket="bkt", Key="a/b.bin")
    assert head["ContentLength"] == 5
    assert head["ETag"] == quoted_etag(b"hello")


def test_local_put_if_none_match_on_existing_key_is_precondition_failed(tmp_path):
    client = LocalDirS3(tmp_path)
    client.put_object(Bucket="bkt", Key="k", Body=b"one")
    with pytest.raises(ClientError) as info:
        client.put_object(Bucket="bkt", Key="k", Body=b"two", IfNoneMatch="*")
    assert is_precondition_failed(info.value)
    assert (tmp_path / "bkt" / "k").read_bytes() == b"one"


def test_local_missing_key_is_not_found(tmp_path):
    client = LocalDirS3(tmp_path)
    with pytest.raises(ClientError) as head_info:
        client.head_object(Bucket="bkt", Key="missing")
    with pytest.raises(ClientError) as get_info:
        client.get_object(Bucket="bkt", Key="missing")
    assert is_not_found(head_info.value)
    assert is_not_found(get_info.value)


def test_put_new_through_local_dir_twice_returns_real_crc_then_already_present(tmp_path):
    client = LocalDirS3(tmp_path)
    kwargs = {"bucket": "bkt", "key": "k.json", "body": b"{}", "content_type": "application/json"}
    first = put_new(client, **kwargs)
    second = put_new(client, **kwargs)
    assert (first.crc64nvme, first.already_present) == ("Sv1wpwepjTw=", False)
    assert (second.crc64nvme, second.already_present) == ("Sv1wpwepjTw=", True)


def test_next_version_defaults_to_the_utc_date(tmp_path: Path, monkeypatch):
    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            # A naive now() is host-local 23:30, the previous day in UTC; a UTC now() is the next day.
            return (
                datetime(2026, 11, 24, 4, 30, tzinfo=UTC)
                if tz is not None
                else datetime(2026, 11, 23, 23, 30)
            )

    monkeypatch.setattr(stage, "datetime", _Clock)
    assert next_version(tmp_path, "bkt", "ds", git_sha="abc1234def5678") == "20261124-abc1234"


def test_next_version_appends_first_free_counter(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678"}
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234"
    base = tmp_path / "bkt" / "datasets" / "ds"
    (base / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    (base / "20261123-abc1234-2").mkdir()
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_counts_a_version_published_in_the_bucket_as_taken(tmp_path: Path):
    client = FakeS3()
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": client}
    client.put("bkt", "datasets/ds/20261123-abc1234/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"
    client.put("bkt", "datasets/ds/20261123-abc1234-2/manifest.json", b"{}")
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-3"


def test_next_version_with_a_client_still_honours_the_staged_dir(tmp_path: Path):
    kwargs = {"today": date(2026, 11, 23), "git_sha": "abc1234def5678", "client": FakeS3()}
    (tmp_path / "bkt" / "datasets" / "ds" / "20261123-abc1234").mkdir(parents=True)
    assert next_version(tmp_path, "bkt", "ds", **kwargs) == "20261123-abc1234-2"


def test_refuse_unpublishable_refuses_a_dirty_tree():
    with pytest.raises(PublishRefused, match="uncommitted changes to tracked files"):
        refuse_unpublishable(GitState(sha="abc", dirty=True, pushed=True))


def test_refuse_unpublishable_refuses_an_unpushed_head():
    with pytest.raises(PublishRefused, match="HEAD abc is not contained in any remote branch"):
        refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=False))


def test_refuse_unpublishable_accepts_a_clean_pushed_tree():
    assert refuse_unpublishable(GitState(sha="abc", dirty=False, pushed=True)) is None
