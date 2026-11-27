"""Tests for the S3 helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from coralnet_scraper import s3_io


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "m"}}, "GetObject")


@pytest.mark.parametrize("code", ["404", "NoSuchKey"])
def test_read_csv_s3_returns_none_when_key_missing(code):
    client = MagicMock()
    client.get_object.side_effect = _client_error(code)
    assert s3_io.read_csv_s3(client, "bkt", "k.csv") is None


def test_read_csv_s3_propagates_other_client_errors():
    client = MagicMock()
    client.get_object.side_effect = _client_error("AccessDenied")
    with pytest.raises(ClientError):
        s3_io.read_csv_s3(client, "bkt", "k.csv")


@pytest.mark.parametrize("other_key", ["p/s10/_complete.json", "p/s1/_complete.json.bak"])
def test_key_exists_matches_only_the_exact_key(fake_s3, other_key):
    fake_s3.put("b", other_key, b"{}")

    assert s3_io.key_exists(fake_s3, "b", "p/s1/_complete.json") is False


def test_delete_prefix_removes_every_page_of_keys_under_the_prefix_only(fake_s3):
    fake_s3.list_page_size = 2
    inside = [f"p/s1/_partial/page-{n}.csv" for n in range(1, 6)]
    outside = ["p/s1/annotations.csv", "p/s1/_partial", "p/s10/_partial/state.json"]
    for key in inside + outside:
        fake_s3.put("b", key, b"x")

    s3_io.delete_prefix(fake_s3, "b", "p/s1/_partial/")

    assert sorted(k for (_b, k) in fake_s3.objects) == sorted(outside)


def test_delete_keys_batches_at_1000_and_removes_every_key(fake_s3):
    keys = [f"p/s1/images/{n}.jpg" for n in range(2001)]
    for key in keys:
        fake_s3.put("b", key, b"x")
    fake_s3.put("b", "p/s1/keep.csv", b"x")

    s3_io.delete_keys(fake_s3, "b", keys)

    assert fake_s3.delete_objects_calls == [1000, 1000, 1]
    assert list(fake_s3.objects) == [("b", "p/s1/keep.csv")]


def test_delete_keys_raises_naming_the_first_failed_key_and_the_count(fake_s3):
    fake_s3.delete_errors = {"p/b": "AccessDenied", "p/c": "InternalError"}
    for key in ("p/a", "p/b", "p/c"):
        fake_s3.put("b", key, b"x")

    with pytest.raises(RuntimeError, match=r"p/b.*AccessDenied.*2|2.*p/b.*AccessDenied"):
        s3_io.delete_keys(fake_s3, "b", ["p/a", "p/b", "p/c"])
