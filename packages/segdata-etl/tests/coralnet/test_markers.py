"""Pinned sources: the sources whose ``_complete.json`` carries a ``parent`` pin."""

from __future__ import annotations

import json

import pytest
from s3_fake import FakeS3

from segdata_etl.datasets.coralnet.markers import pinned_source_ids

BUCKET = "src"
PREFIX = "coralnet-public-images"
PIN = {"dataset": "coralnet-raw", "version": "v1", "manifest_sha256": "abc"}


def _marker_key(source_id: int) -> str:
    return f"{PREFIX}/s{source_id}/_complete.json"


def test_only_a_source_whose_marker_has_a_parent_key_is_pinned() -> None:
    s3 = FakeS3()
    s3.put(BUCKET, _marker_key(1), json.dumps({"parent": PIN}).encode())
    s3.put(BUCKET, _marker_key(2), json.dumps({"count": 5}).encode())

    pinned = pinned_source_ids(s3, BUCKET, PREFIX, [1, 2, 3], workers=2)

    assert pinned == frozenset({1})


def test_a_malformed_marker_raises_naming_its_source() -> None:
    s3 = FakeS3()
    s3.put(BUCKET, _marker_key(1), json.dumps({"parent": PIN}).encode())
    s3.put(BUCKET, _marker_key(7), b"{not json")

    with pytest.raises(RuntimeError, match="source 7"):
        pinned_source_ids(s3, BUCKET, PREFIX, [1, 7], workers=1)
