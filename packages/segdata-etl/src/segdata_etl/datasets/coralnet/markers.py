"""Pinned CoralNet sources: those whose ``_complete.json`` marker carries a ``parent`` pin.

A pinned source is complete given a published ``coralnet-raw`` version: some of its images live
only in that version, not in the scrape bucket.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from botocore.exceptions import ClientError

from coralnet_scraper.s3_io import object_key_complete_marker, read_json_s3

_PARENT_KEY = "parent"


def _is_pinned(client: Any, bucket: str, prefix: str, source_id: int) -> bool:
    """Whether the source's marker has a ``parent`` key; a missing marker is unpinned.

    Raises:
        RuntimeError: The marker is unreadable: an S3 error other than not-found, malformed
            JSON, or a JSON document that is not an object.
    """
    key = object_key_complete_marker(prefix, source_id)
    try:
        document = read_json_s3(client, bucket, key)
    except (ClientError, ValueError) as e:
        raise RuntimeError(f"source {source_id}: cannot read s3://{bucket}/{key}: {e}") from e
    if document is None:
        return False
    if not isinstance(document, dict):
        raise RuntimeError(f"source {source_id}: s3://{bucket}/{key} is not a JSON object")
    return _PARENT_KEY in document


def pinned_source_ids(
    client: Any, bucket: str, prefix: str, source_ids: Iterable[int], *, workers: int
) -> frozenset[int]:
    """Return the ids among ``source_ids`` whose completion marker has a ``parent`` key.

    Markers are read on ``workers`` threads.

    Raises:
        RuntimeError: A marker is unreadable; the message names the source id.
    """
    ids = list(source_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flags = list(pool.map(lambda sid: _is_pinned(client, bucket, prefix, sid), ids))
    return frozenset(sid for sid, flag in zip(ids, flags, strict=True) if flag)
