"""Pinned CoralNet sources: those whose ``_complete.json`` marker carries a ``parent`` pin.

A pinned source is complete given a published ``coralnet-raw`` version: some of its images live
only in that version, not in the scrape bucket. The marker's ``parent`` entry names that version
and carries string ``dataset``, ``version`` and ``manifest_sha256`` fields; any other ``parent``
value, ``null`` included, is a malformed marker.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from botocore.exceptions import ClientError

from coralnet_scraper.s3_io import object_key_complete_marker, read_json_s3

_PARENT_KEY = "parent"
_PARENT_FIELDS = ("dataset", "version", "manifest_sha256")


def _is_valid_parent(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(f), str) for f in _PARENT_FIELDS)


def _parent_of(client: Any, bucket: str, prefix: str, source_id: int) -> dict[str, Any] | None:
    """The source marker's ``parent`` entry; ``None`` for a missing marker or no ``parent`` key.

    Raises:
        RuntimeError: The marker is unreadable (an S3 error other than not-found, malformed
            JSON, or a JSON document that is not an object), or its ``parent`` is not an object
            with string ``dataset``, ``version`` and ``manifest_sha256``.
    """
    key = object_key_complete_marker(prefix, source_id)
    try:
        document = read_json_s3(client, bucket, key)
    except (ClientError, ValueError) as e:
        raise RuntimeError(f"source {source_id}: cannot read s3://{bucket}/{key}: {e}") from e
    if document is None:
        return None
    if not isinstance(document, dict):
        raise RuntimeError(f"source {source_id}: s3://{bucket}/{key} is not a JSON object")
    if _PARENT_KEY not in document:
        return None
    parent = document[_PARENT_KEY]
    if not _is_valid_parent(parent):
        raise RuntimeError(
            f"source {source_id}: s3://{bucket}/{key} has a malformed parent: {parent!r}"
        )
    return parent


def pinned_parents(
    client: Any, bucket: str, prefix: str, source_ids: Iterable[int], *, workers: int
) -> dict[int, dict[str, Any]]:
    """Map each pinned id among ``source_ids`` to the ``parent`` entry its marker records.

    A source with no marker, or a marker without a ``parent`` key, is absent. Markers are read
    on ``workers`` threads.

    Raises:
        RuntimeError: A marker is unreadable or has a malformed ``parent``; the message names
            the source id.
    """
    ids = list(source_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parents = list(pool.map(lambda sid: _parent_of(client, bucket, prefix, sid), ids))
    return {sid: parent for sid, parent in zip(ids, parents, strict=True) if parent is not None}
