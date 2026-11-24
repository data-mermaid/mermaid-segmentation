"""Server-side image copies into a write-once destination, verified on a repeat run."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from botocore.exceptions import ClientError

from segdata_spec._s3 import is_precondition_failed
from segdata_spec.errors import ObjectExists
from segdata_spec.sync import _IN_FLIGHT_PER_WORKER


@dataclass(frozen=True)
class CopyItem:
    """One object to copy from ``source_bucket/source_key`` to ``dest_key`` in the destination."""

    source_bucket: str
    source_key: str
    dest_key: str


@dataclass(frozen=True)
class CopiedObject:
    """The destination object's quoted ETag, byte size and base64 CRC64NVME (None if absent)."""

    etag: str
    size: int
    crc64nvme: str | None


def _md5(client: Any, bucket: str, key: str) -> str:
    data = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _same_content(
    client: Any,
    source_client: Any,
    item: CopyItem,
    dest_bucket: str,
    source_head: dict[str, Any],
    dest_head: dict[str, Any],
) -> bool:
    """Whether the existing destination holds the source's bytes.

    Compares size and the full-object CRC64NVME; when either object lacks a CRC, compares the MD5
    of both objects.
    """
    if dest_head["ContentLength"] != source_head["ContentLength"]:
        return False
    source_crc = source_head.get("ChecksumCRC64NVME")
    dest_crc = dest_head.get("ChecksumCRC64NVME")
    if source_crc is not None and dest_crc is not None:
        return bool(dest_crc == source_crc)
    return _md5(source_client, item.source_bucket, item.source_key) == _md5(
        client, dest_bucket, item.dest_key
    )


def _copied_size(
    client: Any,
    item: CopyItem,
    dest_bucket: str,
    source_head: dict[str, Any],
    result: dict[str, Any],
) -> int:
    """The size of the object the copy wrote, taken from the destination.

    A source CRC that differs from the copy's CRC means the source changed after its HEAD.
    """
    source_crc = source_head.get("ChecksumCRC64NVME")
    if source_crc is not None:
        if result.get("ChecksumCRC64NVME") != source_crc:
            raise RuntimeError(
                f"s3://{item.source_bucket}/{item.source_key} changed while being copied"
            )
        return int(source_head["ContentLength"])
    dest_head = client.head_object(Bucket=dest_bucket, Key=item.dest_key, ChecksumMode="ENABLED")
    return int(dest_head["ContentLength"])


def _copy_one(
    client: Any,
    source_client: Any,
    item: CopyItem,
    *,
    dest_bucket: str,
    storage_class: str | None,
) -> CopiedObject:
    source_head = source_client.head_object(
        Bucket=item.source_bucket, Key=item.source_key, ChecksumMode="ENABLED"
    )
    kwargs: dict[str, Any] = {
        "CopySource": {"Bucket": item.source_bucket, "Key": item.source_key},
        "Bucket": dest_bucket,
        "Key": item.dest_key,
        "IfNoneMatch": "*",
        "ChecksumAlgorithm": "CRC64NVME",
        # REPLACE writes the copy untagged, so no tagging permissions are needed.
        "TaggingDirective": "REPLACE",
    }
    if storage_class is not None:
        kwargs["StorageClass"] = storage_class
    try:
        result = client.copy_object(**kwargs)["CopyObjectResult"]
    except ClientError as exc:
        if not is_precondition_failed(exc):
            raise
        dest_head = client.head_object(
            Bucket=dest_bucket, Key=item.dest_key, ChecksumMode="ENABLED"
        )
        if not _same_content(client, source_client, item, dest_bucket, source_head, dest_head):
            raise ObjectExists(
                f"s3://{dest_bucket}/{item.dest_key} already exists with different bytes"
            ) from exc
        return CopiedObject(
            etag=dest_head["ETag"],
            size=dest_head["ContentLength"],
            crc64nvme=dest_head.get("ChecksumCRC64NVME"),
        )
    return CopiedObject(
        etag=result["ETag"],
        size=_copied_size(client, item, dest_bucket, source_head, result),
        crc64nvme=result.get("ChecksumCRC64NVME"),
    )


def _windowed_futures(
    pool: ThreadPoolExecutor,
    work: Any,
    items: Sequence[CopyItem],
    window: int,
) -> Iterator[tuple[CopyItem, Future[CopiedObject]]]:
    """Yield ``(item, future)`` in item order, with at most ``window`` items submitted at once."""
    pending: deque[tuple[CopyItem, Future[CopiedObject]]] = deque()
    for item in items:
        pending.append((item, pool.submit(work, item)))
        if len(pending) >= window:
            yield pending.popleft()
    while pending:
        yield pending.popleft()


def copy_images(
    client: Any,
    items: Sequence[CopyItem],
    *,
    dest_bucket: str,
    workers: int,
    storage_class: str | None = None,
    source_client: Any = None,
) -> dict[str, CopiedObject]:
    """Copy each item into ``dest_bucket`` with a conditional ``copy_object``; key by dest key.

    ``client`` serves the destination and the copy; ``source_client`` (default ``client``) serves
    the source HEAD and the MD5 fallback read. A destination that already holds the source's bytes
    (a resumed run) counts as copied. The managed ``client.copy`` is not used: a multipart
    completion carries no ``IfNoneMatch``.

    Raises:
        ObjectExists: A destination key exists with different bytes.
        RuntimeError: A source changed between its HEAD and the copy.
        botocore.exceptions.ClientError: Any S3 failure other than a 412 on the copy.
    """
    reader = client if source_client is None else source_client
    copied: dict[str, CopiedObject] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            work = partial(
                _copy_one, client, reader, dest_bucket=dest_bucket, storage_class=storage_class
            )
            for item, future in _windowed_futures(
                pool, work, items, _IN_FLIGHT_PER_WORKER * workers
            ):
                copied[item.dest_key] = future.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    return copied
