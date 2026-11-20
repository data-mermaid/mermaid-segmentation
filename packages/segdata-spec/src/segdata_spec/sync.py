"""Download pinned dataset and taxonomy versions into a local cache, verifying every file."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import awscrt.checksums
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_spec._s3 import (
    MANIFEST_NAME,
    cache_path,
    dataset_prefix,
    is_not_found,
    is_precondition_failed,
    taxonomy_prefix,
)
from segdata_spec.errors import (
    ChecksumMismatch,
    ImageChanged,
    ImageMissing,
    PinMismatch,
    SegdataError,
)
from segdata_spec.manifest import (
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
    TaxonomyManifest,
)

MissingMode = Literal["fail", "drop"]

_IMAGE_COLUMNS = [
    "image_id",
    "key",
    "etag",
    "size",
    "crc64nvme",
    "mask_key",
    "mask_etag",
    "mask_size",
    "mask_crc64nvme",
]


@dataclass(frozen=True)
class DatasetSyncResult:
    """What ``sync`` did for one dataset pin; ``images_dropped`` feeds model lineage."""

    version: str
    manifest_sha256: str
    images_dropped: int
    dropped_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class SyncResult:
    """Per-dataset results, keyed like the ``pins`` argument, and the taxonomy pin synced."""

    datasets: dict[str, DatasetSyncResult]
    taxonomy: Pin | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    # O_EXCL refuses an existing name; the kernel applies the process umask to the 0o666 mode.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fetch_file(
    client: Any,
    cache_dir: Path,
    bucket: str,
    key: str,
    expected_sha256: str,
    error: type[SegdataError],
) -> bytes:
    """Return the file's bytes from the cache or S3, raising ``error`` on a sha256 mismatch.

    A cached file is verified and never re-downloaded; a downloaded file is cached only once it
    verifies.
    """
    path = cache_path(cache_dir, bucket, key)
    if path.exists():
        data = path.read_bytes()
        if _sha256(data) != expected_sha256:
            raise error(f"cached file {path} does not match its expected sha256")
        return data
    data = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if _sha256(data) != expected_sha256:
        raise error(f"s3://{bucket}/{key} does not match its expected sha256")
    _write_atomic(path, data)
    return bytes(data)


def _fetch_tables(
    client: Any, cache_dir: Path, bucket: str, prefix: str, files: Mapping[str, FileEntry]
) -> dict[str, bytes]:
    return {
        name: _fetch_file(client, cache_dir, bucket, prefix + name, entry.sha256, ChecksumMismatch)
        for name, entry in files.items()
    }


def _crc64nvme_b64(data: bytes) -> str:
    """Return the CRC64NVME in S3's ``ChecksumCRC64NVME`` form: base64 of the big-endian CRC."""
    return base64.b64encode(awscrt.checksums.crc64nvme(data).to_bytes(8, "big")).decode()


def _object_problem(data: bytes, etag: str, size: int, crc64nvme: str | None) -> str | None:
    """Describe how ``data`` disagrees with the recorded size, ETag and CRC64NVME, or return None.

    A multipart ETag (``<hash>-<parts>``) is not an MD5 of the content, so the CRC64NVME is the
    content check for those. A ``crc64nvme`` of None skips the CRC check.
    """
    if len(data) != size:
        return f"size {len(data)} differs from the recorded {size}"
    bare = etag.strip('"')
    if "-" not in bare and hashlib.md5(data, usedforsecurity=False).hexdigest() != bare:
        return f"MD5 differs from the recorded ETag {bare}"
    if crc64nvme is not None:
        computed = _crc64nvme_b64(data)
        if computed != crc64nvme:
            return f"CRC64NVME {computed} differs from the recorded {crc64nvme}"
    return None


def _download_object(client: Any, bucket: str, key: str, etag: str) -> bytes:
    try:
        data: bytes = client.get_object(Bucket=bucket, Key=key, IfMatch=etag)["Body"].read()
    except ClientError as exc:
        if is_precondition_failed(exc):
            raise ImageChanged(f"s3://{bucket}/{key} no longer matches ETag {etag}") from exc
        if is_not_found(exc):
            raise ImageMissing(f"s3://{bucket}/{key} does not exist") from exc
        raise
    return data


def _sync_object(
    client: Any,
    cache_dir: Path,
    bucket: str,
    key: str,
    etag: str,
    size: int,
    crc64nvme: str | None,
) -> None:
    """Verify a cached image or mask, or download, verify and cache it."""
    path = cache_path(cache_dir, bucket, key)
    if path.exists():
        problem = _object_problem(path.read_bytes(), etag, size, crc64nvme)
        if problem:
            raise ChecksumMismatch(f"cached file {path}: {problem}")
        return
    data = _download_object(client, bucket, key, etag)
    problem = _object_problem(data, etag, size, crc64nvme)
    if problem:
        raise ChecksumMismatch(f"downloaded s3://{bucket}/{key}: {problem}")
    _write_atomic(path, data)


def _row_objects(row: Mapping[str, Any]) -> list[tuple[str, str, int, str | None]]:
    """Return ``(key, etag, size, crc64nvme)`` for the row's image and, when it has one, its mask."""
    objects = [(row["key"], row["etag"], row["size"], row["crc64nvme"])]
    if row["mask_key"] is not None:
        objects.append((row["mask_key"], row["mask_etag"], row["mask_size"], row["mask_crc64nvme"]))
    return objects


def _select_groups(images: pa.Table, groups: Collection[str] | None) -> pa.Table:
    if groups is None:
        return images
    value_set = pa.array(sorted(groups), pa.string())
    return images.filter(pc.is_in(images.column("group"), value_set=value_set))


def _check_selection(
    selection: Mapping[str, Collection[str] | None], images: Mapping[str, pa.Table]
) -> None:
    """Raise ``ValueError`` naming each selected dataset or group that does not exist."""
    unknown = sorted(set(selection) - set(images))
    if unknown:
        raise ValueError(f"selection names datasets that are not pinned: {unknown}")
    for name, groups in selection.items():
        if groups is None:
            continue
        present = set(pc.unique(images[name].column("group")).to_pylist())
        absent = sorted(set(groups) - present)
        if absent:
            raise ValueError(f"selection names groups absent from dataset {name!r}: {absent}")


def _sync_images(
    client: Any, cache_dir: Path, store: ImageStore, images: pa.Table, mode: MissingMode
) -> tuple[str, ...]:
    """Sync every image and mask in ``images``; return the ids dropped as missing."""
    dropped: list[str] = []
    for row in images.select(_IMAGE_COLUMNS).to_pylist():
        try:
            for key, etag, size, crc in _row_objects(row):
                _sync_object(client, cache_dir, store.bucket, store.prefix + key, etag, size, crc)
        except ImageMissing:
            if mode == "fail":
                raise
            dropped.append(row["image_id"])
    return tuple(dropped)


def _images_table(table_bytes: Mapping[str, bytes]) -> pa.Table:
    return pq.read_table(io.BytesIO(table_bytes["images.parquet"]))


def _dataset_result(pin: Pin, dropped: tuple[str, ...]) -> DatasetSyncResult:
    return DatasetSyncResult(
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        images_dropped=len(dropped),
        dropped_image_ids=dropped,
    )


def _fetch_manifest(client: Any, cache_dir: Path, bucket: str, prefix: str, pin: Pin) -> bytes:
    return _fetch_file(
        client, cache_dir, bucket, prefix + MANIFEST_NAME, pin.manifest_sha256, PinMismatch
    )


def sync(
    client: Any,
    pins: Mapping[str, Pin],
    cache_dir: Path,
    *,
    bucket: str,
    selection: Mapping[str, Collection[str] | None] | None = None,
    on_missing_images: Mapping[str, MissingMode] | None = None,
    check_split: Callable[[str, set[str]], None] | None = None,
    taxonomy: Pin | None = None,
) -> SyncResult:
    """Sync pinned versions into ``<cache_dir>/<bucket>/<key>``, verifying every file.

    Every manifest, then every table, then the selection check, then ``check_split`` per dataset,
    all precede the first image. A dataset absent from ``selection`` syncs all groups; one absent
    from ``on_missing_images`` fails on a missing image.

    Raises:
        PinMismatch: A manifest, downloaded or cached, differs from its pin.
        ChecksumMismatch: A table, image or mask, downloaded or cached, differs from its record.
        ImageChanged: An image or mask no longer matches its recorded ETag.
        ImageMissing: An image or mask is absent and its dataset's mode is ``fail``.
        ValueError: ``selection`` names a dataset not in ``pins`` or a group absent from its
            dataset (raised before any image), or a published key would leave the cache.
    """
    selection = selection or {}
    on_missing_images = on_missing_images or {}
    manifests = {
        name: Manifest.model_validate_json(
            _fetch_manifest(client, cache_dir, bucket, dataset_prefix(pin.name, pin.version), pin)
        )
        for name, pin in pins.items()
    }
    tax_files: Mapping[str, FileEntry] = {}
    if taxonomy is not None:
        tax_bytes = _fetch_manifest(
            client, cache_dir, bucket, taxonomy_prefix(taxonomy.name, taxonomy.version), taxonomy
        )
        tax_files = TaxonomyManifest.model_validate_json(tax_bytes).files
    images = {
        name: _images_table(
            _fetch_tables(
                client,
                cache_dir,
                bucket,
                dataset_prefix(pin.name, pin.version),
                manifests[name].files,
            )
        )
        for name, pin in pins.items()
    }
    if taxonomy is not None:
        _fetch_tables(
            client, cache_dir, bucket, taxonomy_prefix(taxonomy.name, taxonomy.version), tax_files
        )
    _check_selection(selection, images)
    if check_split is not None:
        for name, table in images.items():
            check_split(name, set(pc.unique(table.column("group")).to_pylist()))
    results = {
        name: _dataset_result(
            pin,
            _sync_images(
                client,
                cache_dir,
                manifests[name].image_store,
                _select_groups(images[name], selection.get(name)),
                on_missing_images.get(name, "fail"),
            ),
        )
        for name, pin in pins.items()
    }
    return SyncResult(datasets=results, taxonomy=taxonomy)
