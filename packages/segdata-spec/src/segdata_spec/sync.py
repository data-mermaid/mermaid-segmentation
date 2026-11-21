"""Download pinned dataset and taxonomy versions into a local cache, verifying every file."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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
    resized_cache_path,
    resized_index_path,
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
from segdata_spec.resize import ResizeSpec, resize_image, resize_mask, scaled_size

MissingMode = Literal["fail", "drop"]

_IMAGE_COLUMNS = [
    "image_id",
    "key",
    "etag",
    "size",
    "crc64nvme",
    "width",
    "height",
    "mask_key",
    "mask_etag",
    "mask_size",
    "mask_crc64nvme",
]

# Rows submitted to the pool per worker ahead of the row being consumed.
_IN_FLIGHT_PER_WORKER = 4


@dataclass(frozen=True)
class DatasetSyncResult:
    """What ``sync`` did for one dataset pin; ``images_dropped`` and ``resize`` feed lineage.

    ``resize`` is the spec applied to the dataset's images, or None when they were cached as is.
    """

    version: str
    manifest_sha256: str
    images_dropped: int
    dropped_image_ids: tuple[str, ...]
    resize: ResizeSpec | None = None


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


def _index_entry(output: bytes, source_etag: str) -> bytes:
    entry = {"source_etag": source_etag, "size": len(output), "sha256": _sha256(output)}
    return json.dumps(entry, sort_keys=True).encode()


def _resized_problem(output: bytes, entry: Any, source_etag: str) -> str | None:
    """Describe how a resized output disagrees with its index entry and source ETag, or None."""
    if not isinstance(entry, dict):
        return "its index entry is not a JSON object"
    if entry.get("source_etag") != source_etag:
        return f"index entry source ETag {entry.get('source_etag')!r} is not {source_etag!r}"
    if entry.get("size") != len(output):
        return f"size {len(output)} differs from the indexed {entry.get('size')!r}"
    if entry.get("sha256") != _sha256(output):
        return "sha256 differs from the indexed sha256"
    return None


def _verify_resized(output: Path, index: Path, source_etag: str) -> None:
    """Raise ``ChecksumMismatch`` unless ``output`` matches its sidecar index entry."""
    if not index.is_file():
        raise ChecksumMismatch(f"cached file {output} has no index entry at {index}")
    try:
        entry = json.loads(index.read_bytes())
    except ValueError as exc:
        raise ChecksumMismatch(f"cached file {output}: index entry {index} is not JSON") from exc
    problem = _resized_problem(output.read_bytes(), entry, source_etag)
    if problem:
        raise ChecksumMismatch(f"cached file {output}: {problem}")


def _sync_resized_object(
    client: Any,
    cache_dir: Path,
    spec: ResizeSpec,
    obj: tuple[str, str, int, str | None],
    bucket: str,
    transform: Callable[[bytes], bytes],
) -> None:
    """Verify a cached resized object, or download, verify, transform and cache it.

    The sidecar is written before the output, so an output on disk always has its index entry.
    """
    key, etag, size, crc64nvme = obj
    output = resized_cache_path(cache_dir, spec, bucket, key)
    index = resized_index_path(cache_dir, spec, bucket, key)
    if output.exists():
        _verify_resized(output, index, etag)
        return
    data = _download_object(client, bucket, key, etag)
    problem = _object_problem(data, etag, size, crc64nvme)
    if problem:
        raise ChecksumMismatch(f"downloaded s3://{bucket}/{key}: {problem}")
    resized = transform(data)
    _write_atomic(index, _index_entry(resized, etag))
    _write_atomic(output, resized)


def _sync_row(
    client: Any,
    cache_dir: Path,
    store: ImageStore,
    spec: ResizeSpec | None,
    row: Mapping[str, Any],
) -> None:
    """Sync one row's image and mask, resized per ``spec`` or as is when it is None."""
    objects = [(store.prefix + key, etag, size, crc) for key, etag, size, crc in _row_objects(row)]
    if spec is None:
        for key, etag, size, crc in objects:
            _sync_object(client, cache_dir, store.bucket, key, etag, size, crc)
        return
    transforms: list[Callable[[bytes], bytes]] = [partial(resize_image, spec=spec)]
    if row["mask_key"] is not None:
        mask_size = scaled_size(row["width"], row["height"], spec)
        transforms.append(partial(resize_mask, size=mask_size))
    for obj, transform in zip(objects, transforms, strict=True):
        _sync_resized_object(client, cache_dir, spec, obj, store.bucket, transform)


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


def _check_resize(resize: Mapping[str, ResizeSpec | None], pins: Mapping[str, Pin]) -> None:
    unknown = sorted(set(resize) - set(pins))
    if unknown:
        raise ValueError(f"resize names datasets that are not pinned: {unknown}")


def _resolve_max_workers(max_workers: int | None) -> int:
    if max_workers is None:
        return min(8, os.cpu_count() or 1)
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")
    return max_workers


def _ordered_futures(
    pool: ThreadPoolExecutor,
    work: Callable[[dict[str, Any]], None],
    rows: Iterable[dict[str, Any]],
    window: int,
) -> Iterator[tuple[dict[str, Any], Future[None]]]:
    """Yield ``(row, future)`` in row order, with at most ``window`` rows submitted at once."""
    pending: deque[tuple[dict[str, Any], Future[None]]] = deque()
    for row in rows:
        pending.append((row, pool.submit(work, row)))
        if len(pending) >= window:
            yield pending.popleft()
    while pending:
        yield pending.popleft()


def _collect_dropped(
    outcomes: Iterable[tuple[dict[str, Any], Future[None]]], mode: MissingMode
) -> tuple[str, ...]:
    """Wait on each row in order; return the ids dropped as missing, raising any other failure."""
    dropped: list[str] = []
    for row, future in outcomes:
        try:
            future.result()
        except ImageMissing:
            if mode == "fail":
                raise
            dropped.append(row["image_id"])
    return tuple(dropped)


def _sync_images(
    client: Any,
    cache_dir: Path,
    store: ImageStore,
    images: pa.Table,
    mode: MissingMode,
    spec: ResizeSpec | None,
    max_workers: int,
) -> tuple[str, ...]:
    """Sync every image and mask in ``images`` on a worker pool; return the ids dropped as missing.

    Results are consumed in row order, so the error raised and the dropped ids match a serial run.
    """
    work = partial(_sync_row, client, cache_dir, store, spec)
    rows = images.select(_IMAGE_COLUMNS).to_pylist()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        try:
            outcomes = _ordered_futures(pool, work, rows, _IN_FLIGHT_PER_WORKER * max_workers)
            return _collect_dropped(outcomes, mode)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def _images_table(table_bytes: Mapping[str, bytes]) -> pa.Table:
    return pq.read_table(io.BytesIO(table_bytes["images.parquet"]))


def _dataset_result(
    pin: Pin, dropped: tuple[str, ...], spec: ResizeSpec | None
) -> DatasetSyncResult:
    return DatasetSyncResult(
        version=pin.version,
        manifest_sha256=pin.manifest_sha256,
        images_dropped=len(dropped),
        dropped_image_ids=dropped,
        resize=spec,
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
    resize: Mapping[str, ResizeSpec | None] | None = None,
    max_workers: int | None = None,
) -> SyncResult:
    """Sync pinned versions into ``<cache_dir>/<bucket>/<key>``, verifying every file.

    Every manifest, then every table, then the selection and ``resize`` checks, then
    ``check_split`` per dataset, all precede the first image. A dataset absent from ``selection``
    syncs all groups; one absent from ``on_missing_images`` fails on a missing image.

    A dataset with a spec in ``resize`` caches each image resized per the spec, and each mask
    resized to its image's scaled size, at ``<cache_dir>/resized/<spec.id>/<bucket>/<key>``; the
    original is not cached. A sidecar at
    ``<cache_dir>/resized/<spec.id>/.index/<bucket>/<key>.json`` records the source ETag and the
    output's size and sha256, and is written before the output.
    A cached output is verified against its sidecar and never re-downloaded. A dataset absent from
    ``resize``, or mapped to None, caches the original bytes.

    Images and masks are synced on ``max_workers`` threads (default ``min(8, os.cpu_count())``),
    with results taken in row order, so the error raised and ``dropped_image_ids`` match a serial
    run. Configure ``client`` with ``max_pool_connections`` of at least ``max_workers``.

    Raises:
        PinMismatch: A manifest, downloaded or cached, differs from its pin.
        ChecksumMismatch: A table, image or mask, downloaded or cached, differs from its record,
            or a cached resized output has no sidecar or differs from it.
        ImageChanged: An image or mask no longer matches its recorded ETag.
        ImageMissing: An image or mask is absent and its dataset's mode is ``fail``.
        ValueError: ``max_workers`` is below 1 (raised before any S3 call); ``selection`` names
            a dataset not in ``pins`` or a group absent from its dataset, or ``resize`` names a
            dataset not in ``pins`` (raised before any image); or a published key would leave
            the cache.
        PIL.Image.DecompressionBombError: An original to resize exceeds Pillow's pixel limit.
    """
    workers = _resolve_max_workers(max_workers)
    selection = selection or {}
    on_missing_images = on_missing_images or {}
    resize = resize or {}
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
    _check_resize(resize, pins)
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
                resize.get(name),
                workers,
            ),
            resize.get(name),
        )
        for name, pin in pins.items()
    }
    return SyncResult(datasets=results, taxonomy=taxonomy)
