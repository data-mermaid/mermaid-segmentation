"""``segdata promote``: upload one staged dataset version to the write-once dataset bucket.

The bucket denies deletes, so everything is checked before the first write. Images go first, then
the tables, and ``manifest.json`` last: a version is visible only once its manifest exists, and a
rerun after a crash finds the written objects already present and carries on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_etl.common.s3_objects import (
    ObjectConflict,
    copy_new,
    crc64nvme_b64,
    head_with_checksum,
    put_new,
)
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import MANIFEST_NAME, cache_path, dataset_prefix, is_not_found
from segdata_spec.errors import ChecksumMismatch, PublishRefused
from segdata_spec.manifest import Manifest, manifest_sha256

logger = logging.getLogger(__name__)

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")
_NAMED_CONFLICTS = 5

COPIES_NAME = "copies.jsonl"
UPLOADS_NAME = "uploads.jsonl"
_PARQUET = "application/vnd.apache.parquet"
_JSON = "application/json"


@dataclass(frozen=True)
class _Expected:
    """The destination object's expected fingerprint, from ``images.parquet``."""

    etag: str
    size: int
    crc64nvme: str


@dataclass(frozen=True)
class _Copy:
    src_bucket: str
    src_key: str
    src_etag: str
    dst_key: str
    expect: _Expected


@dataclass(frozen=True)
class _Upload:
    local_path: Path
    dst_key: str
    content_type: str
    expect: _Expected


@dataclass(frozen=True)
class _Plan:
    manifest: Manifest
    manifest_bytes: bytes
    steps: tuple[_Copy | _Upload, ...]


def _norm_etag(etag: str) -> str:
    return etag.strip('"')


def _read_manifest(version_dir: Path, dataset: str, version: str) -> tuple[bytes, Manifest]:
    path = version_dir / MANIFEST_NAME
    if not path.is_file():
        raise PublishRefused(f"{path} is not staged")
    data = path.read_bytes()
    manifest = Manifest.model_validate_json(data)
    if (manifest.dataset, manifest.version) != (dataset, version):
        raise PublishRefused(
            f"{path} describes {manifest.dataset} {manifest.version}, not {dataset} {version}"
        )
    if manifest.stats.n_images == 0:
        raise PublishRefused(f"{path} holds no images")
    return data, manifest


def _verify_tables(version_dir: Path, manifest: Manifest) -> dict[str, bytes]:
    """Check each listed table against the manifest; return the staged bytes by table name."""
    tables = {name: _read_table(version_dir, name, manifest) for name in manifest.files}
    if "images.parquet" not in tables:
        raise PublishRefused("the manifest lists no images.parquet")
    return tables


def _table_expectations(tables: Mapping[str, bytes], prefix: str) -> dict[str, _Expected]:
    """Map each table's destination key to the fingerprint of its staged bytes."""
    return {
        prefix + name: _Expected(
            hashlib.md5(data, usedforsecurity=False).hexdigest(), len(data), crc64nvme_b64(data)
        )
        for name, data in tables.items()
    }


def _read_table(version_dir: Path, name: str, manifest: Manifest) -> bytes:
    """Return the staged bytes of table ``name`` after checking them against the manifest."""
    entry = manifest.files[name]
    path = version_dir / name
    if not path.is_file():
        raise ChecksumMismatch(f"staged table {name} is missing")
    data = path.read_bytes()
    if len(data) != entry.bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ChecksumMismatch(f"staged table {name} does not match the manifest")
    return data


def _image_expectations(images: bytes, prefix: str) -> dict[str, _Expected]:
    """Map each image and mask destination key in ``images.parquet`` to its fingerprint."""
    expected: dict[str, _Expected] = {}
    for row in pq.read_table(BytesIO(images)).to_pylist():
        expected[prefix + row["key"]] = _Expected(row["etag"], row["size"], row["crc64nvme"])
        if row["mask_key"] is not None:
            expected[prefix + row["mask_key"]] = _Expected(
                row["mask_etag"], row["mask_size"], row["mask_crc64nvme"]
            )
    return expected


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require_filled(line: Mapping[str, Any], fields: Iterable[str], where: str) -> None:
    for name in fields:
        if not line.get(name):
            raise PublishRefused(f"{where}: {name} is null or empty")


def _check_against_row(
    line: Mapping[str, Any], expect: _Expected, check_etag: bool, where: str
) -> None:
    same = line["size"] == expect.size and line["crc64nvme"] == expect.crc64nvme
    if check_etag:
        same = same and _norm_etag(line["etag"]) == _norm_etag(expect.etag)
    if not same:
        raise PublishRefused(f"{where} disagrees with images.parquet")


def _copy_step(line: Mapping[str, Any], expected: Mapping[str, _Expected], where: str) -> _Copy:
    _require_filled(line, ("src_etag", "crc64nvme", "etag"), where)
    expect = _Expected(line["etag"], line["size"], line["crc64nvme"])
    _check_against_row(line, expected[line["dst_key"]], True, where)
    return _Copy(line["src_bucket"], line["src_key"], line["src_etag"], line["dst_key"], expect)


def _upload_step(
    stage_dir: Path, line: Mapping[str, Any], expected: Mapping[str, _Expected], where: str
) -> _Upload:
    _require_filled(line, ("crc64nvme",), where)
    row = expected[line["dst_key"]]
    _require_filled({"etag": row.etag}, ("etag",), f"images.parquet row for {where}")
    _check_against_row(line, row, False, where)
    path = cache_path(stage_dir, *line["local_path"].split("/", 1))
    data = path.read_bytes() if path.is_file() else None
    if data is None or len(data) != line["size"] or crc64nvme_b64(data) != line["crc64nvme"]:
        raise PublishRefused(f"{where}: staged file {path} does not match the plan")
    return _Upload(path, line["dst_key"], line["content_type"], row)


def _plan_steps(
    stage_dir: Path, version_dir: Path, expected: Mapping[str, _Expected]
) -> tuple[_Copy | _Upload, ...]:
    present = [n for n in (COPIES_NAME, UPLOADS_NAME) if (version_dir / n).is_file()]
    if len(present) != 1:
        raise PublishRefused(
            f"exactly one of {COPIES_NAME} and {UPLOADS_NAME} must be staged, found {present}"
        )
    name = present[0]
    steps: list[_Copy | _Upload] = []
    for number, line in enumerate(_read_lines(version_dir / name), start=1):
        where = f"{name} line {number}"
        if line.get("dst_bucket") != DATASET_BUCKET:
            raise PublishRefused(f"{where}: destination bucket is not {DATASET_BUCKET}")
        if line.get("dst_key") not in expected:
            raise PublishRefused(f"plan keys differ from images.parquet: {where} is not in it")
        if name == COPIES_NAME:
            steps.append(_copy_step(line, expected, where))
        else:
            steps.append(_upload_step(stage_dir, line, expected, where))
    return tuple(steps)


def _check_key_sets(steps: Iterable[_Copy | _Upload], expected: Mapping[str, _Expected]) -> None:
    counts = Counter(step.dst_key for step in steps)
    planned = set(counts)
    duplicated = sorted(k for k, n in counts.items() if n > 1)
    if planned != set(expected) or duplicated:
        raise PublishRefused(
            "plan keys differ from images.parquet: "
            f"{len(planned - set(expected))} extra, {len(set(expected) - planned)} missing, "
            f"{len(duplicated)} duplicated"
        )


def _manifest_published(client: Any, dataset: str, version: str) -> bool:
    """Return whether the version's ``manifest.json`` exists in the dataset bucket."""
    try:
        client.head_object(
            Bucket=DATASET_BUCKET, Key=dataset_prefix(dataset, version) + MANIFEST_NAME
        )
    except ClientError as exc:
        if is_not_found(exc):
            return False
        raise
    return True


def _conflicting_key(client: Any, key: str, expect: _Expected) -> str | None:
    """Return ``key`` if it exists in the dataset bucket with a fingerprint other than ``expect``."""
    head = head_with_checksum(client, DATASET_BUCKET, key)
    if head is None:
        return None
    same = (
        _norm_etag(head["ETag"]) == _norm_etag(expect.etag)
        and head["ContentLength"] == expect.size
        and head.get("ChecksumCRC64NVME") == expect.crc64nvme
    )
    return None if same else key


def _check_destinations(client: Any, planned: Mapping[str, _Expected], workers: int) -> None:
    """Refuse when a planned key exists with other content; matching keys are left to the rerun."""
    conflicts: list[str] = []

    def collect(key: str | None) -> None:
        if key is not None:
            conflicts.append(key)

    _run_bounded(lambda item: _conflicting_key(client, *item), planned.items(), workers, collect)
    if conflicts:
        named = ", ".join(sorted(conflicts)[:_NAMED_CONFLICTS])
        raise PublishRefused(
            f"{len(conflicts)} planned keys already hold other content in {DATASET_BUCKET}: {named}"
        )


def _preflight(client: Any, stage_dir: Path, dataset: str, version: str, workers: int) -> _Plan:
    """Check everything that can be checked before the first write; return the plan."""
    version_dir = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version))
    manifest_bytes, manifest = _read_manifest(version_dir, dataset, version)
    tables = _verify_tables(version_dir, manifest)
    expected = _image_expectations(tables["images.parquet"], manifest.image_store.prefix)
    steps = _plan_steps(stage_dir, version_dir, expected)
    _check_key_sets(steps, expected)
    if _manifest_published(client, dataset, version):
        raise PublishRefused(f"{dataset} {version} is already published in {DATASET_BUCKET}")
    table_expected = _table_expectations(tables, dataset_prefix(dataset, version))
    _check_destinations(client, {**{s.dst_key: s.expect for s in steps}, **table_expected}, workers)
    return _Plan(manifest, manifest_bytes, steps)


def _run_copy(client: Any, step: _Copy) -> bool:
    result = copy_new(
        client,
        src_bucket=step.src_bucket,
        src_key=step.src_key,
        src_etag=step.src_etag,
        dst_bucket=DATASET_BUCKET,
        dst_key=step.dst_key,
        expect_etag=step.expect.etag,
        expect_size=step.expect.size,
        expect_crc64nvme=step.expect.crc64nvme,
    )
    return result.already_present


def _run_upload(client: Any, step: _Upload) -> bool:
    result = put_new(
        client,
        bucket=DATASET_BUCKET,
        key=step.dst_key,
        body=step.local_path.read_bytes(),
        content_type=step.content_type,
    )
    expect = step.expect
    if (
        _norm_etag(result.etag) != _norm_etag(expect.etag)
        or result.size != expect.size
        or result.crc64nvme != expect.crc64nvme
    ):
        raise ObjectConflict(
            f"s3://{DATASET_BUCKET}/{step.dst_key} holds ETag {result.etag}, size {result.size} "
            f"and CRC64NVME {result.crc64nvme}, not the planned values"
        )
    return result.already_present


def _run_step(client: Any, step: _Copy | _Upload) -> tuple[str, bool]:
    if isinstance(step, _Copy):
        return "copied", _run_copy(client, step)
    return "uploaded", _run_upload(client, step)


def _drain(
    pending: set[Future[_Result]], consume: Callable[[_Result], None], *, all_done: bool
) -> set[Future[_Result]]:
    done, pending = wait(pending, return_when=ALL_COMPLETED if all_done else FIRST_COMPLETED)
    for future in done:
        consume(future.result())
    return pending


def _run_bounded(
    fn: Callable[[_Item], _Result],
    items: Iterable[_Item],
    workers: int,
    consume: Callable[[_Result], None],
) -> None:
    """Run ``fn`` on every item in a bounded pool, passing each result to ``consume``.

    The first failure stops submitting and raises.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[_Result]] = set()
        try:
            for item in items:
                pending.add(pool.submit(fn, item))
                if len(pending) >= workers * 2:
                    pending = _drain(pending, consume, all_done=False)
            _drain(pending, consume, all_done=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def _write_images(
    client: Any, steps: Iterable[_Copy | _Upload], workers: int, tally: Counter[str]
) -> None:
    """Run every copy and put in a bounded pool; the first failure stops submitting and raises."""

    def count(result: tuple[str, bool]) -> None:
        kind, present = result
        tally["present" if present else kind] += 1

    _run_bounded(lambda step: _run_step(client, step), steps, workers, count)


def _write_tables(
    client: Any, version_dir: Path, manifest: Manifest, prefix: str, tally: Counter[str]
) -> None:
    for name in manifest.files:
        put_new(
            client,
            bucket=DATASET_BUCKET,
            key=prefix + name,
            body=_read_table(version_dir, name, manifest),
            content_type=_PARQUET,
        )
        tally["tables"] += 1


def _write_version(
    client: Any, plan: _Plan, version_dir: Path, workers: int, tally: Counter[str]
) -> None:
    prefix = dataset_prefix(plan.manifest.dataset, plan.manifest.version)
    _write_images(client, plan.steps, workers, tally)
    _write_tables(client, version_dir, plan.manifest, prefix, tally)
    put_new(
        client,
        bucket=DATASET_BUCKET,
        key=prefix + MANIFEST_NAME,
        body=plan.manifest_bytes,
        content_type=_JSON,
    )
    tally["manifest"] += 1


def promote(client: Any, *, stage_dir: Path, dataset: str, version: str, workers: int) -> None:
    """Upload the staged ``dataset`` ``version`` to the dataset bucket, exactly as staged.

    Raises:
        PublishRefused: The staging is inconsistent or holds no images, the version is already
            published, or a planned key holds other content; nothing was written.
        ChecksumMismatch: A staged table differs from the manifest.
        ObjectConflict: An object landed with, or already holds, other content than planned.
        SourceChanged: A copy source no longer has the planned ETag.
    """
    plan = _preflight(client, stage_dir, dataset, version, workers)
    version_dir = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version))
    tally: Counter[str] = Counter()
    try:
        _write_version(client, plan, version_dir, workers, tally)
    except BaseException:
        logger.error(
            "promote of %s %s failed: at least %d of %d images and %d of %d tables written, "
            "manifest %s; rerun to continue",
            dataset,
            version,
            tally["copied"] + tally["uploaded"] + tally["present"],
            len(plan.steps),
            tally["tables"],
            len(plan.manifest.files),
            "written" if tally["manifest"] else "not written",
        )
        raise
    logger.info(
        "promoted %s %s (manifest sha256 %s): %d copied, %d uploaded, %d already present",
        dataset,
        version,
        manifest_sha256(plan.manifest_bytes),
        tally["copied"],
        tally["uploaded"],
        tally["present"],
    )
