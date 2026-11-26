"""``segdata promote``: upload one staged dataset or taxonomy version to the write-once dataset bucket.

A taxonomy version has no images: its tables go first and ``manifest.json`` last.

The bucket denies deletes, so everything is checked before the first write. Images go first, then
the tables, and ``manifest.json`` last: a version is visible only once its manifest exists, and a
rerun after a crash finds the written objects already present and carries on.

A staged version carries ``copies.jsonl``, ``uploads.jsonl``, or both (a dense dataset copies its
images from S3 and uploads its masks). Together the two files name each image and mask key in
``images.parquet`` at most once, and no other key. Every image and mask is written in the
Intelligent-Tiering storage class; the tables and the manifest keep the bucket default.

A key in ``images.parquet`` that no plan names is carried: the image is already published. A
version with carried keys needs a published parent on the same image store, and each carried key
must already exist there with the row's ETag, size and CRC64NVME. A version with neither file is
tables-only: every key is carried.

A version folder may also stage feature sets, one per ``features/<family>/`` with ``index.parquet``,
``manifest.json`` and ``copies.jsonl``. Each is a unit of its own, written after the dataset
manifest: the family is claimed for the set manifest's extractor, the vectors are copied in the
``STANDARD`` storage class, then the index and the set manifest go up. A family's keys follow the
dataset's rule: an index key that its ``copies.jsonl`` does not name is carried and must already
exist in the dataset bucket with the row's fingerprint.

A unit whose manifest is already published with the staged bytes is done and skipped, so a rerun
writes only what remains; one published with other bytes is refused. The dataset manifest need not
be staged when the version is published: the stage then holds only feature sets.
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

from segdata_etl.common.copy_plan import COPIES_NAME, UPLOADS_NAME
from segdata_etl.common.featurevector import FEATURE_STORAGE_CLASS
from segdata_etl.common.s3_objects import (
    ObjectConflict,
    copy_new,
    head_with_checksum,
    put_new,
)
from segdata_spec import DATASET_BUCKET
from segdata_spec.errors import ChecksumMismatch, PublishRefused
from segdata_spec.features import check_claimed_extractor, claim_extractor
from segdata_spec.layout import (
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    cache_path,
    dataset_prefix,
    feature_set_prefix,
    taxonomy_prefix,
)
from segdata_spec.manifest import (
    FeatureSetManifest,
    ImageStore,
    Manifest,
    TaxonomyManifest,
    manifest_sha256,
)
from segdata_spec.s3 import crc64nvme_b64, exists, get_bytes, is_not_found, norm_etag
from segdata_spec.sync import require_published_taxonomy

logger = logging.getLogger(__name__)

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")
_NAMED_CONFLICTS = 5

_PARQUET = "application/vnd.apache.parquet"
_JSON = "application/json"
IMAGE_STORAGE_CLASS = "INTELLIGENT_TIERING"


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
    storage_class: str = IMAGE_STORAGE_CLASS


@dataclass(frozen=True)
class _Upload:
    local_path: Path
    dst_key: str
    content_type: str
    expect: _Expected


@dataclass(frozen=True)
class _DatasetUnit:
    """A staged dataset manifest still to publish: its steps and the folder of its tables."""

    manifest: Manifest
    manifest_bytes: bytes
    steps: tuple[_Copy | _Upload, ...]
    version_dir: Path


@dataclass(frozen=True)
class _FamilyUnit:
    """A staged feature set still to publish: its vector copies and the folder of its index."""

    manifest: FeatureSetManifest
    manifest_bytes: bytes
    steps: tuple[_Copy, ...]
    set_dir: Path


@dataclass(frozen=True)
class _Plan:
    image_store: ImageStore
    dataset: _DatasetUnit | None
    families: tuple[_FamilyUnit, ...]


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


def _read_table(
    version_dir: Path, name: str, manifest: Manifest | TaxonomyManifest | FeatureSetManifest
) -> bytes:
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
    line: Mapping[str, Any], expect: _Expected, check_etag: bool, where: str, table: str
) -> None:
    same = line["size"] == expect.size and line["crc64nvme"] == expect.crc64nvme
    if check_etag:
        same = same and norm_etag(line["etag"]) == norm_etag(expect.etag)
    if not same:
        raise PublishRefused(f"{where} disagrees with {table}")


def _copy_step(
    line: Mapping[str, Any],
    expected: Mapping[str, _Expected],
    where: str,
    table: str,
    storage_class: str,
) -> _Copy:
    _require_filled(line, ("src_etag", "crc64nvme", "etag"), where)
    expect = _Expected(line["etag"], line["size"], line["crc64nvme"])
    _check_against_row(line, expected[line["dst_key"]], True, where, table)
    return _Copy(
        line["src_bucket"],
        line["src_key"],
        line["src_etag"],
        line["dst_key"],
        expect,
        storage_class,
    )


def _upload_step(
    stage_dir: Path, line: Mapping[str, Any], expected: Mapping[str, _Expected], where: str
) -> _Upload:
    _require_filled(line, ("crc64nvme",), where)
    row = expected[line["dst_key"]]
    _require_filled({"etag": row.etag}, ("etag",), f"images.parquet row for {where}")
    _check_against_row(line, row, False, where, "images.parquet")
    path = cache_path(stage_dir, *line["local_path"].split("/", 1))
    data = path.read_bytes() if path.is_file() else None
    if data is None or len(data) != line["size"] or crc64nvme_b64(data) != line["crc64nvme"]:
        raise PublishRefused(f"{where}: staged file {path} does not match the plan")
    return _Upload(path, line["dst_key"], line["content_type"], row)


def _plan_steps(
    stage_dir: Path,
    plan_dir: Path,
    expected: Mapping[str, _Expected],
    table: str,
    storage_class: str,
) -> tuple[_Copy | _Upload, ...]:
    """Return the steps of the plan files in ``plan_dir``, copies before uploads.

    Every line must name a key of ``expected``, the rows of the staged ``table``, and no key may
    be named twice.
    """
    present = [n for n in (COPIES_NAME, UPLOADS_NAME) if (plan_dir / n).is_file()]
    steps: list[_Copy | _Upload] = []
    for name in present:
        for number, line in enumerate(_read_lines(plan_dir / name), start=1):
            where = f"{name} line {number}"
            if line.get("dst_bucket") != DATASET_BUCKET:
                raise PublishRefused(f"{where}: destination bucket is not {DATASET_BUCKET}")
            if line.get("dst_key") not in expected:
                raise PublishRefused(f"plan keys differ from {table}: {where} is not in it")
            if name == COPIES_NAME:
                steps.append(_copy_step(line, expected, where, table, storage_class))
            else:
                steps.append(_upload_step(stage_dir, line, expected, where))
    duplicated = [k for k, n in Counter(step.dst_key for step in steps).items() if n > 1]
    if duplicated:
        raise PublishRefused(f"plan keys differ from {table}: {len(duplicated)} duplicated")
    return tuple(steps)


def _index_expectations(index: bytes) -> dict[str, _Expected]:
    """Map each feature file key in a staged ``index.parquet`` to its row's fingerprint."""
    return {
        row["key"]: _Expected(row["etag"], row["size"], row["crc64nvme"])
        for row in pq.read_table(BytesIO(index)).to_pylist()
    }


def _manifest_published(client: Any, prefix: str) -> bool:
    """Return whether ``manifest.json`` exists under the version folder ``prefix`` in the dataset bucket."""
    return exists(client, DATASET_BUCKET, prefix + MANIFEST_NAME)


def _check_parents(client: Any, manifest: Manifest) -> list[bytes]:
    """Refuse unless every parent's published ``manifest.json`` has the pinned bytes; return them."""
    bodies: list[bytes] = []
    for parent in manifest.parents:
        key = dataset_prefix(parent.dataset, parent.version) + MANIFEST_NAME
        try:
            body = get_bytes(client, DATASET_BUCKET, key)
        except ClientError as exc:
            if is_not_found(exc):
                raise PublishRefused(
                    f"parent {parent.dataset} {parent.version} is not published in {DATASET_BUCKET}"
                ) from exc
            raise
        if manifest_sha256(body) != parent.manifest_sha256:
            raise PublishRefused(
                f"parent manifest s3://{DATASET_BUCKET}/{key} does not match its pin"
            )
        bodies.append(body)
    return bodies


def _check_shared_store(manifest: Manifest, parent_bodies: Iterable[bytes]) -> None:
    """Refuse a version with carried keys unless one parent publishes the same image store."""
    if not manifest.parents:
        raise PublishRefused(
            f"a version with images no plan names, or neither {COPIES_NAME} nor {UPLOADS_NAME}, "
            "needs a published parent"
        )
    stores = [Manifest.model_validate_json(body).image_store for body in parent_bodies]
    if manifest.image_store not in stores:
        raise PublishRefused("no parent shares this version's image store")


def _matches(head: Mapping[str, Any], expect: _Expected) -> bool:
    return bool(
        norm_etag(head["ETag"]) == norm_etag(expect.etag)
        and head["ContentLength"] == expect.size
        and head.get("ChecksumCRC64NVME") == expect.crc64nvme
    )


def _conflicting_key(client: Any, key: str, expect: _Expected) -> str | None:
    """Return ``key`` if it exists in the dataset bucket with a fingerprint other than ``expect``."""
    head = head_with_checksum(client, DATASET_BUCKET, key)
    return None if head is None or _matches(head, expect) else key


def _unmatched_key(client: Any, bucket: str, key: str, expect: _Expected) -> str | None:
    """Return ``key`` unless it exists in ``bucket`` with the fingerprint ``expect``."""
    head = head_with_checksum(client, bucket, key)
    return None if head is not None and _matches(head, expect) else key


def _check_present(
    client: Any, bucket: str, expected: Mapping[str, _Expected], workers: int
) -> None:
    """Refuse unless every carried key exists in ``bucket`` with its staged row's fingerprint."""
    unmatched: list[str] = []

    def collect(key: str | None) -> None:
        if key is not None:
            unmatched.append(key)

    _run_bounded(
        lambda item: _unmatched_key(client, bucket, *item), expected.items(), workers, collect
    )
    if unmatched:
        named = ", ".join(sorted(unmatched)[:_NAMED_CONFLICTS])
        raise PublishRefused(
            f"{len(unmatched)} carried keys are absent from s3://{bucket} or hold other content: "
            f"{named}"
        )


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


def _published_status(client: Any, key: str, staged: bytes, what: str) -> bool:
    """Return whether ``key`` is published with the ``staged`` bytes; False when absent.

    Raises:
        PublishRefused: ``key`` is published with other bytes.
    """
    if not exists(client, DATASET_BUCKET, key):
        return False
    if get_bytes(client, DATASET_BUCKET, key) != staged:
        raise PublishRefused(f"{what} is already published in {DATASET_BUCKET} with other bytes")
    return True


def _staged_dataset(stage_dir: Path, version_dir: Path, dataset: str, version: str) -> _DatasetUnit:
    """Read and check the staged dataset version and its plans, with no S3 call."""
    manifest_bytes, manifest = _read_manifest(version_dir, dataset, version)
    tables = _verify_tables(version_dir, manifest)
    expected = _image_expectations(tables["images.parquet"], manifest.image_store.prefix)
    steps = _plan_steps(stage_dir, version_dir, expected, "images.parquet", IMAGE_STORAGE_CLASS)
    return _DatasetUnit(manifest, manifest_bytes, steps, version_dir)


def _staged_family(stage_dir: Path, set_dir: Path) -> tuple[_FamilyUnit, dict[str, _Expected]]:
    """Read and check one staged feature set and its plan, with no S3 call.

    Returns the unit and the fingerprint of every key in its index.
    """
    if (set_dir / UPLOADS_NAME).is_file():
        raise PublishRefused(f"{set_dir / UPLOADS_NAME}: a feature set stages copies only")
    manifest_bytes = (set_dir / MANIFEST_NAME).read_bytes()
    manifest = FeatureSetManifest.model_validate_json(manifest_bytes)
    if manifest.family != set_dir.name:
        raise PublishRefused(f"{set_dir / MANIFEST_NAME} describes family {manifest.family}")
    expected = _index_expectations(_read_table(set_dir, FEATURE_INDEX_NAME, manifest))
    steps = _plan_steps(stage_dir, set_dir, expected, FEATURE_INDEX_NAME, FEATURE_STORAGE_CLASS)
    copies = tuple(step for step in steps if isinstance(step, _Copy))
    return _FamilyUnit(manifest, manifest_bytes, copies, set_dir), expected


def _staged_family_dirs(version_dir: Path) -> list[Path]:
    """Return the staged ``features/<family>/`` folders, each holding its set manifest.

    Raises:
        PublishRefused: A folder has no ``manifest.json``, so the set was not fully staged.
    """
    root = version_dir / "features"
    if not root.is_dir():
        return []
    dirs = sorted(root.iterdir())
    for p in dirs:
        if not (p / MANIFEST_NAME).is_file():
            raise PublishRefused(
                f"{p} holds no {MANIFEST_NAME}: the feature set was not fully staged"
            )
    return dirs


def _published_dataset(client: Any, version_dir: Path, dataset: str, version: str) -> bytes:
    """Return the published manifest of a version whose stage holds only feature sets.

    Raises:
        PublishRefused: The version is not published.
    """
    try:
        return get_bytes(client, DATASET_BUCKET, dataset_prefix(dataset, version) + MANIFEST_NAME)
    except ClientError as exc:
        if is_not_found(exc):
            raise PublishRefused(
                f"{version_dir / MANIFEST_NAME} is not staged and {dataset} {version} is not "
                f"published in {DATASET_BUCKET}"
            ) from exc
        raise


def _check_dataset(client: Any, unit: _DatasetUnit, workers: int) -> None:
    """Check the dataset unit's taxonomy, parents, carried images and destinations."""
    manifest = unit.manifest
    prefix = dataset_prefix(manifest.dataset, manifest.version)
    tables = {name: _read_table(unit.version_dir, name, manifest) for name in manifest.files}
    expected = _image_expectations(tables["images.parquet"], manifest.image_store.prefix)
    if manifest.taxonomy is not None:
        require_published_taxonomy(client, DATASET_BUCKET, manifest.taxonomy)
    parent_bodies = _check_parents(client, manifest)
    planned = {s.dst_key for s in unit.steps}
    carried = {k: v for k, v in expected.items() if k not in planned}
    if carried:
        _check_shared_store(manifest, parent_bodies)
        _check_present(client, manifest.image_store.bucket, carried, workers)
    destinations = {s.dst_key: s.expect for s in unit.steps}
    _check_destinations(client, {**destinations, **_table_expectations(tables, prefix)}, workers)


def _check_family(
    client: Any,
    unit: _FamilyUnit,
    expected: Mapping[str, _Expected],
    store: ImageStore,
    workers: int,
) -> None:
    """Check the family's extractor claim, carried vectors and destinations."""
    manifest = unit.manifest
    prefix = feature_set_prefix(manifest.dataset.name, manifest.dataset.version, manifest.family)
    check_claimed_extractor(client, DATASET_BUCKET, store, manifest.family, manifest.extractor)
    planned = {s.dst_key for s in unit.steps}
    carried = {k: v for k, v in expected.items() if k not in planned}
    if carried:
        _check_present(client, DATASET_BUCKET, carried, workers)
    index = _read_table(unit.set_dir, FEATURE_INDEX_NAME, manifest)
    destinations = {s.dst_key: s.expect for s in unit.steps}
    tables = _table_expectations({FEATURE_INDEX_NAME: index}, prefix)
    _check_destinations(client, {**destinations, **tables}, workers)


def _preflight(client: Any, stage_dir: Path, dataset: str, version: str, workers: int) -> _Plan:
    """Check every staged unit before the first write; return the units still to publish.

    Every staged file is checked before the first S3 call. A unit published with its staged
    manifest bytes is dropped from the plan.

    Raises:
        PublishRefused: A unit is inconsistent or published with other bytes, a
            ``features/<family>/`` folder has no ``manifest.json``, or every staged unit is already
            published.
    """
    version_dir = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version))
    staged = (
        _staged_dataset(stage_dir, version_dir, dataset, version)
        if (version_dir / MANIFEST_NAME).is_file()
        else None
    )
    families = [_staged_family(stage_dir, d) for d in _staged_family_dirs(version_dir)]
    if staged is None:
        manifest_bytes = _published_dataset(client, version_dir, dataset, version)
        manifest = Manifest.model_validate_json(manifest_bytes)
    else:
        manifest_bytes, manifest = staged.manifest_bytes, staged.manifest
    sha = manifest_sha256(manifest_bytes)
    for unit, _ in families:
        pinned = unit.manifest.dataset
        if (pinned.name, pinned.version, pinned.manifest_sha256) != (dataset, version, sha):
            raise PublishRefused(
                f"{unit.set_dir / MANIFEST_NAME} is not feature set {unit.set_dir.name} of "
                f"{dataset} {version} with manifest sha256 {sha}"
            )
    write_dataset = staged is not None and not _published_status(
        client,
        dataset_prefix(dataset, version) + MANIFEST_NAME,
        manifest_bytes,
        f"{dataset} {version}",
    )
    if write_dataset and staged is not None:
        _check_dataset(client, staged, workers)
    pending: list[_FamilyUnit] = []
    for unit, expected in families:
        family = unit.manifest.family
        key = feature_set_prefix(dataset, version, family) + MANIFEST_NAME
        what = f"feature set {family} of {dataset} {version}"
        if not _published_status(client, key, unit.manifest_bytes, what):
            _check_family(client, unit, expected, manifest.image_store, workers)
            pending.append(unit)
    if not write_dataset and not pending:
        raise PublishRefused(f"{dataset} {version} is already published in {DATASET_BUCKET}")
    return _Plan(manifest.image_store, staged if write_dataset else None, tuple(pending))


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
        storage_class=step.storage_class,
    )
    return result.already_present


def _run_upload(client: Any, step: _Upload) -> bool:
    result = put_new(
        client,
        bucket=DATASET_BUCKET,
        key=step.dst_key,
        body=step.local_path.read_bytes(),
        content_type=step.content_type,
        storage_class=IMAGE_STORAGE_CLASS,
    )
    expect = step.expect
    if (
        norm_etag(result.etag) != norm_etag(expect.etag)
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
    client: Any,
    version_dir: Path,
    manifest: Manifest | TaxonomyManifest | FeatureSetManifest,
    prefix: str,
    tally: Counter[str],
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


def _put_manifest(client: Any, key: str, body: bytes) -> None:
    put_new(client, bucket=DATASET_BUCKET, key=key, body=body, content_type=_JSON)


def _write_dataset(client: Any, unit: _DatasetUnit, workers: int, tally: Counter[str]) -> None:
    prefix = dataset_prefix(unit.manifest.dataset, unit.manifest.version)
    _write_images(client, unit.steps, workers, tally)
    _write_tables(client, unit.version_dir, unit.manifest, prefix, tally)
    _put_manifest(client, prefix + MANIFEST_NAME, unit.manifest_bytes)
    tally["manifest"] += 1


def _write_family(
    client: Any, store: ImageStore, unit: _FamilyUnit, workers: int, tally: Counter[str]
) -> None:
    manifest = unit.manifest
    claim_extractor(client, DATASET_BUCKET, store, manifest.family, manifest.extractor)
    _write_images(client, unit.steps, workers, tally)
    prefix = feature_set_prefix(manifest.dataset.name, manifest.dataset.version, manifest.family)
    _write_tables(client, unit.set_dir, manifest, prefix, tally)
    _put_manifest(client, prefix + MANIFEST_NAME, unit.manifest_bytes)
    tally["feature sets"] += 1
    logger.info(
        "published feature set %s (manifest sha256 %s)",
        manifest.family,
        manifest_sha256(unit.manifest_bytes),
    )


def promote(client: Any, *, stage_dir: Path, dataset: str, version: str, workers: int) -> None:
    """Upload the staged units of ``dataset`` ``version`` to the dataset bucket, exactly as staged.

    The dataset goes first when staged and unpublished, then each staged feature set in family order.

    Raises:
        PublishRefused: The staging is inconsistent or holds no images, a ``features/<family>/``
            folder has no ``manifest.json``, every staged unit is already
            published, a unit is published with other bytes, a planned key holds other content, a
            carried image or vector is absent or differs, or a version with carried images lacks a
            parent on its image store; nothing was written.
        ChecksumMismatch: A staged table differs from its manifest.
        ExtractorMismatch: Raised from preflight when a family's stored spec names another
            extractor, and nothing was written; or when another writer claims the family between
            preflight and the claim, and the units before it are in place for a rerun.
        ObjectConflict: An object landed with, or already holds, other content than planned.
        SourceChanged: A copy source no longer has the planned ETag.
    """
    plan = _preflight(client, stage_dir, dataset, version, workers)
    tally: Counter[str] = Counter()
    try:
        if plan.dataset is not None:
            _write_dataset(client, plan.dataset, workers, tally)
        for unit in plan.families:
            _write_family(client, plan.image_store, unit, workers, tally)
    except BaseException:
        logger.error(
            "promote of %s %s failed: %d objects and %d tables written, dataset manifest %s, "
            "%d of %d feature sets published; rerun to continue",
            dataset,
            version,
            tally["copied"] + tally["uploaded"] + tally["present"],
            tally["tables"],
            "written" if tally["manifest"] or plan.dataset is None else "not written",
            tally["feature sets"],
            len(plan.families),
        )
        raise
    logger.info(
        "promoted %s %s: dataset %s, %d feature sets; %d copied, %d uploaded, %d already present",
        dataset,
        version,
        "published" if plan.dataset is not None else "already published",
        len(plan.families),
        tally["copied"],
        tally["uploaded"],
        tally["present"],
    )


@dataclass(frozen=True)
class _TaxonomyPlan:
    manifest: TaxonomyManifest
    manifest_bytes: bytes
    prefix: str


def _read_taxonomy_manifest(
    version_dir: Path, name: str, version: str
) -> tuple[bytes, TaxonomyManifest]:
    path = version_dir / MANIFEST_NAME
    if not path.is_file():
        raise PublishRefused(f"{path} is not staged")
    data = path.read_bytes()
    manifest = TaxonomyManifest.model_validate_json(data)
    if (manifest.name, manifest.version) != (name, version):
        raise PublishRefused(
            f"{path} describes {manifest.name} {manifest.version}, not {name} {version}"
        )
    return data, manifest


def _taxonomy_preflight(
    client: Any, stage_dir: Path, name: str, version: str, workers: int
) -> tuple[_TaxonomyPlan, Path]:
    """Check the staged taxonomy and the destination keys before the first write."""
    prefix = taxonomy_prefix(name, version)
    version_dir = cache_path(stage_dir, DATASET_BUCKET, prefix)
    manifest_bytes, manifest = _read_taxonomy_manifest(version_dir, name, version)
    tables = {table: _read_table(version_dir, table, manifest) for table in manifest.files}
    if _manifest_published(client, prefix):
        raise PublishRefused(f"taxonomy {name} {version} is already published in {DATASET_BUCKET}")
    _check_destinations(client, _table_expectations(tables, prefix), workers)
    return _TaxonomyPlan(manifest, manifest_bytes, prefix), version_dir


def promote_taxonomy(
    client: Any, *, stage_dir: Path, name: str, version: str, workers: int
) -> None:
    """Upload the staged taxonomy ``name`` ``version`` to the dataset bucket, tables then manifest.

    Raises:
        PublishRefused: The staging is missing or names another version, the version is already
            published, or a destination key holds other content; nothing was written.
        ChecksumMismatch: A staged table differs from the manifest.
    """
    plan, version_dir = _taxonomy_preflight(client, stage_dir, name, version, workers)
    tally: Counter[str] = Counter()
    _write_tables(client, version_dir, plan.manifest, plan.prefix, tally)
    put_new(
        client,
        bucket=DATASET_BUCKET,
        key=plan.prefix + MANIFEST_NAME,
        body=plan.manifest_bytes,
        content_type=_JSON,
    )
    logger.info(
        "promoted taxonomy %s %s (manifest sha256 %s)",
        name,
        version,
        manifest_sha256(plan.manifest_bytes),
    )
