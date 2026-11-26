"""``segdata promote``: upload one staged dataset version to the write-once dataset bucket.

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

A staged version may also carry ``features.jsonl`` with a feature index and extractor spec under
``features/<family>/`` in the version folder: the feature vectors are copied after the images, in the
bucket's default storage class, and the feature set is published once the dataset manifest exists.

A rerun on a version whose manifest is published with the staged bytes, and which stages a feature
set, writes only the feature copies and the feature set.
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

import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_etl.common.copy_plan import (
    COPIES_NAME,
    FEATURE_STAGE_NAME,
    FEATURES_NAME,
    UPLOADS_NAME,
)
from segdata_etl.common.featurevector import FEATURE_STORAGE_CLASS
from segdata_etl.common.s3_objects import (
    ObjectConflict,
    copy_new,
    crc64nvme_b64,
    head_with_checksum,
    put_new,
)
from segdata_spec import DATASET_BUCKET
from segdata_spec._s3 import (
    EXTRACTOR_SPEC_NAME,
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    cache_path,
    dataset_prefix,
    feature_key,
    feature_store_prefix,
    is_not_found,
)
from segdata_spec.errors import (
    ChecksumMismatch,
    ExtractorMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.features import (
    EXTRACTOR_IDENTITY_KEYS,
    claim_extractor,
    feature_store_name,
    publish_feature_set,
)
from segdata_spec.manifest import Manifest, Pin, manifest_sha256
from segdata_spec.schema import FEATURE_INDEX_SCHEMA, conform

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
class _FeatureSet:
    """A staged feature set: its vector copies, index and extractor spec."""

    family: str
    steps: tuple[_Copy, ...]
    index: pa.Table
    extractor: dict[str, Any]
    allow_missing: bool


@dataclass(frozen=True)
class _Plan:
    manifest: Manifest
    manifest_bytes: bytes
    steps: tuple[_Copy | _Upload, ...]
    features: _FeatureSet | None
    # The dataset manifest is already published with the staged bytes; only features remain.
    dataset_published: bool = False


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
    return _Copy(
        line["src_bucket"],
        line["src_key"],
        line["src_etag"],
        line["dst_key"],
        expect,
    )


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
    """Return the steps of every staged plan file, copies before uploads; none when tables-only.

    The caller checks that the combined destination keys match ``expected`` exactly once each.
    """
    present = [n for n in (COPIES_NAME, UPLOADS_NAME) if (version_dir / n).is_file()]
    steps: list[_Copy | _Upload] = []
    for name in present:
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
    """Refuse a plan key that ``images.parquet`` lacks, or one that two plan lines name."""
    counts = Counter(step.dst_key for step in steps)
    extra = counts.keys() - expected.keys()
    duplicated = [k for k, n in counts.items() if n > 1]
    if extra or duplicated:
        raise PublishRefused(
            f"plan keys differ from images.parquet: {len(extra)} extra, {len(duplicated)} duplicated"
        )


def _staged_family(version_dir: Path) -> str:
    """Return the one family staged under ``features/`` in the version folder."""
    root = version_dir / "features"
    families = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if len(families) != 1:
        raise PublishRefused(
            f"{FEATURES_NAME} needs exactly one staged feature family under {root}, "
            f"found {families}"
        )
    return families[0]


def _index_expectations(index: pa.Table) -> dict[str, _Expected]:
    """Map each staged feature index key to its row's fingerprint, refusing a repeated key."""
    rows = index.to_pylist()
    expected = {row["key"]: _Expected(row["etag"], row["size"], row["crc64nvme"]) for row in rows}
    if len(expected) != len(rows):
        raise PublishRefused("the staged feature index repeats a key")
    return expected


def _feature_step(line: Mapping[str, Any], expected: Mapping[str, _Expected], where: str) -> _Copy:
    _require_filled(line, ("src_etag", "crc64nvme", "etag"), where)
    if line.get("dst_bucket") != DATASET_BUCKET:
        raise PublishRefused(f"{where}: destination bucket is not {DATASET_BUCKET}")
    row = expected.get(line["dst_key"])
    if row is None:
        raise PublishRefused(f"{where}: {line['dst_key']} is not in the staged feature index")
    expect = _Expected(line["etag"], line["size"], line["crc64nvme"])
    if (_norm_etag(expect.etag), expect.size, expect.crc64nvme) != (
        _norm_etag(row.etag),
        row.size,
        row.crc64nvme,
    ):
        raise PublishRefused(f"{where} disagrees with the staged feature index")
    return _Copy(
        line["src_bucket"],
        line["src_key"],
        line["src_etag"],
        line["dst_key"],
        expect,
        FEATURE_STORAGE_CLASS,
    )


def _check_feature_keys(steps: Iterable[_Copy], expected: Mapping[str, _Expected]) -> None:
    counts = Counter(step.dst_key for step in steps)
    planned = set(counts)
    duplicated = sorted(k for k, n in counts.items() if n > 1)
    if planned != set(expected) or duplicated:
        raise PublishRefused(
            f"{FEATURES_NAME} keys differ from the staged feature index: "
            f"{len(planned - set(expected))} extra, {len(set(expected) - planned)} missing, "
            f"{len(duplicated)} duplicated"
        )


_FEATURE_EXTENSION = ".featurevector"


def _is_family_key(key: str, store: str, family: str, image_id: str) -> bool:
    """Return whether ``key`` is ``image_id``'s feature file in the family, with or without a suffix."""
    if key == feature_key(store, family, image_id):
        return True
    stem = f"{feature_store_prefix(store, family)}{image_id}~"
    if not (key.startswith(stem) and key.endswith(_FEATURE_EXTENSION)):
        return False
    try:
        return key == feature_key(
            store, family, image_id, key[len(stem) : -len(_FEATURE_EXTENSION)]
        )
    except ValueError:
        return False


def _check_staged_index(
    index: pa.Table, version_ids: set[str], store: str, family: str, allow_missing: bool
) -> pa.Table:
    """Return the staged index conformed, refusing what ``publish_feature_set`` would refuse."""
    try:
        conformed = conform(index, FEATURE_INDEX_SCHEMA)
    except SchemaValidationError as exc:
        raise PublishRefused(f"the staged feature index does not conform: {exc}") from exc
    ids = conformed.column("image_id").to_pylist()
    unknown = sorted(set(ids) - version_ids)
    if unknown:
        raise PublishRefused(
            f"staged feature index ids not in images.parquet: {unknown[:_NAMED_CONFLICTS]}"
        )
    if len(set(ids)) != len(ids):
        raise PublishRefused("the staged feature index repeats an image id")
    bad = [
        key
        for image_id, key in zip(ids, conformed.column("key").to_pylist(), strict=True)
        if not _is_family_key(key, store, family, image_id)
    ]
    if bad:
        raise PublishRefused(
            f"{len(bad)} staged feature index key(s) are not their image's file under "
            f"{feature_store_prefix(store, family)}: {bad[:_NAMED_CONFLICTS]}"
        )
    missing = len(version_ids) - len(ids)
    if missing and not allow_missing:
        raise PublishRefused(
            f"the staged feature index leaves out {missing} image(s) and {FEATURE_STAGE_NAME} "
            "does not allow missing ones"
        )
    return conformed


def _read_stage(family_dir: Path) -> bool:
    """Return the staged ``allow_missing`` choice, refusing a missing or malformed sidecar."""
    path = family_dir / FEATURE_STAGE_NAME
    if not path.is_file():
        raise PublishRefused(f"{path} is not staged")
    allow_missing = json.loads(path.read_text()).get("allow_missing")
    if not isinstance(allow_missing, bool):
        raise PublishRefused(f"{path} holds no boolean allow_missing")
    return allow_missing


def _read_extractor(family_dir: Path) -> dict[str, Any]:
    extractor: dict[str, Any] = json.loads((family_dir / EXTRACTOR_SPEC_NAME).read_text())
    absent = [name for name in EXTRACTOR_IDENTITY_KEYS if name not in extractor]
    if absent:
        raise PublishRefused(f"the staged extractor spec lacks identity keys: {absent}")
    return extractor


def _read_feature_set(version_dir: Path, manifest: Manifest, images: bytes) -> _FeatureSet | None:
    """Return the staged feature set, or None when the version stages no ``features.jsonl``.

    Every check ``publish_feature_set`` makes on the index and extractor spec runs here, so a
    bad feature set is refused before the dataset is written.
    """
    plan_path = version_dir / FEATURES_NAME
    if not plan_path.is_file():
        return None
    family = _staged_family(version_dir)
    family_dir = version_dir / "features" / family
    allow_missing = _read_stage(family_dir)
    version_ids = set(pq.read_table(BytesIO(images), columns=["image_id"]).column(0).to_pylist())
    index = _check_staged_index(
        pq.read_table(family_dir / FEATURE_INDEX_NAME),
        version_ids,
        feature_store_name(manifest.image_store, DATASET_BUCKET),
        family,
        allow_missing,
    )
    expected = _index_expectations(index)
    steps = tuple(
        _feature_step(line, expected, f"{FEATURES_NAME} line {number}")
        for number, line in enumerate(_read_lines(plan_path), start=1)
    )
    _check_feature_keys(steps, expected)
    return _FeatureSet(family, steps, index, _read_extractor(family_dir), allow_missing)


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


def _check_parents(client: Any, manifest: Manifest) -> list[bytes]:
    """Refuse unless every parent's published ``manifest.json`` has the pinned bytes; return them."""
    bodies: list[bytes] = []
    for parent in manifest.parents:
        key = dataset_prefix(parent.dataset, parent.version) + MANIFEST_NAME
        try:
            body = client.get_object(Bucket=DATASET_BUCKET, Key=key)["Body"].read()
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
        _norm_etag(head["ETag"]) == _norm_etag(expect.etag)
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


def _check_images_present(
    client: Any, bucket: str, expected: Mapping[str, _Expected], workers: int
) -> None:
    """Refuse unless every expected image and mask key exists in ``bucket`` as planned."""
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
            f"{len(unmatched)} image keys are absent from s3://{bucket} or hold other content: "
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


def _published_bytes(client: Any, dataset: str, version: str) -> bytes:
    key = dataset_prefix(dataset, version) + MANIFEST_NAME
    body: bytes = client.get_object(Bucket=DATASET_BUCKET, Key=key)["Body"].read()
    return body


def _resumes_features(
    client: Any, manifest_bytes: bytes, manifest: Manifest, features: _FeatureSet | None
) -> bool:
    """Return whether the published version only lacks its staged feature set; refuse otherwise.

    Raises:
        PublishRefused: The version is published and stages no features, or was published with
            other manifest bytes.
    """
    dataset, version = manifest.dataset, manifest.version
    if not _manifest_published(client, dataset, version):
        return False
    if features is None or _published_bytes(client, dataset, version) != manifest_bytes:
        raise PublishRefused(f"{dataset} {version} is already published in {DATASET_BUCKET}")
    return True


def _check_stored_extractor(client: Any, manifest: Manifest, features: _FeatureSet) -> None:
    """Refuse when the family's published extractor spec names another identity than the staged one.

    Raises:
        ExtractorMismatch: An identity field differs.
    """
    store = feature_store_name(manifest.image_store, DATASET_BUCKET)
    key = feature_store_prefix(store, features.family) + EXTRACTOR_SPEC_NAME
    try:
        body = client.get_object(Bucket=DATASET_BUCKET, Key=key)["Body"].read()
    except ClientError as exc:
        if is_not_found(exc):
            return
        raise
    stored = json.loads(body)
    differences = [
        f"{name}: stored {stored.get(name)!r}, staged {features.extractor.get(name)!r}"
        for name in EXTRACTOR_IDENTITY_KEYS
        if stored.get(name) != features.extractor.get(name)
    ]
    if differences:
        raise ExtractorMismatch(
            f"s3://{DATASET_BUCKET}/{key} names another extractor: {differences}"
        )


def _preflight(client: Any, stage_dir: Path, dataset: str, version: str, workers: int) -> _Plan:
    """Check everything that can be checked before the first write; return the plan."""
    prefix = dataset_prefix(dataset, version)
    version_dir = cache_path(stage_dir, DATASET_BUCKET, prefix)
    manifest_bytes, manifest = _read_manifest(version_dir, dataset, version)
    tables = _verify_tables(version_dir, manifest)
    expected = _image_expectations(tables["images.parquet"], manifest.image_store.prefix)
    steps = _plan_steps(stage_dir, version_dir, expected)
    if steps:
        _check_key_sets(steps, expected)
    features = _read_feature_set(version_dir, manifest, tables["images.parquet"])
    feature_steps = features.steps if features is not None else ()
    dataset_published = _resumes_features(client, manifest_bytes, manifest, features)
    if features is not None:
        _check_stored_extractor(client, manifest, features)
        if dataset_published:
            _check_destinations(client, {s.dst_key: s.expect for s in feature_steps}, workers)
            return _Plan(manifest, manifest_bytes, (), features, dataset_published=True)
    parent_bodies = _check_parents(client, manifest)
    planned = {s.dst_key for s in steps}
    carried = {k: v for k, v in expected.items() if k not in planned}
    if carried:
        _check_shared_store(manifest, parent_bodies)
        _check_images_present(client, manifest.image_store.bucket, carried, workers)
    file_expected = _table_expectations(tables, prefix)
    destinations = {s.dst_key: s.expect for s in (*steps, *feature_steps)}
    _check_destinations(client, {**destinations, **file_expected}, workers)
    return _Plan(manifest, manifest_bytes, steps, features)


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


def _publish_features(client: Any, plan: _Plan, features: _FeatureSet) -> str:
    """Publish the staged feature set against the dataset manifest just written."""
    pin = Pin(
        name=plan.manifest.dataset,
        version=plan.manifest.version,
        manifest_sha256=manifest_sha256(plan.manifest_bytes),
    )
    return publish_feature_set(
        client,
        pin,
        features.family,
        features.index,
        features.extractor,
        bucket=DATASET_BUCKET,
        etl=plan.manifest.etl,
        allow_missing=features.allow_missing,
    ).manifest_sha256


def _write_version(
    client: Any, plan: _Plan, version_dir: Path, workers: int, tally: Counter[str]
) -> None:
    prefix = dataset_prefix(plan.manifest.dataset, plan.manifest.version)
    _write_images(client, plan.steps, workers, tally)
    if plan.features is not None:
        # The writer contract: the family is claimed before any file is written into it.
        claim_extractor(
            client,
            DATASET_BUCKET,
            plan.manifest.image_store,
            plan.features.family,
            plan.features.extractor,
        )
        _write_images(client, plan.features.steps, workers, tally)
    if not plan.dataset_published:
        _write_tables(client, version_dir, plan.manifest, prefix, tally)
        put_new(
            client,
            bucket=DATASET_BUCKET,
            key=prefix + MANIFEST_NAME,
            body=plan.manifest_bytes,
            content_type=_JSON,
        )
    tally["manifest"] += 1
    if plan.features is not None:
        sha = _publish_features(client, plan, plan.features)
        logger.info("published feature set %s (manifest sha256 %s)", plan.features.family, sha)


def promote(client: Any, *, stage_dir: Path, dataset: str, version: str, workers: int) -> None:
    """Upload the staged ``dataset`` ``version`` to the dataset bucket, exactly as staged.

    Raises:
        PublishRefused: The staging is inconsistent or holds no images, the version is already
            published (unless with the staged manifest bytes and a staged feature set, which then
            is all that is written), a planned key holds other content, a version with carried
            images lacks a parent on its image store or an image there, or ``features.jsonl``
            disagrees with the staged feature index; nothing was written.
        ChecksumMismatch: A staged table differs from the manifest.
        ObjectConflict: An object landed with, or already holds, other content than planned.
        SourceChanged: A copy source no longer has the planned ETag.
        ExtractorMismatch: Raised from preflight when the stored spec names another extractor,
            in which case nothing was written; or raised when the family is claimed just before
            the feature copies because another writer claimed it in between, in which case the
            image copies are already in place and a rerun continues.
    """
    plan = _preflight(client, stage_dir, dataset, version, workers)
    version_dir = cache_path(stage_dir, DATASET_BUCKET, dataset_prefix(dataset, version))
    tally: Counter[str] = Counter()
    try:
        _write_version(client, plan, version_dir, workers, tally)
    except BaseException:
        logger.error(
            "promote of %s %s failed: at least %d of %d images and feature vectors and %d of %d "
            "tables written, manifest %s; rerun to continue",
            dataset,
            version,
            tally["copied"] + tally["uploaded"] + tally["present"],
            len(plan.steps) + (len(plan.features.steps) if plan.features is not None else 0),
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
