"""Publish dataset and taxonomy versions to S3 as write-once objects, manifest last."""

from __future__ import annotations

import hashlib
import io
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_spec.errors import (
    ChecksumMismatch,
    ObjectExists,
    PinMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.manifest import (
    DiffFromParent,
    EtlInfo,
    FileEntry,
    ImageStore,
    LabelMapping,
    Manifest,
    ParentRef,
    SourceInfo,
    Stats,
    TaxonomyManifest,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.schema import (
    DATASET_TABLES,
    POINTS_SCHEMA,
    TAXONOMY_TABLES,
    conform,
    validate_dataset_tables,
)

_MANIFEST_NAME = "manifest.json"
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


@dataclass(frozen=True)
class GitState:
    """The state of the ETL repository at publish time."""

    sha: str
    dirty: bool
    pushed: bool


def _git(repo_dir: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def probe_git(repo_dir: Path) -> GitState:
    """Read HEAD, tracked-file dirtiness and whether a remote branch contains HEAD."""
    sha = _git(repo_dir, "rev-parse", "HEAD")
    dirty = bool(_git(repo_dir, "status", "--porcelain", "--untracked-files=no"))
    pushed = bool(_git(repo_dir, "branch", "-r", "--contains", "HEAD"))
    return GitState(sha=sha, dirty=dirty, pushed=pushed)


def uv_lock_sha256(repo_dir: Path) -> str | None:
    """Return the hex sha256 of ``uv.lock`` in ``repo_dir``, or None when it is absent."""
    lock = repo_dir / "uv.lock"
    if not lock.is_file():
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _is_not_found(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return (
        code in _NOT_FOUND_CODES
        or exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
    )


def _is_precondition_failed(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return (
        code == "PreconditionFailed"
        or exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412
    )


def _guard_git(repo_dir: Path, git: GitState | None) -> tuple[GitState, str]:
    """Return the git state and lock hash, refusing a dirty, unpushed or lock-less tree."""
    state = git or probe_git(repo_dir)
    if state.dirty:
        raise PublishRefused("working tree has uncommitted changes to tracked files")
    if not state.pushed:
        raise PublishRefused(f"HEAD {state.sha} is not contained in any remote branch")
    lock = uv_lock_sha256(repo_dir)
    if lock is None:
        raise PublishRefused(f"no uv.lock in {repo_dir}")
    return state, lock


def _parquet_bytes(table: pa.Table) -> bytes:
    """Serialise to parquet; the same table always yields the same bytes."""
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


def _conform_all(
    tables: Mapping[str, pa.Table], schemas: Mapping[str, pa.Schema]
) -> dict[str, pa.Table]:
    unknown = sorted(set(tables) - set(schemas))
    if unknown:
        raise SchemaValidationError(f"unknown table names: {unknown}")
    return {name: conform(table, schemas[name]) for name, table in tables.items()}


def _file_entries(
    tables: Mapping[str, pa.Table], blobs: Mapping[str, bytes]
) -> dict[str, FileEntry]:
    return {
        name: FileEntry(
            sha256=hashlib.sha256(blobs[name]).hexdigest(),
            bytes=len(blobs[name]),
            rows=table.num_rows,
        )
        for name, table in tables.items()
    }


def _exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_not_found(exc):
            return False
        raise
    return True


def _put_new(client: Any, bucket: str, key: str, data: bytes) -> None:
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
    except ClientError as exc:
        if _is_precondition_failed(exc):
            raise ObjectExists(f"s3://{bucket}/{key} already exists") from exc
        raise


def _get_bytes(client: Any, bucket: str, key: str) -> bytes:
    data: bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return data


def _refuse_if_published(client: Any, bucket: str, prefix: str) -> None:
    if _exists(client, bucket, prefix + _MANIFEST_NAME):
        raise PublishRefused(f"s3://{bucket}/{prefix}{_MANIFEST_NAME} already exists")


def _write_version(
    client: Any, bucket: str, prefix: str, blobs: Mapping[str, bytes], manifest_bytes: bytes
) -> None:
    for name, data in blobs.items():
        _put_new(client, bucket, prefix + name, data)
    _put_new(client, bucket, prefix + _MANIFEST_NAME, manifest_bytes)


def _etl_info(package: str, command: str, git: GitState, lock_sha: str) -> EtlInfo:
    return EtlInfo(
        package=package,
        command=command,
        git_sha=git.sha,
        git_dirty=False,
        uv_lock_sha256=lock_sha,
    )


def _dataset_stats(tables: Mapping[str, pa.Table], dropped: Mapping[str, int]) -> Stats:
    images = tables["images.parquet"]
    points = tables.get("points.parquet")
    return Stats(
        n_images=images.num_rows,
        n_points=points.num_rows if points is not None else 0,
        n_groups=pc.count_distinct(images.column("group")).as_py(),
        image_bytes=pc.sum(images.column("size")).as_py() or 0,
        dropped=dict(dropped),
    )


def _anti_count(left: pa.Table, right: pa.Table, keys: list[str]) -> int:
    return left.join(right.select(keys), keys=keys, join_type="left anti").num_rows


def compute_diff(parent: Mapping[str, pa.Table], child: Mapping[str, pa.Table]) -> DiffFromParent:
    """Count image and point changes from ``parent`` to ``child`` with pyarrow joins.

    Tables are keyed by file name; ``points.parquet`` absent on a side counts as empty.
    """
    cols = ["image_id", "key", "etag", "size"]
    p_img = parent["images.parquet"].select(cols)
    c_img = child["images.parquet"].select(cols)
    common = p_img.join(c_img.select(["image_id"]), keys="image_id", join_type="inner").num_rows
    unchanged = p_img.join(c_img, keys=cols, join_type="inner").num_rows
    point_keys = ["image_id", "row", "col", "label_id", "annotator"]
    p_pts = parent.get("points.parquet", POINTS_SCHEMA.empty_table())
    c_pts = child.get("points.parquet", POINTS_SCHEMA.empty_table())
    return DiffFromParent(
        images_added=_anti_count(c_img, p_img, ["image_id"]),
        images_removed=_anti_count(p_img, c_img, ["image_id"]),
        images_changed=common - unchanged,
        points_added=_anti_count(c_pts, p_pts, point_keys),
        points_removed=_anti_count(p_pts, c_pts, point_keys),
    )


def _fetch_parent_tables(client: Any, bucket: str, parent: ParentRef) -> dict[str, pa.Table]:
    """Fetch and verify the parent's manifest, images and points tables."""
    prefix = f"datasets/{parent.dataset}/{parent.version}/"
    manifest_bytes = _get_bytes(client, bucket, prefix + _MANIFEST_NAME)
    if manifest_sha256(manifest_bytes) != parent.manifest_sha256:
        raise PinMismatch(f"parent manifest {prefix}{_MANIFEST_NAME} does not match its pin")
    manifest = Manifest.model_validate_json(manifest_bytes)
    tables: dict[str, pa.Table] = {}
    for name in ("images.parquet", "points.parquet"):
        entry = manifest.files.get(name)
        if entry is None:
            continue
        data = _get_bytes(client, bucket, prefix + name)
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise ChecksumMismatch(f"{prefix}{name} does not match the parent manifest")
        tables[name] = pq.read_table(io.BytesIO(data))
    return tables


def publish_dataset(
    client: Any,
    *,
    bucket: str,
    dataset: str,
    version: str,
    kind: Literal["points", "dense"],
    tables: Mapping[str, pa.Table],
    image_store: ImageStore,
    etl_package: str,
    etl_command: str,
    source: SourceInfo,
    parents: Sequence[ParentRef] = (),
    label_mapping: LabelMapping | None = None,
    dropped: Mapping[str, int] | None = None,
    repo_dir: Path,
    git: GitState | None = None,
) -> tuple[Manifest, str]:
    """Publish a dataset version under ``datasets/<dataset>/<version>/``.

    Returns the manifest and its sha256.

    Raises:
        PublishRefused: The tree is dirty or unpushed, or the version already exists.
        SchemaValidationError: A table fails its schema or the cross-table rules.
        PinMismatch: The first parent's manifest differs from its ``ParentRef``.
        ChecksumMismatch: A parent table differs from the parent manifest.
        ObjectExists: A table key already exists; the manifest is then not written.
    """
    state, lock_sha = _guard_git(repo_dir, git)
    conformed = _conform_all(tables, DATASET_TABLES)
    validate_dataset_tables(kind, conformed)
    blobs = {name: _parquet_bytes(table) for name, table in conformed.items()}
    prefix = f"datasets/{dataset}/{version}/"
    _refuse_if_published(client, bucket, prefix)
    diff = None
    if parents:
        diff = compute_diff(_fetch_parent_tables(client, bucket, parents[0]), conformed)
    manifest = Manifest(
        schema_version=1,
        dataset=dataset,
        version=version,
        kind=kind,
        created_at=datetime.now(UTC),
        image_store=image_store,
        etl=_etl_info(etl_package, etl_command, state, lock_sha),
        source=source,
        parents=tuple(parents),
        label_mapping=label_mapping,
        files=_file_entries(conformed, blobs),
        stats=_dataset_stats(conformed, dropped or {}),
        diff_from_parent=diff,
    )
    manifest_bytes = dump_manifest(manifest)
    _write_version(client, bucket, prefix, blobs, manifest_bytes)
    return manifest, manifest_sha256(manifest_bytes)


def publish_taxonomy(
    client: Any,
    *,
    bucket: str,
    name: str,
    version: str,
    tables: Mapping[str, pa.Table],
    etl_package: str,
    etl_command: str,
    source: SourceInfo,
    repo_dir: Path,
    git: GitState | None = None,
) -> tuple[TaxonomyManifest, str]:
    """Publish a taxonomy version under ``taxonomy/<name>/<version>/``.

    Returns the manifest and its sha256.

    Raises:
        PublishRefused: The tree is dirty or unpushed, or the version already exists.
        SchemaValidationError: A table is missing, unknown or fails its schema.
        ObjectExists: A table key already exists; the manifest is then not written.
    """
    state, lock_sha = _guard_git(repo_dir, git)
    missing = sorted(set(TAXONOMY_TABLES) - set(tables))
    if missing:
        raise SchemaValidationError(f"missing taxonomy tables: {missing}")
    conformed = _conform_all(tables, TAXONOMY_TABLES)
    blobs = {table_name: _parquet_bytes(table) for table_name, table in conformed.items()}
    prefix = f"taxonomy/{name}/{version}/"
    _refuse_if_published(client, bucket, prefix)
    manifest = TaxonomyManifest(
        schema_version=1,
        name=name,
        version=version,
        created_at=datetime.now(UTC),
        etl=_etl_info(etl_package, etl_command, state, lock_sha),
        source=source,
        files=_file_entries(conformed, blobs),
    )
    manifest_bytes = dump_manifest(manifest)
    _write_version(client, bucket, prefix, blobs, manifest_bytes)
    return manifest, manifest_sha256(manifest_bytes)
