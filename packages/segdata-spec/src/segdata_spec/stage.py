"""Stage dataset and taxonomy versions as local files, tables first and the manifest last.

``segdata promote`` is the only code that uploads a staged version to S3.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from segdata_spec.errors import (
    ChecksumMismatch,
    ManifestNotFound,
    ObjectExists,
    PinMismatch,
    PublishRefused,
    SchemaValidationError,
)
from segdata_spec.features import check_index, extractor_identity, feature_store_name
from segdata_spec.git import Publishable
from segdata_spec.layout import (
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    cache_path,
    dataset_prefix,
    feature_set_prefix,
    taxonomy_prefix,
)
from segdata_spec.manifest import (
    DiffFromParent,
    EtlInfo,
    FeatureSetManifest,
    FeatureSetPin,
    FeatureSetStats,
    FileEntry,
    ImageStore,
    LabelMapping,
    Manifest,
    ParentRef,
    Pin,
    SourceInfo,
    Stats,
    TaxonomyManifest,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.schema import (
    DATASET_TABLES,
    FEATURE_INDEX_SCHEMA,
    POINTS_SCHEMA,
    TAXONOMY_TABLES,
    conform,
    validate_dataset_tables,
)


def table_bytes(table: pa.Table) -> bytes:
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


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_new(path: Path, data: bytes) -> None:
    """Write ``data`` once; an existing file with identical bytes counts as already written."""
    if path.is_file():
        if path.read_bytes() != data:
            raise ObjectExists(f"{path} already exists with different bytes")
        return
    _atomic_write(path, data)


def _refuse_if_staged(version_dir: Path) -> None:
    if (version_dir / MANIFEST_NAME).exists():
        raise PublishRefused(f"{version_dir / MANIFEST_NAME} is already staged")


def _write_version(version_dir: Path, blobs: Mapping[str, bytes], manifest_bytes: bytes) -> None:
    for name, data in blobs.items():
        _write_new(version_dir / name, data)
    _write_new(version_dir / MANIFEST_NAME, manifest_bytes)


def _etl_info(package: str, command: str, repo: Publishable) -> EtlInfo:
    return EtlInfo(
        package=package,
        command=command,
        git_sha=repo.git.sha,
        git_dirty=False,
        uv_lock_sha256=repo.uv_lock_sha256,
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


def _find_manifest(dirs: Sequence[Path], bucket: str, prefix: str) -> tuple[Path, bytes]:
    for directory in dirs:
        path = cache_path(directory, bucket, prefix + MANIFEST_NAME)
        if path.is_file():
            return path.parent, path.read_bytes()
    searched = ", ".join(str(d) for d in dirs)
    raise ManifestNotFound(f"{prefix}{MANIFEST_NAME} is in none of: {searched}")


def read_version_tables(
    dirs: Sequence[Path],
    bucket: str,
    name: str,
    version: str,
    sha256: str,
    tables: Sequence[str],
) -> tuple[Manifest, dict[str, bytes]]:
    """Read a dataset version's manifest and ``tables`` from the first of ``dirs`` holding its manifest.

    ``dirs`` are stage or cache roots laid out as ``<dir>/<bucket>/<key>``. The manifest is checked
    against ``sha256`` and each table against the manifest. A table the manifest does not
    list is left out of the result.

    Raises:
        ManifestNotFound: No directory holds the manifest.
        PinMismatch: The manifest's sha256 differs from ``sha256``.
        ChecksumMismatch: A listed table is missing or its sha256 differs from its manifest entry.
    """
    prefix = dataset_prefix(name, version)
    version_dir, manifest_bytes = _find_manifest(dirs, bucket, prefix)
    if manifest_sha256(manifest_bytes) != sha256:
        raise PinMismatch(f"manifest {prefix}{MANIFEST_NAME} does not match its pin")
    manifest = Manifest.model_validate_json(manifest_bytes)
    blobs: dict[str, bytes] = {}
    for table in tables:
        entry = manifest.files.get(table)
        if entry is None:
            continue
        path = version_dir / table
        if not path.is_file():
            raise ChecksumMismatch(f"{prefix}{table} is missing next to its manifest")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise ChecksumMismatch(f"{prefix}{table} does not match its manifest")
        blobs[table] = data
    return manifest, blobs


def read_parent_tables(
    parent_dirs: Sequence[Path], bucket: str, parent: ParentRef
) -> dict[str, bytes]:
    """Read the parent's images and points tables from the first of ``parent_dirs`` holding its manifest.

    Returns the table bytes by file name; ``points.parquet`` is absent when the parent lacks it.

    Raises:
        ManifestNotFound: No directory holds the parent's manifest.
        PinMismatch: The parent manifest's sha256 differs from ``parent.manifest_sha256``.
        ChecksumMismatch: A table is missing or its sha256 differs from its manifest entry.
    """
    _, blobs = read_version_tables(
        parent_dirs,
        bucket,
        parent.dataset,
        parent.version,
        parent.manifest_sha256,
        ("images.parquet", "points.parquet"),
    )
    return blobs


def _parent_tables(
    parent_dirs: Sequence[Path], bucket: str, parent: ParentRef
) -> dict[str, pa.Table]:
    return {
        name: pq.read_table(io.BytesIO(data))
        for name, data in read_parent_tables(parent_dirs, bucket, parent).items()
    }


def stage_dataset(
    stage_dir: Path,
    *,
    parent_dirs: Sequence[Path],
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
    taxonomy: Pin | None = None,
    repo: Publishable,
) -> tuple[Manifest, str]:
    """Stage a dataset version under ``<stage_dir>/<bucket>/datasets/<dataset>/<version>/``.

    Returns the manifest and its sha256. The tables are written before the manifest. The first parent is
    read from the first of ``parent_dirs`` that holds it. A rerun after a partial stage adopts tables whose
    bytes are identical.

    Raises:
        SchemaValidationError: A table fails its schema or the cross-table rules.
        ManifestNotFound: No directory in ``parent_dirs`` holds the first parent's manifest.
        PinMismatch: The first parent's manifest differs from its ``ParentRef``.
        ChecksumMismatch: A parent table differs from the parent manifest.
        ObjectExists: A staged table already exists with different bytes; the manifest is then not written.
    """
    conformed = _conform_all(tables, DATASET_TABLES)
    validate_dataset_tables(kind, conformed)
    blobs = {name: table_bytes(table) for name, table in conformed.items()}
    version_dir = cache_path(stage_dir, bucket, dataset_prefix(dataset, version))
    diff = None
    if parents:
        diff = compute_diff(_parent_tables(parent_dirs, bucket, parents[0]), conformed)
    manifest = Manifest(
        schema_version=1,
        dataset=dataset,
        version=version,
        kind=kind,
        created_at=datetime.now(UTC),
        image_store=image_store,
        etl=_etl_info(etl_package, etl_command, repo),
        source=source,
        parents=tuple(parents),
        label_mapping=label_mapping,
        files=_file_entries(conformed, blobs),
        stats=_dataset_stats(conformed, dropped or {}),
        diff_from_parent=diff,
        taxonomy=taxonomy,
    )
    manifest_bytes = dump_manifest(manifest)
    _write_version(version_dir, blobs, manifest_bytes)
    return manifest, manifest_sha256(manifest_bytes)


def stage_taxonomy(
    stage_dir: Path,
    *,
    bucket: str,
    name: str,
    version: str,
    tables: Mapping[str, pa.Table],
    etl_package: str,
    etl_command: str,
    source: SourceInfo,
    repo: Publishable,
) -> tuple[TaxonomyManifest, str]:
    """Stage a taxonomy version under ``<stage_dir>/<bucket>/taxonomy/<name>/<version>/``.

    Returns the manifest and its sha256.

    Raises:
        SchemaValidationError: A table is missing, unknown or fails its schema.
        ObjectExists: A staged table already exists with different bytes; the manifest is then not written.
    """
    missing = sorted(set(TAXONOMY_TABLES) - set(tables))
    if missing:
        raise SchemaValidationError(f"missing taxonomy tables: {missing}")
    conformed = _conform_all(tables, TAXONOMY_TABLES)
    blobs = {table_name: table_bytes(table) for table_name, table in conformed.items()}
    version_dir = cache_path(stage_dir, bucket, taxonomy_prefix(name, version))
    manifest = TaxonomyManifest(
        schema_version=1,
        name=name,
        version=version,
        created_at=datetime.now(UTC),
        etl=_etl_info(etl_package, etl_command, repo),
        source=source,
        files=_file_entries(conformed, blobs),
    )
    manifest_bytes = dump_manifest(manifest)
    _write_version(version_dir, blobs, manifest_bytes)
    return manifest, manifest_sha256(manifest_bytes)


def stage_feature_set(
    stage_dir: Path,
    *,
    dataset_dirs: Sequence[Path],
    bucket: str,
    dataset: Pin,
    family: str,
    index: pa.Table,
    extractor: Mapping[str, Any],
    allow_missing: bool = False,
    etl_package: str,
    etl_command: str,
    repo: Publishable,
) -> FeatureSetPin:
    """Stage a feature set under ``<stage_dir>/<bucket>/datasets/<d>/<v>/features/<family>/``.

    The dataset version is read from the first of ``dataset_dirs`` that holds its manifest: the
    stage directory for a version staged alongside, or the cache that ``sync_tables`` filled for a
    published one. ``index.parquet`` is written before ``manifest.json``. The set manifest pins
    ``dataset`` and records ``extractor``, from which promote writes the family's
    ``_extractor_spec.json``.

    Returns:
        The feature-set pin of the staged manifest, which promote publishes unchanged.

    Raises:
        ValueError: ``extractor`` lacks an identity key, ``family`` is not a key segment, or the
            dataset's image store is not ``images/<segment>/`` in ``bucket``.
        PublishRefused: The set's manifest is already staged.
        ManifestNotFound: No directory in ``dataset_dirs`` holds the dataset manifest.
        PinMismatch: The dataset manifest differs from ``dataset``.
        ChecksumMismatch: The dataset's ``images.parquet`` differs from its manifest.
        SchemaValidationError: ``index`` does not conform to ``FEATURE_INDEX_SCHEMA``.
        FeatureIndexInvalid: ``index`` names an image outside the version, repeats an image or a
            key, gives a key other than that image's file under ``features/<store>/<family>/``, or
            leaves images out without ``allow_missing``.
        ObjectExists: A staged file already exists with different bytes.
    """
    extractor_identity(extractor)
    set_dir = cache_path(
        stage_dir, bucket, feature_set_prefix(dataset.name, dataset.version, family)
    )
    manifest, blobs = read_version_tables(
        dataset_dirs,
        bucket,
        dataset.name,
        dataset.version,
        dataset.manifest_sha256,
        ("images.parquet",),
    )
    images = pq.read_table(io.BytesIO(blobs["images.parquet"]), columns=["image_id"])
    conformed = conform(index, FEATURE_INDEX_SCHEMA)
    store = feature_store_name(manifest.image_store, bucket)
    missing = check_index(
        conformed, set(images.column("image_id").to_pylist()), allow_missing, store, family
    )
    _refuse_if_staged(set_dir)
    index_bytes = table_bytes(conformed)
    set_manifest = FeatureSetManifest(
        schema_version=1,
        dataset=dataset,
        family=family,
        extractor=dict(extractor),
        files={
            FEATURE_INDEX_NAME: FileEntry(
                sha256=hashlib.sha256(index_bytes).hexdigest(),
                bytes=len(index_bytes),
                rows=conformed.num_rows,
            )
        },
        stats=FeatureSetStats(n_images=conformed.num_rows, n_missing=missing),
        etl=_etl_info(etl_package, etl_command, repo),
        created_at=datetime.now(UTC),
    )
    manifest_bytes = dump_manifest(set_manifest)
    _write_version(set_dir, {FEATURE_INDEX_NAME: index_bytes}, manifest_bytes)
    return FeatureSetPin(
        dataset=dataset, family=family, manifest_sha256=manifest_sha256(manifest_bytes)
    )
