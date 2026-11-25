"""Feature sets: write-once pyspacer feature files shared across versions, indexed per version."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_spec._s3 import (
    EXTRACTOR_SPEC_NAME,
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    dataset_prefix,
    feature_key,
    feature_set_prefix,
    feature_store_prefix,
    is_precondition_failed,
)
from segdata_spec.errors import (
    ChecksumMismatch,
    ExtractorMismatch,
    FeatureIndexInvalid,
    PinMismatch,
)
from segdata_spec.manifest import (
    EtlInfo,
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
    _Model,
    dump_manifest,
    manifest_sha256,
)
from segdata_spec.publish import (
    GitState,
    _etl_info,
    _get_bytes,
    _guard_git,
    _write_version,
    table_bytes,
)
from segdata_spec.schema import FEATURE_INDEX_SCHEMA, conform

_FEATURE_EXTENSION = ".featurevector"
# The fields that identify an extractor; two specs with equal values produce the same features.
EXTRACTOR_IDENTITY_KEYS = ("class_path", "crop_size", "feature_dim", "weights_sha256")


class FeatureSetStats(_Model):
    """Coverage of a feature set: images indexed, and version images it leaves out."""

    n_images: int
    n_missing: int


class FeatureSetManifest(_Model):
    """``manifest.json`` of a feature set; its sha256 pins the set."""

    schema_version: Literal[1]
    dataset: Pin
    family: str
    extractor: dict[str, Any]
    files: dict[str, FileEntry]
    stats: FeatureSetStats
    etl: EtlInfo
    created_at: datetime


def _check_dataset_pin(manifest: FeatureSetManifest, pin: Pin) -> None:
    """Raise ``PinMismatch`` unless the feature set was published for exactly ``pin``."""
    recorded = manifest.dataset.manifest_sha256
    if recorded != pin.manifest_sha256:
        raise PinMismatch(
            f"feature set {manifest.family!r} was published for dataset manifest {recorded}, "
            f"but the pin expects {pin.manifest_sha256}"
        )


def _extractor_identity(extractor: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity fields of ``extractor``, raising ``ValueError`` when one is absent."""
    missing = [key for key in EXTRACTOR_IDENTITY_KEYS if key not in extractor]
    if missing:
        raise ValueError(f"extractor spec lacks identity keys: {missing}")
    return {key: extractor[key] for key in EXTRACTOR_IDENTITY_KEYS}


def _identity_differences(stored: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    """Describe each identity field where ``stored`` disagrees with ``identity``."""
    return [
        f"{key}: stored {stored.get(key)!r}, given {identity[key]!r}"
        for key in EXTRACTOR_IDENTITY_KEYS
        if stored.get(key) != identity[key]
    ]


def _store_name(image_store: ImageStore) -> str:
    """Return the last path segment of the image store prefix, e.g. ``coralnet``."""
    return image_store.prefix.rstrip("/").rsplit("/", 1)[-1]


def _is_feature_key(key: str, store: str, family: str, image_id: str) -> bool:
    """Return whether ``key`` is ``image_id``'s feature file in the family, with or without a suffix."""
    if key == feature_key(store, family, image_id):
        return True
    stem = f"{feature_store_prefix(store, family)}{image_id}~"
    if not (key.startswith(stem) and key.endswith(_FEATURE_EXTENSION)):
        return False
    suffix = key[len(stem) : -len(_FEATURE_EXTENSION)]
    try:
        return key == feature_key(store, family, image_id, suffix)
    except ValueError:
        return False


def _check_index(
    index: pa.Table, version_ids: set[str], allow_missing: bool, store: str, family: str
) -> int:
    """Return how many version images ``index`` leaves out, refusing an invalid index.

    Raises:
        FeatureIndexInvalid: An id is not in the version, an id repeats, a row's ``key`` is not
            its image's file under ``features/<store>/<family>/``, or images are left out and
            ``allow_missing`` is false.
    """
    ids = index.column("image_id").to_pylist()
    unknown = sorted(set(ids) - version_ids)
    if unknown:
        raise FeatureIndexInvalid(f"index image ids not in the version: {unknown}")
    if pc.count_distinct(index.column("image_id")).as_py() != index.num_rows:
        raise FeatureIndexInvalid("index holds duplicate image ids")
    keys = index.column("key").to_pylist()
    bad = [
        key
        for image_id, key in zip(ids, keys, strict=True)
        if not _is_feature_key(key, store, family, image_id)
    ]
    if bad:
        raise FeatureIndexInvalid(
            f"{len(bad)} index key(s) are not their image's file under "
            f"{feature_store_prefix(store, family)}: {bad[:10]}"
        )
    missing = len(version_ids) - len(ids)
    if missing and not allow_missing:
        raise FeatureIndexInvalid(f"index leaves out {missing} image(s) of the version")
    return missing


def _load_version(client: Any, bucket: str, pin: Pin) -> tuple[Manifest, set[str]]:
    """Return the published dataset manifest and its image ids, each verified against its pin.

    Raises:
        ManifestNotFound: The manifest is absent.
        PinMismatch: The manifest differs from ``pin``.
        ChecksumMismatch: ``images.parquet`` differs from its manifest entry.
    """
    # sync imports this module for FeatureSetManifest, so this import waits for the call.
    from segdata_spec.sync import check_published_manifest

    check_published_manifest(client, bucket, pin)
    prefix = dataset_prefix(pin.name, pin.version)
    manifest_bytes = _get_bytes(client, bucket, prefix + MANIFEST_NAME)
    if manifest_sha256(manifest_bytes) != pin.manifest_sha256:
        raise PinMismatch(f"s3://{bucket}/{prefix}{MANIFEST_NAME} does not match the pin")
    manifest = Manifest.model_validate_json(manifest_bytes)
    images_bytes = _get_bytes(client, bucket, prefix + "images.parquet")
    if hashlib.sha256(images_bytes).hexdigest() != manifest.files["images.parquet"].sha256:
        raise ChecksumMismatch(f"s3://{bucket}/{prefix}images.parquet does not match its manifest")
    images = pq.read_table(io.BytesIO(images_bytes), columns=["image_id"])
    return manifest, set(images.column("image_id").to_pylist())


def _ensure_extractor_spec(
    client: Any, bucket: str, key: str, extractor: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    """Write the family's extractor spec once, or check the stored one has the same identity.

    Raises:
        ExtractorMismatch: The stored spec names another identity.
    """
    data = (json.dumps(dict(extractor), indent=2, sort_keys=True) + "\n").encode()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
    except ClientError as exc:
        if not is_precondition_failed(exc):
            raise
    else:
        return
    differences = _identity_differences(json.loads(_get_bytes(client, bucket, key)), identity)
    if differences:
        raise ExtractorMismatch(f"s3://{bucket}/{key} names another extractor: {differences}")


def _resolve_etl(
    etl: EtlInfo | None,
    repo_dir: Path | None,
    git: GitState | None,
    etl_package: str | None,
    etl_command: str | None,
) -> EtlInfo:
    """Return ``etl`` as given, or build it from ``repo_dir`` behind the git guard.

    Raises:
        ValueError: Both or neither of ``etl`` and ``repo_dir`` are given, or ``repo_dir`` comes
            without ``etl_package`` and ``etl_command``.
        PublishRefused: The repository is dirty or unpushed, or has no ``uv.lock``.
    """
    if repo_dir is None:
        if etl is None:
            raise ValueError("pass exactly one ETL source: repo_dir or etl; got neither")
        return etl
    if etl is not None:
        raise ValueError("pass exactly one ETL source: repo_dir or etl; got both")
    if etl_package is None or etl_command is None:
        raise ValueError("repo_dir needs etl_package and etl_command to record the etl")
    state, lock_sha = _guard_git(repo_dir, git)
    return _etl_info(etl_package, etl_command, state, lock_sha)


def publish_feature_set(
    client: Any,
    pin: Pin,
    family: str,
    index: pa.Table,
    extractor: Mapping[str, Any],
    *,
    bucket: str,
    etl_package: str | None = None,
    etl_command: str | None = None,
    repo_dir: Path | None = None,
    git: GitState | None = None,
    etl: EtlInfo | None = None,
    allow_missing: bool = False,
) -> str:
    """Publish a feature set for a dataset version under ``datasets/<d>/<v>/features/<family>/``.

    Every check precedes the first write. The family's ``_extractor_spec.json`` under
    ``features/<store>/<family>/`` is written once; a stored spec with the same identity
    (``class_path``, ``crop_size``, ``feature_dim``, ``weights_sha256``) is left as is, whatever
    its other keys. Then ``index.parquet``, then ``manifest.json`` last, both write-once.

    The ETL provenance comes from exactly one source: ``repo_dir`` with ``etl_package`` and
    ``etl_command``, checked like a dataset publish, or a ready ``etl`` recorded verbatim with no
    git check.

    Args:
        client: A boto3 S3 client.
        pin: The published dataset version the features describe.
        family: The feature family, a lowercase key segment.
        index: One row per indexed image, conforming to ``FEATURE_INDEX_SCHEMA``.
        extractor: The extractor spec; it must hold the four identity keys.
        bucket: The dataset bucket.
        etl_package: The package that built the features; required with ``repo_dir``.
        etl_command: The command that built the features; required with ``repo_dir``.
        repo_dir: The ETL repository, checked for a clean, pushed tree and a ``uv.lock``.
        git: The repository state; probed from ``repo_dir`` when None.
        etl: Ready provenance for a caller outside the ETL repository; excludes ``repo_dir``.
        allow_missing: Publish an index that leaves out some of the version's images.

    Returns:
        The sha256 of the written feature-set manifest.

    Raises:
        ValueError: ``extractor`` lacks an identity key, ``family`` is not a key segment, or
            the ETL source is not exactly one of ``repo_dir`` (with ``etl_package`` and
            ``etl_command``) and ``etl``; raised before any S3 call.
        PublishRefused: With ``repo_dir``, the tree is dirty or unpushed, or has no ``uv.lock``.
        ManifestNotFound: The dataset manifest is absent.
        PinMismatch: The dataset manifest differs from ``pin``.
        ChecksumMismatch: The dataset's ``images.parquet`` differs from its manifest.
        SchemaValidationError: ``index`` does not conform to ``FEATURE_INDEX_SCHEMA``.
        FeatureIndexInvalid: ``index`` names an image outside the version, repeats one, gives
            a ``key`` other than that image's file under ``features/<store>/<family>/``
            (``<image_id>.featurevector`` or ``<image_id>~<suffix>.featurevector``), or leaves
            images out without ``allow_missing``.
        ExtractorMismatch: The family's stored extractor spec has another identity.
        ObjectExists: The index or manifest key already exists with different bytes.
    """
    identity = _extractor_identity(extractor)
    prefix = feature_set_prefix(pin.name, pin.version, family)
    etl_info = _resolve_etl(etl, repo_dir, git, etl_package, etl_command)
    dataset, version_ids = _load_version(client, bucket, pin)
    conformed = conform(index, FEATURE_INDEX_SCHEMA)
    store = _store_name(dataset.image_store)
    missing = _check_index(conformed, version_ids, allow_missing, store, family)
    store_prefix = feature_store_prefix(store, family)
    index_bytes = table_bytes(conformed)
    manifest = FeatureSetManifest(
        schema_version=1,
        dataset=pin,
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
        etl=etl_info,
        created_at=datetime.now(UTC),
    )
    manifest_bytes = dump_manifest(manifest)
    _ensure_extractor_spec(client, bucket, store_prefix + EXTRACTOR_SPEC_NAME, extractor, identity)
    _write_version(client, bucket, prefix, {FEATURE_INDEX_NAME: index_bytes}, manifest_bytes)
    return manifest_sha256(manifest_bytes)
