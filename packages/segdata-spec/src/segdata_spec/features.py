"""Feature sets: write-once pyspacer feature files shared across versions, indexed per version."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from segdata_spec._s3 import (
    EXTRACTOR_SPEC_NAME,
    FEATURE_INDEX_NAME,
    MANIFEST_NAME,
    _check_segment,
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
    FeatureSetManifest,
    FeatureSetPin,
    FeatureSetStats,
    FileEntry,
    ImageStore,
    Manifest,
    Pin,
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
from segdata_spec.sync import check_published_manifest

_FEATURE_EXTENSION = ".featurevector"
# The fields that identify an extractor; two specs with equal values produce the same features.
EXTRACTOR_IDENTITY_KEYS = ("class_path", "crop_size", "feature_dim", "weights_sha256")


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


_IMAGE_STORE_PREFIX = re.compile(r"images/([^/]+)/")


def feature_store_name(image_store: ImageStore, bucket: str) -> str:
    """Return the store name of ``image_store``, the segment of its ``images/<segment>/`` prefix.

    Raises:
        ValueError: The image store is outside ``bucket``, its prefix is not exactly
            ``images/<segment>/``, or the segment is not a lowercase key segment.
    """
    match = _IMAGE_STORE_PREFIX.fullmatch(image_store.prefix)
    if image_store.bucket != bucket or match is None:
        raise ValueError(
            f"image store s3://{image_store.bucket}/{image_store.prefix} is not "
            f"s3://{bucket}/images/<segment>/"
        )
    return _check_segment(match.group(1), "image store segment")


def feature_file_key(
    image_store: ImageStore,
    bucket: str,
    family: str,
    image_id: str,
    suffix: str | None = None,
) -> str:
    """Return one image's feature file key, ``features/<store>/<family>/<image_id>[~<suffix>].featurevector``.

    Raises:
        ValueError: The image store is not a valid store in ``bucket``, or ``family`` or
            ``suffix`` is not a lowercase key segment.
    """
    return feature_key(feature_store_name(image_store, bucket), family, image_id, suffix)


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
        FeatureIndexInvalid: An id is not in the version, an id or a key repeats, a row's ``key`` is not
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
    if len(set(keys)) != len(keys):
        raise FeatureIndexInvalid("index holds a duplicate key: two images name one feature file")
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


def claim_extractor(
    client: Any,
    bucket: str,
    image_store: ImageStore,
    family: str,
    extractor: Mapping[str, Any],
) -> None:
    """Write the family's extractor spec once, or check the stored one has the same identity.

    Writers call this before writing any ``.featurevector`` into the family, and put feature
    files with ``If-None-Match: *``.

    Raises:
        ValueError: ``extractor`` lacks an identity key, or the image store or ``family`` is
            invalid; raised before any S3 call.
        ExtractorMismatch: The stored spec names another identity.
    """
    key = (
        feature_store_prefix(feature_store_name(image_store, bucket), family) + EXTRACTOR_SPEC_NAME
    )
    identity = _extractor_identity(extractor)
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
) -> FeatureSetPin:
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
        The feature-set pin: ``pin``, ``family`` and the sha256 of the written manifest.

    Raises:
        ValueError: ``extractor`` lacks an identity key, ``family`` is not a key segment, or
            the ETL source is not exactly one of ``repo_dir`` (with ``etl_package`` and
            ``etl_command``) and ``etl``; raised before any S3 call. The dataset's image store
            is not ``images/<segment>/`` in ``bucket``; raised before any write.
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
    _extractor_identity(extractor)
    prefix = feature_set_prefix(pin.name, pin.version, family)
    etl_info = _resolve_etl(etl, repo_dir, git, etl_package, etl_command)
    dataset, version_ids = _load_version(client, bucket, pin)
    conformed = conform(index, FEATURE_INDEX_SCHEMA)
    store = feature_store_name(dataset.image_store, bucket)
    missing = _check_index(conformed, version_ids, allow_missing, store, family)
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
    claim_extractor(client, bucket, dataset.image_store, family, extractor)
    _write_version(client, bucket, prefix, {FEATURE_INDEX_NAME: index_bytes}, manifest_bytes)
    return FeatureSetPin(
        dataset=pin, family=family, manifest_sha256=manifest_sha256(manifest_bytes)
    )
