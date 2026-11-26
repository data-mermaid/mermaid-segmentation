"""Feature sets: write-once pyspacer feature files shared across versions, indexed per version.

``stage.stage_feature_set`` stages a version's index and set manifest; ``segdata promote`` publishes
them. ``claim_extractor`` is the one write here: it claims a family for its extractor before any
feature file is written into it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from botocore.exceptions import ClientError

from segdata_spec.errors import ExtractorMismatch, FeatureIndexInvalid
from segdata_spec.layout import (
    EXTRACTOR_SPEC_NAME,
    check_segment,
    feature_key,
    feature_store_prefix,
)
from segdata_spec.manifest import ImageStore
from segdata_spec.s3 import get_bytes, is_not_found, is_precondition_failed

_FEATURE_EXTENSION = ".featurevector"
# The fields that identify an extractor; two specs with equal values produce the same features.
EXTRACTOR_IDENTITY_KEYS = ("class_path", "crop_size", "feature_dim", "weights_sha256")


def extractor_identity(extractor: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity fields of ``extractor``.

    Raises:
        ValueError: An identity key is absent.
    """
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
    return check_segment(match.group(1), "image store segment")


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


def check_index(
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


def extractor_spec_bytes(extractor: Mapping[str, Any]) -> bytes:
    """Serialise an extractor spec as ``_extractor_spec.json`` holds it: indented, key-sorted JSON."""
    return (json.dumps(dict(extractor), indent=2, sort_keys=True) + "\n").encode()


def _extractor_key(image_store: ImageStore, bucket: str, family: str) -> str:
    return (
        feature_store_prefix(feature_store_name(image_store, bucket), family) + EXTRACTOR_SPEC_NAME
    )


def _refuse_other_identity(stored: bytes, extractor: Mapping[str, Any], where: str) -> None:
    differences = _identity_differences(json.loads(stored), extractor_identity(extractor))
    if differences:
        raise ExtractorMismatch(f"{where} names another extractor: {differences}")


def claim_extractor(
    client: Any,
    bucket: str,
    image_store: ImageStore,
    family: str,
    extractor: Mapping[str, Any],
) -> None:
    """Write the family's extractor spec once, or check the stored one has the same identity.

    Every writer of ``.featurevector`` files calls this before writing any file into the family,
    and puts feature files with ``If-None-Match: *``. ``segdata promote`` calls it before it copies
    a staged feature set's vectors; an extraction job outside this repo calls it before it writes
    vectors into a family directly.

    Raises:
        ValueError: ``extractor`` lacks an identity key, or the image store or ``family`` is
            invalid; raised before any S3 call.
        ExtractorMismatch: The stored spec names another identity.
    """
    key = _extractor_key(image_store, bucket, family)
    extractor_identity(extractor)
    try:
        client.put_object(
            Bucket=bucket, Key=key, Body=extractor_spec_bytes(extractor), IfNoneMatch="*"
        )
    except ClientError as exc:
        if not is_precondition_failed(exc):
            raise
    else:
        return
    _refuse_other_identity(get_bytes(client, bucket, key), extractor, f"s3://{bucket}/{key}")


def check_claimed_extractor(
    client: Any,
    bucket: str,
    image_store: ImageStore,
    family: str,
    extractor: Mapping[str, Any],
) -> None:
    """Refuse when the family's stored extractor spec has another identity; write nothing.

    A family with no stored spec passes: ``claim_extractor`` claims it when the first file is
    written.

    Raises:
        ValueError: ``extractor`` lacks an identity key, or the image store or ``family`` is
            invalid; raised before any S3 call.
        ExtractorMismatch: The stored spec names another identity.
    """
    key = _extractor_key(image_store, bucket, family)
    extractor_identity(extractor)
    try:
        stored = get_bytes(client, bucket, key)
    except ClientError as exc:
        if is_not_found(exc):
            return
        raise
    _refuse_other_identity(stored, extractor, f"s3://{bucket}/{key}")
