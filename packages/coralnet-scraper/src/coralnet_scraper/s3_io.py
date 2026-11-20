"""Thin S3 helpers for the CoralNet scraper."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from botocore.exceptions import ClientError

_MISSING_KEY_CODES = ("404", "NoSuchKey")
_DELETE_BATCH_SIZE = 1000  # the DeleteObjects API limit


def source_prefix(base_prefix: str, source_id: int) -> str:
    """Return the key prefix ``<base_prefix>/s<source_id>/`` for one source."""
    bp = base_prefix.strip("/")
    return f"{bp}/s{source_id}/"


def object_key_annotation_csv(base_prefix: str, source_id: int) -> str:
    """Return the S3 key of a source's ``annotations.csv``."""
    return f"{source_prefix(base_prefix, source_id)}annotations.csv"


def object_key_image_list_csv(base_prefix: str, source_id: int) -> str:
    """Return the S3 key of a source's ``image_list.csv``."""
    return f"{source_prefix(base_prefix, source_id)}image_list.csv"


def object_key_labelset_csv(base_prefix: str, source_id: int) -> str:
    """Return the S3 key of a source's ``labelset.csv``."""
    return f"{source_prefix(base_prefix, source_id)}labelset.csv"


def object_key_metadata_csv(base_prefix: str, source_id: int) -> str:
    """Return the S3 key of a source's ``metadata.csv``."""
    return f"{source_prefix(base_prefix, source_id)}metadata.csv"


def object_key_image(base_prefix: str, source_id: int, image_id: str) -> str:
    """Return the S3 key of one source image, ``images/<image_id>.jpg``."""
    return f"{source_prefix(base_prefix, source_id)}images/{image_id}.jpg"


def object_key_complete_marker(base_prefix: str, source_id: int) -> str:
    """Return the S3 key of a source's completion marker, ``_complete.json``."""
    return f"{source_prefix(base_prefix, source_id)}_complete.json"


def partial_prefix(base_prefix: str, source_id: int) -> str:
    """Return the key prefix ``<source prefix>_partial/`` of a source's unfinished work."""
    return f"{source_prefix(base_prefix, source_id)}_partial/"


def key_exists(client: Any, bucket: str, key: str) -> bool:
    """Return whether an object with exactly ``key`` exists in ``bucket``."""
    # S3 lists keys in order, so an exact match is the first key under its own prefix.
    resp = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(o["Key"] == key for o in resp.get("Contents", []))


def object_size(client: Any, bucket: str, key: str) -> int | None:
    """Return the size in bytes of the object at ``key``, or None when it does not exist.

    Other ClientErrors propagate.
    """
    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except ClientError as e:
        if e.response["Error"]["Code"] in _MISSING_KEY_CODES:
            return None
        raise


def read_json_s3(client: Any, bucket: str, key: str) -> Any:
    """Return the parsed JSON document at ``key``, or None when the key is missing.

    Other ClientErrors propagate.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in _MISSING_KEY_CODES:
            return None
        raise
    return json.loads(response["Body"].read())


def write_json_s3(client: Any, bucket: str, key: str, document: Any) -> None:
    """Upload ``document`` as JSON to ``key`` in ``bucket``, replacing any existing object."""
    client.put_object(
        Bucket=bucket, Key=key, Body=json.dumps(document), ContentType="application/json"
    )


def object_sizes(client: Any, bucket: str, prefix: str) -> dict[str, int]:
    """Return the size in bytes of every object under ``prefix``, across all listing pages."""
    return {
        obj["Key"]: obj["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    }


def delete_keys(client: Any, bucket: str, keys: list[str]) -> None:
    """Delete ``keys`` from ``bucket`` in batches of :data:`_DELETE_BATCH_SIZE`.

    Raises:
        RuntimeError: If a response reports per-key errors; names the first failed key, its
            code and the number of failed keys in that batch.
    """
    for start in range(0, len(keys), _DELETE_BATCH_SIZE):
        batch = keys[start : start + _DELETE_BATCH_SIZE]
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        errors = response.get("Errors") or []
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"Could not delete {len(errors)} key(s) from s3://{bucket}; "
                f"first: {first.get('Key')} ({first.get('Code')})"
            )


def delete_prefix(client: Any, bucket: str, prefix: str) -> None:
    """Delete every object whose key starts with ``prefix``, across all listing pages."""
    delete_keys(client, bucket, list(object_sizes(client, bucket, prefix)))


def upload_csv_body(
    client: Any,
    *,
    bucket: str,
    key: str,
    csv_text: str,
) -> None:
    """Upload ``csv_text`` to ``key`` in ``bucket`` with a ``text/csv`` content type."""
    client.put_object(Bucket=bucket, Key=key, Body=csv_text, ContentType="text/csv")


def read_csv_s3(client: Any, bucket: str, key: str, **read_csv_kwargs: Any) -> pd.DataFrame | None:
    """Read a CSV from S3 into a DataFrame, or return None when the key is missing.

    Other ClientErrors propagate.
    """
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in _MISSING_KEY_CODES:
            return None
        raise
    return pd.read_csv(io.BytesIO(response["Body"].read()), **read_csv_kwargs)
