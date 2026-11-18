"""Thin S3 helpers for the CoralNet scraper."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from botocore.exceptions import ClientError

_MISSING_KEY_CODES = ("404", "NoSuchKey")


def source_prefix(base_prefix: str, source_id: int) -> str:
    bp = base_prefix.strip("/")
    return f"{bp}/s{source_id}/"


def object_key_annotation_csv(base_prefix: str, source_id: int) -> str:
    return f"{source_prefix(base_prefix, source_id)}annotations.csv"


def annotations_csv_exists(client: Any, bucket: str, base_prefix: str, source_id: int) -> bool:
    prefix = object_key_annotation_csv(base_prefix, source_id)
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp


def upload_csv_body(
    client: Any,
    *,
    bucket: str,
    key: str,
    csv_text: str,
) -> None:
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
