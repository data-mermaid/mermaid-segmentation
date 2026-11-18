"""Thin S3 helpers for CoralNet scraper (bucketing conventions match ETL)."""

from __future__ import annotations

from typing import Any


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


def delete_source_prefix(client: Any, bucket: str, base_prefix: str, source_id: int) -> int:
    """Delete all keys under ``s<source_id>/``; returns approximate object count."""
    prefix = source_prefix(base_prefix, source_id)
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=bucket, Key=obj["Key"])
            deleted += 1
    return deleted
