"""Dataset manifest construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .release import S3Target, annotations_parquet_name


@dataclass(frozen=True)
class Source:
    """Provenance of an external dataset."""

    name: str
    citation: str
    raw_data_url: str
    license: str


def build_manifest(
    source: Source,
    target: S3Target,
    *,
    image_key_template: str,
    parquet_columns: dict[str, str],
    ingestion: dict[str, Any],
) -> dict[str, Any]:
    """Return the ``manifest.json`` content describing a published dataset."""
    return {
        "dataset_name": source.name,
        "source": {
            "citation": source.citation,
            "raw_data_url": source.raw_data_url,
            "license": source.license,
        },
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "s3": {
            "bucket": target.bucket,
            "prefix": target.prefix,
            "annotations_parquet": f"{target.prefix}/{annotations_parquet_name(source.name)}",
            "image_key_template": image_key_template,
        },
        "schema": {"parquet_columns": parquet_columns},
        "ingestion": ingestion,
    }
