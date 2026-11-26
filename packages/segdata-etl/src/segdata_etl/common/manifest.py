"""The typed ``manifest.json`` that ``segdata ingest`` writes and the dataset builders read."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .release import S3Target, annotations_parquet_name


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)


class IngestSource(_Model):
    """Provenance of an external dataset."""

    citation: str
    raw_data_url: str
    license: str


class IngestS3(_Model):
    """Where the ingested files live."""

    bucket: str
    prefix: str
    annotations_parquet: str
    image_key_template: str


class IngestSchema(_Model):
    """Column descriptions of the annotations parquet."""

    parquet_columns: dict[str, str]


class Ingestion(_Model):
    """What the ingest did: drop counts, upload count, and dataset-specific details."""

    dropped: dict[str, int]
    images_uploaded: int
    details: dict[str, Any] = Field(default_factory=dict)


class IngestManifest(_Model):
    """The ``manifest.json`` of one ingested dataset prefix."""

    dataset_name: str
    source: IngestSource
    created_at_utc: str
    s3: IngestS3
    # ``schema`` would shadow ``BaseModel.schema``; the JSON key stays ``schema``.
    parquet_schema: IngestSchema = Field(alias="schema")
    ingestion: Ingestion


def build_manifest(
    dataset_name: str,
    source: IngestSource,
    target: S3Target,
    *,
    image_key_template: str,
    parquet_columns: dict[str, str],
    dropped: dict[str, int],
    images_uploaded: int,
    details: dict[str, Any] | None = None,
) -> IngestManifest:
    """Return the ``manifest.json`` content describing an ingested dataset."""
    return IngestManifest(
        dataset_name=dataset_name,
        source=source,
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        s3=IngestS3(
            bucket=target.bucket,
            prefix=target.prefix,
            annotations_parquet=f"{target.prefix}/{annotations_parquet_name(dataset_name)}",
            image_key_template=image_key_template,
        ),
        schema=IngestSchema(parquet_columns=parquet_columns),
        ingestion=Ingestion(
            dropped=dropped, images_uploaded=images_uploaded, details=details or {}
        ),
    )


def read_ingest_manifest(data: bytes) -> IngestManifest:
    """Parse ``manifest.json`` bytes; raises ``ValidationError`` on any other shape."""
    return IngestManifest.model_validate_json(data)
