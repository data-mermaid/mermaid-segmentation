"""Reproducible ETL for CoralNet annotations + image-size metadata.

Builds three parquets per run (audit, annotations, images) under a versioned output prefix so
training defaults can be pinned to a specific build. See ``.docs/CORALNET_ETL.md`` for usage.
"""

from __future__ import annotations

from .annotations import build_annotations
from .audit import audit_sources
from .images import build_images
from .io import compute_version_tag, make_s3_client, write_parquet_deterministic
from .schemas import (
    ANNOTATIONS_PRIMARY_KEY,
    ANNOTATIONS_SCHEMA,
    AUDIT_PRIMARY_KEY,
    AUDIT_SCHEMA,
    IMAGES_PRIMARY_KEY,
    IMAGES_SCHEMA,
    OUTPUT_SCHEMAS,
    SchemaValidationError,
    validate,
)

__all__ = [
    "ANNOTATIONS_PRIMARY_KEY",
    "ANNOTATIONS_SCHEMA",
    "AUDIT_PRIMARY_KEY",
    "AUDIT_SCHEMA",
    "IMAGES_PRIMARY_KEY",
    "IMAGES_SCHEMA",
    "OUTPUT_SCHEMAS",
    "SchemaValidationError",
    "audit_sources",
    "build_annotations",
    "build_images",
    "compute_version_tag",
    "make_s3_client",
    "validate",
    "write_parquet_deterministic",
]
