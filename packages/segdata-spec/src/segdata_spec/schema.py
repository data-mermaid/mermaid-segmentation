"""PyArrow schemas for the dataset and taxonomy tables, with conformance checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from segdata_spec.errors import SchemaValidationError

IMAGES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("key", pa.string(), nullable=False),
        pa.field("etag", pa.string(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("crc64nvme", pa.string(), nullable=True),
        pa.field("media_type", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("group", pa.string(), nullable=False),
        pa.field("mask_key", pa.string(), nullable=True),
        pa.field("mask_etag", pa.string(), nullable=True),
        pa.field("mask_size", pa.int64(), nullable=True),
        pa.field("mask_crc64nvme", pa.string(), nullable=True),
        pa.field("source_fingerprint", pa.string(), nullable=False),
        pa.field("source_name", pa.string(), nullable=True),
        # Dataset-specific extras, serialised as a JSON object string.
        pa.field("attrs", pa.string(), nullable=True),
    ]
)

POINTS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),
        pa.field("col", pa.int32(), nullable=False),
        pa.field("label_id", pa.int32(), nullable=False),
        pa.field("annotator", pa.string(), nullable=False),
    ]
)

LABELS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("label_id", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("mermaid_benthic_attribute_id", pa.string(), nullable=True),
        pa.field("mermaid_benthic_attribute", pa.string(), nullable=True),
        pa.field("mermaid_growth_form", pa.string(), nullable=True),
        pa.field("mapping_source", pa.string(), nullable=False),
    ]
)

LABEL_COUNTS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("label_id", pa.int32(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
    ]
)

BENTHIC_ATTRIBUTES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("parent_id", pa.string(), nullable=True),
    ]
)

GROWTH_FORMS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
    ]
)

FEATURE_INDEX_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("key", pa.string(), nullable=False),
        pa.field("etag", pa.string(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("crc64nvme", pa.string(), nullable=True),
        pa.field("n_points", pa.int32(), nullable=False),
    ]
)

DATASET_TABLES: dict[str, pa.Schema] = {
    "images.parquet": IMAGES_SCHEMA,
    "points.parquet": POINTS_SCHEMA,
    "labels.parquet": LABELS_SCHEMA,
    "stats/label_counts.parquet": LABEL_COUNTS_SCHEMA,
}

TAXONOMY_TABLES: dict[str, pa.Schema] = {
    "benthic_attributes.parquet": BENTHIC_ATTRIBUTES_SCHEMA,
    "growth_forms.parquet": GROWTH_FORMS_SCHEMA,
}


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Select ``schema``'s columns in order and cast them to its types.

    Raises:
        SchemaValidationError: A column is missing, a value cannot be cast, or a non-nullable
            column holds nulls.
    """
    missing = [name for name in schema.names if name not in table.column_names]
    if missing:
        raise SchemaValidationError(f"table is missing required columns: {missing}")
    try:
        result = table.select(schema.names).cast(schema)
    except (ValueError, pa.ArrowNotImplementedError) as exc:
        # pyarrow raises ValueError for nulls in a non-nullable column and ArrowInvalid for bad values.
        raise SchemaValidationError(f"table does not cast to the schema: {exc}") from exc
    return result


def _require_unique(table: pa.Table, column: str, file_name: str) -> None:
    if pc.count_distinct(table.column(column)).as_py() != table.num_rows:
        raise SchemaValidationError(f"{file_name}: {column!r} values are not unique")


def validate_dataset_tables(
    kind: Literal["points", "dense"], tables: Mapping[str, pa.Table]
) -> None:
    """Check cross-table rules on conformed dataset tables keyed by file name.

    Raises:
        SchemaValidationError: A required table is absent, ``points.parquet`` presence disagrees
            with ``kind``, ids repeat, ``mask_key`` population disagrees with ``kind``, or
            a row with a ``mask_key`` lacks ``mask_etag`` or ``mask_size``.
    """
    for name in ("images.parquet", "labels.parquet", "stats/label_counts.parquet"):
        if name not in tables:
            raise SchemaValidationError(f"missing table {name}")
    if (kind == "points") != ("points.parquet" in tables):
        raise SchemaValidationError("points.parquet must be present exactly when kind is points")
    images = tables["images.parquet"]
    _require_unique(images, "image_id", "images.parquet")
    _require_unique(tables["labels.parquet"], "label_id", "labels.parquet")
    masks = images.column("mask_key")
    if kind == "dense" and masks.null_count:
        raise SchemaValidationError("dense dataset: every image needs a mask_key")
    if kind == "dense":
        for column in ("mask_etag", "mask_size"):
            if images.column(column).null_count:
                raise SchemaValidationError(f"dense dataset: every image needs a {column}")
    if kind == "points" and masks.null_count != images.num_rows:
        raise SchemaValidationError("points dataset: mask_key must be null for every image")
