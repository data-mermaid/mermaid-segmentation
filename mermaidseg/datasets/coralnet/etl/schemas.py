"""PyArrow schemas for the three CoralNet ETL output parquets.

Single source of truth for column names, dtypes, and primary keys. The writer in
:mod:`mermaidseg.datasets.coralnet.etl.io` casts every DataFrame to the declared schema before
writing so byte output is deterministic and downstream readers can rely on the column types.
"""

from __future__ import annotations

import pyarrow as pa

AUDIT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("source_id", pa.int32(), nullable=False),
        pa.field("audit_timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("has_images_folder", pa.bool_(), nullable=False),
        pa.field("has_annotations_csv", pa.bool_(), nullable=False),
        pa.field("has_image_list_csv", pa.bool_(), nullable=False),
        pa.field("has_labelset_csv", pa.bool_(), nullable=False),
        pa.field("has_metadata_csv", pa.bool_(), nullable=False),
        pa.field("n_images_s3", pa.int64(), nullable=False),
        pa.field("n_images_csv", pa.int64(), nullable=False),
        pa.field("n_annotations", pa.int64(), nullable=False),
        pa.field("n_unique_images_annotated", pa.int64(), nullable=False),
        pa.field("n_labels", pa.int64(), nullable=True),
        pa.field("n_metadata_rows", pa.int64(), nullable=True),
        pa.field("annotations_empty", pa.bool_(), nullable=False),
        pa.field("image_list_empty", pa.bool_(), nullable=False),
        pa.field("labelset_empty", pa.bool_(), nullable=False),
        pa.field("metadata_empty", pa.bool_(), nullable=False),
        pa.field("annotations_csv_read_failed", pa.bool_(), nullable=False),
        pa.field("image_list_csv_read_failed", pa.bool_(), nullable=False),
        pa.field("labelset_csv_read_failed", pa.bool_(), nullable=False),
        pa.field("metadata_csv_read_failed", pa.bool_(), nullable=False),
        pa.field("is_complete", pa.bool_(), nullable=False),
        pa.field("image_count_match", pa.bool_(), nullable=False),
        pa.field("errors", pa.list_(pa.string()), nullable=True),
    ]
)

ANNOTATIONS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("source_id", pa.int32(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("row", pa.int32(), nullable=False),
        pa.field("col", pa.int32(), nullable=False),
        pa.field("coralnet_id", pa.int32(), nullable=False),
        pa.field("status", pa.dictionary(pa.int32(), pa.string()), nullable=True),
    ]
)

IMAGES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("source_id", pa.int32(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("s3_key", pa.string(), nullable=False),
        pa.field("width", pa.int32(), nullable=True),
        pa.field("height", pa.int32(), nullable=True),
        pa.field("longest_edge", pa.int32(), nullable=True),
        pa.field("file_size", pa.int64(), nullable=True),
        pa.field("needs_resize", pa.bool_(), nullable=False),
        pa.field("header_status", pa.dictionary(pa.int32(), pa.string()), nullable=False),
        pa.field("error_message", pa.string(), nullable=True),
    ]
)

AUDIT_PRIMARY_KEY: tuple[str, ...] = ("source_id",)
ANNOTATIONS_PRIMARY_KEY: tuple[str, ...] = ("source_id", "image_id", "row", "col")
IMAGES_PRIMARY_KEY: tuple[str, ...] = ("source_id", "image_id")

OUTPUT_SCHEMAS: dict[str, pa.Schema] = {
    "audit": AUDIT_SCHEMA,
    "annotations": ANNOTATIONS_SCHEMA,
    "images": IMAGES_SCHEMA,
}


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not satisfy a declared parquet schema."""


def validate(df, schema: pa.Schema) -> None:
    """Verify that ``df`` has every column the schema declares.

    We don't check dtypes here — the writer in ``io.write_parquet_deterministic`` casts everything
    to the declared types. This function exists to catch the common case where a column is silently
    dropped or renamed in upstream code.
    """
    declared = {field.name for field in schema}
    actual = set(df.columns)
    missing = declared - actual
    if missing:
        raise SchemaValidationError(
            f"DataFrame is missing required columns for schema: {sorted(missing)}"
        )
