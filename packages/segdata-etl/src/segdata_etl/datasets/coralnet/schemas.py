"""PyArrow schemas for the CoralNet annotations and images parquets.

Single source of truth for column names, dtypes, and primary keys. The writer in
:mod:`segdata_etl.datasets.coralnet.io` casts every DataFrame to the declared schema before writing so byte
output is deterministic and downstream readers can rely on the column types.
"""

from __future__ import annotations

import pyarrow as pa

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

ANNOTATIONS_PRIMARY_KEY: tuple[str, ...] = ("source_id", "image_id", "row", "col")
IMAGES_PRIMARY_KEY: tuple[str, ...] = ("source_id", "image_id")
