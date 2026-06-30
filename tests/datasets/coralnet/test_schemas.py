"""Tests for the parquet schema definitions and the validate() helper."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from mermaidseg.datasets.coralnet.etl.schemas import (
    ANNOTATIONS_SCHEMA,
    AUDIT_SCHEMA,
    IMAGES_SCHEMA,
    OUTPUT_SCHEMAS,
    SchemaValidationError,
    validate,
)


def _empty_df(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in schema})


@pytest.mark.parametrize(
    "schema_name",
    sorted(OUTPUT_SCHEMAS.keys()),
)
def test_validate_passes_when_all_columns_present(schema_name):
    schema = OUTPUT_SCHEMAS[schema_name]
    df = _empty_df(schema)
    validate(df, schema)  # must not raise


def test_validate_raises_on_missing_column():
    df = _empty_df(AUDIT_SCHEMA).drop(columns=["is_complete"])
    with pytest.raises(SchemaValidationError, match="is_complete"):
        validate(df, AUDIT_SCHEMA)


def test_output_schemas_keyed_by_short_name():
    assert OUTPUT_SCHEMAS["audit"] is AUDIT_SCHEMA
    assert OUTPUT_SCHEMAS["annotations"] is ANNOTATIONS_SCHEMA
    assert OUTPUT_SCHEMAS["images"] is IMAGES_SCHEMA


def test_annotations_schema_has_status_column():
    """Status column is the productionized addition over the 30112025 parquet."""
    names = [f.name for f in ANNOTATIONS_SCHEMA]
    assert "status" in names


def test_images_schema_has_needs_resize_flag():
    names = [f.name for f in IMAGES_SCHEMA]
    assert "needs_resize" in names
    field = IMAGES_SCHEMA.field("needs_resize")
    assert pa.types.is_boolean(field.type)


def test_audit_schema_carries_is_complete_and_image_count_match():
    names = [f.name for f in AUDIT_SCHEMA]
    assert {"is_complete", "image_count_match"}.issubset(names)


def test_audit_schema_has_image_list_coverage_flag():
    names = {field.name for field in AUDIT_SCHEMA}
    assert "image_list_covers_annotations" in names
