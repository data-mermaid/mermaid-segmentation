"""Tests for the parquet schema definitions and the validate() helper."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from coralnet_dataset.schemas import (
    ANNOTATIONS_SCHEMA,
    IMAGES_SCHEMA,
    SchemaValidationError,
    validate,
)


def _empty_df(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in schema})


@pytest.mark.parametrize(
    "schema",
    [ANNOTATIONS_SCHEMA, IMAGES_SCHEMA],
    ids=["annotations", "images"],
)
def test_validate_passes_when_all_columns_present(schema):
    df = _empty_df(schema)
    validate(df, schema)  # must not raise


def test_validate_raises_on_missing_column():
    df = _empty_df(IMAGES_SCHEMA).drop(columns=["needs_resize"])
    with pytest.raises(SchemaValidationError, match="needs_resize"):
        validate(df, IMAGES_SCHEMA)


def test_annotations_schema_has_status_column():
    """Status column is the productionized addition over the 30112025 parquet."""
    names = [f.name for f in ANNOTATIONS_SCHEMA]
    assert "status" in names


def test_images_schema_has_needs_resize_flag():
    names = [f.name for f in IMAGES_SCHEMA]
    assert "needs_resize" in names
    field = IMAGES_SCHEMA.field("needs_resize")
    assert pa.types.is_boolean(field.type)
