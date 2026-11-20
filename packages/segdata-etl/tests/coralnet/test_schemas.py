"""Tests for the parquet schema definitions and the validate() helper."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from segdata_etl.datasets.coralnet.schemas import (
    IMAGES_SCHEMA,
    SchemaValidationError,
    validate,
)


def _empty_df(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in schema})


def test_validate_raises_on_missing_column():
    df = _empty_df(IMAGES_SCHEMA).drop(columns=["needs_resize"])
    with pytest.raises(SchemaValidationError, match="needs_resize"):
        validate(df, IMAGES_SCHEMA)
