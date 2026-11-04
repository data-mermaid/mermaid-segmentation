"""Tests for the parquet schema definitions and the validate() helper."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from mermaidseg.datasets.coralnet.etl.schemas import (
    AUDIT_SCHEMA,
    SchemaValidationError,
    validate,
)


def _empty_df(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({field.name: pd.Series(dtype="object") for field in schema})


def test_validate_passes_when_all_columns_present():
    validate(_empty_df(AUDIT_SCHEMA), AUDIT_SCHEMA)  # must not raise


def test_validate_raises_on_missing_column():
    df = _empty_df(AUDIT_SCHEMA).drop(columns=["is_complete"])
    with pytest.raises(SchemaValidationError, match="is_complete"):
        validate(df, AUDIT_SCHEMA)
