"""write_json, write_csv and fmt."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mermaidseg.evaluation.reporting import fmt, write_csv, write_json


def test_write_json_converts_numpy_values_and_replaces_atomically(tmp_path):
    path = tmp_path / "deep" / "dir" / "metrics.json"
    payload = {"n": np.int64(3), "x": np.float32(0.5), "arr": np.array([1, 2]), "p": tmp_path}
    write_json(path, payload)
    assert json.loads(path.read_text()) == {"n": 3, "x": 0.5, "arr": [1, 2], "p": str(tmp_path)}
    assert [f.name for f in path.parent.iterdir()] == ["metrics.json"]

    write_json(path, {"second": 1})
    assert json.loads(path.read_text()) == {"second": 1}
    assert [f.name for f in path.parent.iterdir()] == ["metrics.json"]


def test_write_csv_uses_the_first_rows_keys_and_drops_extras_in_later_rows(tmp_path):
    path = tmp_path / "sub" / "rows.csv"
    write_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": 4, "extra": 9}])
    assert path.read_text().splitlines() == ["a,b", "1,2", "3,4"]


def test_write_csv_honours_an_explicit_column_order(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv(path, [{"a": 1, "b": 2}], columns=["b", "a"])
    assert path.read_text().splitlines() == ["b,a", "2,1"]


def test_write_csv_of_no_rows_creates_an_empty_file(tmp_path):
    path = tmp_path / "sub" / "rows.csv"
    write_csv(path, [])
    assert path.read_text() == ""


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (None, 4, "n/a"),
        (float("nan"), 4, "n/a"),
        (0.123456, 4, "0.1235"),
        ("text", 4, "text"),
    ],
)
def test_fmt_renders_numbers_to_fixed_digits_and_missing_values_as_na(value, digits, expected):
    assert fmt(value, digits) == expected
