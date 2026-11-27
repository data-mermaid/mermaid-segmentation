"""Tests for the pyspacer ``.featurevector`` reader."""

from __future__ import annotations

import io

import numpy as np
import pytest

from segdata_etl.common.featurevector import (
    FeatureVectorsInvalid,
    VectorFacts,
    refuse_invalid,
    vector_points,
    vector_problem,
)


def featurevector_bytes(points: list[tuple[int, int]], dim: int = 2) -> bytes:
    """A pyspacer ``.featurevector``: an npz of ``meta``, ``rows``, ``cols`` and ``feat``."""
    buf = io.BytesIO()
    np.savez(
        buf,
        meta=np.array([1, 0, 0], dtype=np.int64),
        rows=np.array([p[0] for p in points], dtype=np.uint16),
        cols=np.array([p[1] for p in points], dtype=np.uint16),
        feat=np.zeros((len(points), dim), dtype=np.float64),
    )
    return buf.getvalue()


def npz_bytes(
    *,
    rows: np.ndarray | None = None,
    feat: np.ndarray | None = None,
) -> bytes:
    """A one-point ``.featurevector`` at (1, 3), width 2, with ``rows`` or ``feat`` replaced."""
    buf = io.BytesIO()
    np.savez(
        buf,
        meta=np.array([1, 0, 0], dtype=np.int64),
        rows=np.array([1], dtype=np.uint16) if rows is None else rows,
        cols=np.array([3], dtype=np.uint16),
        feat=np.zeros((1, 2), dtype=np.float64) if feat is None else feat,
    )
    return buf.getvalue()


def test_vector_points_pairs_rows_with_cols():
    assert vector_points(featurevector_bytes([(1, 3), (2, 4)])) == {(1, 3), (2, 4)}


@pytest.mark.parametrize(
    ("data", "points", "dim", "reason"),
    [
        (b'{"rows": [1], "cols": [3]}', [(1, 3)], 2, "unreadable"),
        (featurevector_bytes([(1, 3)], dim=4), [(1, 3)], 1280, "width 4, expected 1280"),
        (featurevector_bytes([(1, 3)]), [(1, 3), (2, 4)], 2, "lacks 1 of 2 point"),
        (npz_bytes(feat=np.zeros((2, 2))), [(1, 3)], 2, "feat has 2 rows for 1 point"),
        (npz_bytes(feat=np.zeros((1, 2), dtype=np.int64)), [(1, 3)], 2, "not float"),
        (npz_bytes(rows=np.array([1.0])), [(1, 3)], 2, "not integer"),
    ],
    ids=["json", "width", "gap", "row-count", "int-feat", "float-rows"],
)
def test_vector_problem_names_why_a_vector_is_refused(data, points, dim, reason):
    n_points, problem = vector_problem(data, points, dim)

    assert problem is not None and reason in problem


def test_vector_problem_is_none_for_a_covering_vector_of_the_right_width():
    data = featurevector_bytes([(1, 3), (2, 4), (9, 9)])

    assert vector_problem(data, [(1, 3), (2, 4), (1, 3)], 2) == (2, None)


def test_refuse_invalid_lists_every_refused_key_with_its_reason():
    facts = {
        "a": VectorFacts("e", 1, "c", 1, None),
        "b": VectorFacts("e", 1, "c", 1, "width 4, expected 1280"),
        "c": VectorFacts("e", 1, "c", 0, "unreadable: bad zip"),
    }
    keys = {"a": "s3://v/a", "b": "s3://v/b", "c": "s3://v/c"}

    with pytest.raises(FeatureVectorsInvalid) as raised:
        refuse_invalid(facts, keys)

    message = str(raised.value)
    assert message.startswith("2 feature vector(s) refused")
    assert "s3://v/b: width 4, expected 1280" in message
    assert "s3://v/c: unreadable: bad zip" in message
    assert "s3://v/a" not in message
