"""CoralNet ids stay numeric: labels are int strings and source filters compare as ints.

Hermetic: the parquet read and boto3 client are stubbed.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

import mermaidseg.datasets.coralnet.coralnet_dataset as cnd
from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset


def _make_dataset(monkeypatch, df_annotations, **kwargs) -> CoralNetDataset:
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: df_annotations.copy())
    monkeypatch.setattr(cnd.boto3, "client", lambda *a, **k: MagicMock())
    return CoralNetDataset(**kwargs)


def _df(coralnet_id: list, source_id: list | None = None) -> pd.DataFrame:
    n = len(coralnet_id)
    return pd.DataFrame(
        {
            "source_id": source_id or [1] * n,
            "image_id": [f"img{i}" for i in range(n)],
            "row": [0] * n,
            "col": [0] * n,
            "coralnet_id": coralnet_id,
        }
    )


def test_float_coralnet_ids_become_int_strings(monkeypatch):
    """A float64 column (e.g. after a null was dropped upstream) must not yield
    "82.0"."""
    ds = _make_dataset(monkeypatch, _df([82.0, 83.0]))
    assert sorted(ds.source_name2id) == ["82", "83"]


@pytest.mark.parametrize("bad", [[82, None], [82.5, 83.0]], ids=["null", "non-integral"])
def test_invalid_coralnet_ids_raise(monkeypatch, bad):
    with pytest.raises(ValueError, match="coralnet_id"):
        _make_dataset(monkeypatch, _df(bad))


def test_string_whitelist_ids_match_int_source_ids(monkeypatch):
    """A quoted id in YAML must still select its source, not silently match nothing."""
    ds = _make_dataset(monkeypatch, _df([82, 83], source_id=[1, 2]), whitelist_sources=["1"])
    assert ds.df_images["source_id"].tolist() == [1]


def test_whitelist_id_absent_from_data_raises(monkeypatch):
    with pytest.raises(ValueError, match=r"whitelist_sources.*\[99\]"):
        _make_dataset(monkeypatch, _df([82, 83], source_id=[1, 2]), whitelist_sources=[1, 99])


def test_blacklist_id_absent_from_data_warns(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=cnd.__name__):
        ds = _make_dataset(
            monkeypatch, _df([82, 83], source_id=[1, 2]), blacklist_sources=["2", 99]
        )
    assert ds.df_images["source_id"].tolist() == [1]
    assert "blacklist_sources" in caplog.text and "99" in caplog.text
