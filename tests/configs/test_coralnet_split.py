"""Invariants of configs/splits/coralnet.csv and its meta file.

The whitelist-equality test goes away when the whitelists are removed from
configs/data_config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from segdata_spec.splits import read_split_file

_REPO = Path(__file__).resolve().parents[2]
_SPLITS = _REPO / "configs" / "splits"


@pytest.fixture(scope="module")
def split_table() -> pd.DataFrame:
    return pd.read_csv(_SPLITS / "coralnet.csv", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def meta() -> dict:
    return yaml.safe_load((_SPLITS / "coralnet.meta.yaml").read_text())


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_split_sources_equal_data_config_whitelist(split_table, split):
    cfg = yaml.safe_load((_REPO / "configs" / "data_config.yaml").read_text())
    whitelist = cfg["data"]["coralnet"][split]["whitelist_sources"]
    in_split = split_table.loc[split_table["split"] == split, "source_id"]
    assert sorted(in_split.astype(int)) == sorted(whitelist)


def test_group_is_s_prefixed_source_id(split_table):
    wrong = split_table[split_table["group"] != "s" + split_table["source_id"]]
    assert wrong.empty, wrong.to_dict("records")


def test_committed_split_file_reads_through_read_split_file():
    assert read_split_file(_SPLITS / "coralnet.csv").num_rows > 0


def test_meta_overrides_match_csv(split_table, meta):
    rows = split_table.set_index("source_id")
    for o in meta["overrides"]["sources"]:
        sid = str(o["source_id"])
        assert sid in rows.index, sid
        row = rows.loc[sid]
        assert o["from"] != o["to"], sid
        assert row["split"] == o["to"], sid
        assert row["group"] == o["group"], sid
        assert row["split_group"] == str(o["split_group"]), sid
