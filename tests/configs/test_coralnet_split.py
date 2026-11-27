"""Invariants of configs/splits/coralnet.csv, its meta file and the data config that uses it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mermaidseg.data.config import DataSection, entry_split_groups, sync_plan
from segdata_spec.splits import read_split_file

_REPO = Path(__file__).resolve().parents[2]
_SPLITS = _REPO / "configs" / "splits"


@pytest.fixture(scope="module")
def split_table() -> pd.DataFrame:
    return pd.read_csv(_SPLITS / "coralnet.csv", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def meta() -> dict:
    return yaml.safe_load((_SPLITS / "coralnet.meta.yaml").read_text())


@pytest.fixture(scope="module")
def data_cfg() -> dict:
    return yaml.safe_load((_REPO / "configs" / "data_config.yaml").read_text())["data"]


def test_committed_data_config_selects_coralnet_train_val_for_training_and_test_held_out(
    split_table, data_cfg
):
    entry = DataSection.model_validate(data_cfg).datasets["coralnet"]
    train = entry_split_groups(entry, "train", _REPO)
    test = entry_split_groups(entry, "test", _REPO)

    assert train is not None and test is not None
    assert train == set(split_table.loc[split_table["split"].isin(["train", "val"]), "group"])
    assert test == set(split_table.loc[split_table["split"] == "test", "group"])
    assert sync_plan(data_cfg, _REPO).selection["coralnet"] == train | test


def test_group_is_s_prefixed_source_id(split_table):
    wrong = split_table[split_table["group"] != "s" + split_table["source_id"]]
    assert wrong.empty, wrong.to_dict("records")


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


def test_excluded_rows_have_blank_split_group_and_read_through_read_split_file(split_table):
    excluded = split_table[split_table["split"] == "excluded"]
    assert not excluded.empty
    assert (excluded["split_group"] == "").all()
    listed = set(read_split_file(_SPLITS / "coralnet.csv").column("group").to_pylist())
    assert listed == set(split_table["group"])
