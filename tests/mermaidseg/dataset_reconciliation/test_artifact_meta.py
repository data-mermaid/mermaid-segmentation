"""Artifact meta helpers: sidecar path, group-to-source parsing, repo-relative paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mermaidseg.dataset_reconciliation.artifact_meta import (
    check_label_keys,
    repo_relative,
    select_groups,
    source_id_of_group,
)
from mermaidseg.io import REPO_ROOT


@pytest.mark.parametrize(("group", "source_id"), [("s007", "007"), ("s12", "12")])
def test_a_group_maps_to_its_digits_as_written(group, source_id):
    assert source_id_of_group(group) == source_id


@pytest.mark.parametrize("group", ["7", "s1a"])
def test_a_group_that_is_not_s_and_digits_raises(group):
    with pytest.raises(ValueError, match="not s<source id>"):
        source_id_of_group(group)


def test_a_path_inside_the_repo_is_relative_and_one_outside_stays_absolute(tmp_path):
    assert repo_relative(REPO_ROOT / "configs" / "x.csv") == str(Path("configs") / "x.csv")
    assert repo_relative(tmp_path / "x.csv") == str((tmp_path / "x.csv").resolve())


IMAGES = pd.DataFrame({"image_id": ["a", "b", "c"], "group": ["s1", "s2", "s3"]})


def test_select_groups_keeps_only_the_rows_of_the_given_groups():
    out = select_groups("coralnet", IMAGES, ["s1", "s3"])

    assert out["image_id"].tolist() == ["a", "c"]


def test_select_groups_names_the_dataset_and_the_sorted_unknown_groups():
    with pytest.raises(
        ValueError, match=r"coralnet: groups not in the dataset version: \['s8', 's9'\]"
    ):
        select_groups("coralnet", IMAGES, ["s9", "s1", "s8"])


def test_check_label_keys_names_the_dataset_and_the_sorted_unique_missing_ids():
    label_ids = pd.Series([9, 1, 9, 4])
    keys = pd.Series(["x", "y", None, None])

    with pytest.raises(
        ValueError, match=r"coralnet: label ids missing from labels.parquet: \[4, 9\]"
    ):
        check_label_keys("coralnet", label_ids, keys)
