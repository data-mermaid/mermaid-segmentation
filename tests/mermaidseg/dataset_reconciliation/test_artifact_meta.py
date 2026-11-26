"""Artifact meta helpers: sidecar path, group-to-source parsing, repo-relative paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from mermaidseg.dataset_reconciliation.artifact_meta import (
    meta_path,
    repo_relative,
    source_id_of_group,
)
from mermaidseg.io import REPO_ROOT


def test_meta_path_sits_next_to_the_csv(tmp_path):
    assert (
        meta_path(tmp_path / "coralnet_only_false.csv")
        == tmp_path / "coralnet_only_false.meta.yaml"
    )


@pytest.mark.parametrize(("group", "source_id"), [("s007", "007"), ("s12", "12")])
def test_a_group_maps_to_its_digits_as_written(group, source_id):
    assert source_id_of_group(group) == source_id


@pytest.mark.parametrize("group", ["7", "abc", "s", "s1a"])
def test_a_group_that_is_not_s_and_digits_raises(group):
    with pytest.raises(ValueError, match="not s<source id>"):
        source_id_of_group(group)


def test_a_path_inside_the_repo_is_relative_and_one_outside_stays_absolute(tmp_path):
    assert repo_relative(REPO_ROOT / "configs" / "x.csv") == str(Path("configs") / "x.csv")
    assert repo_relative(tmp_path / "x.csv") == str((tmp_path / "x.csv").resolve())
