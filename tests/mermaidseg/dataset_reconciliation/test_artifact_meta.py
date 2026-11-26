"""Artifact meta helpers: sidecar path, group-to-source parsing, repo-relative paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import (
    check_label_keys,
    concept_sidecar_meta,
    meta_path,
    read_sidecar,
    repo_relative,
    select_groups,
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


def test_read_sidecar_names_the_missing_key_and_the_builder(tmp_path):
    csv = tmp_path / "x.csv"
    meta_path(csv).write_text(yaml.safe_dump({"a": 1}))

    with pytest.raises(ValueError, match=r"\['b'\].*scripts/build_x\.py"):
        read_sidecar(csv, artifact="thing", builder="build_x", required=("a", "b"))


def test_read_sidecar_returns_the_dict_when_required_keys_are_present(tmp_path):
    csv = tmp_path / "x.csv"
    meta_path(csv).write_text(yaml.safe_dump({"a": 1, "b": 2}))

    meta = read_sidecar(csv, artifact="thing", builder="build_x", required=("a", "b"))

    assert meta == {"a": 1, "b": 2}


def test_concept_sidecar_meta_hashes_inputs_and_keeps_the_field_order(tmp_path):
    data_config = tmp_path / "data.yaml"
    concepts = tmp_path / "concepts.csv"
    schema_config = tmp_path / "schema.yaml"
    data_config.write_bytes(b"data: {}\n")
    concepts.write_bytes(b"a,b\n1,2\n")
    schema_config.write_bytes(b"schema\n")
    git = {"git_sha": "abc", "git_dirty": True}

    meta = concept_sidecar_meta(data_config, concepts, schema_config, git)

    assert list(meta) == [
        "data_config",
        "data_config_sha1",
        "concepts_csv",
        "concepts_csv_sha1",
        "schema_config",
        "created",
        "git_sha",
        "git_dirty",
    ]
    assert meta["data_config_sha1"] == hashlib.sha1(b"data: {}\n").hexdigest()
    assert meta["concepts_csv_sha1"] == hashlib.sha1(b"a,b\n1,2\n").hexdigest()
    assert meta["data_config"] == str(data_config.resolve())
    assert meta["schema_config"] == str(schema_config.resolve())
    assert (meta["git_sha"], meta["git_dirty"]) == ("abc", True)


IMAGES = pd.DataFrame({"image_id": ["a", "b", "c"], "group": ["s1", "s2", "s3"]})


def test_select_groups_keeps_all_images_when_groups_is_none():
    assert select_groups("coralnet", IMAGES, None) is IMAGES


def test_select_groups_keeps_only_the_rows_of_the_given_groups():
    out = select_groups("coralnet", IMAGES, ["s1", "s3"])

    assert out["image_id"].tolist() == ["a", "c"]


def test_select_groups_names_the_dataset_and_the_sorted_unknown_groups():
    with pytest.raises(
        ValueError, match=r"coralnet: groups not in the dataset version: \['s8', 's9'\]"
    ):
        select_groups("coralnet", IMAGES, ["s9", "s1", "s8"])


def test_check_label_keys_passes_when_every_label_has_a_key():
    check_label_keys("coralnet", pd.Series([1, 2]), pd.Series(["a", "b"]))


def test_check_label_keys_names_the_dataset_and_the_sorted_unique_missing_ids():
    label_ids = pd.Series([9, 1, 9, 4])
    keys = pd.Series(["x", "y", None, None])

    with pytest.raises(
        ValueError, match=r"coralnet: label ids missing from labels.parquet: \[4, 9\]"
    ):
        check_label_keys("coralnet", label_ids, keys)
