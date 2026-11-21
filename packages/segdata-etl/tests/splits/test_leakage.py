from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from segdata_etl import cli
from segdata_etl.splits.leakage import check_leakage

from .conftest import BUCKET, FixtureCache, images_table

IMAGE_ROWS: list[tuple[str, str, str | None, int]] = [
    ("t1", "s1", "K=", 10),
    ("t2", "s1", "Q=", 5),
    ("v1", "s2", "K=", 10),
    ("v2", "s2", "M=", 10),
    ("w1", "s3", "M=", 10),
    ("w2", "s3", "Q=", 6),
    ("x1", "s4", "Z=", 1),
]


def _images() -> pd.DataFrame:
    return images_table(IMAGE_ROWS).to_pandas()


def _split(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["group", "split_group", "split"])


LEAKY = _split([("s1", "0", "train"), ("s2", "0", "val"), ("s3", "1", "val"), ("s4", "2", "test")])


def test_counts_eval_images_whose_key_is_in_another_split():
    per_source = check_leakage(LEAKY, _images()).per_source
    got = {
        (r.eval_split, r.other_split, r.group): (r.n_images, r.n_dup)
        for r in per_source.itertuples()
    }
    # v1 repeats t1. v2/w1 share a key inside val; w2 matches t2's crc but not its size.
    assert got == {
        ("val", "train", "s2"): (2, 1),
        ("val", "train", "s3"): (2, 0),
        ("test", "train", "s4"): (1, 0),
        ("test", "val", "s4"): (1, 0),
    }


def test_lists_cross_split_source_pairs():
    pairs = check_leakage(LEAKY, _images()).source_pairs
    assert pairs.to_dict("records") == [
        {"group_a": "s1", "group_b": "s2", "split_a": "train", "split_b": "val", "n_shared": 1}
    ]


def test_flags_split_groups_spanning_splits():
    spanning = check_leakage(LEAKY, _images()).spanning_split_groups
    assert spanning.to_dict("records") == [{"split_group": "0", "splits": "train;val"}]


def test_has_leakage_is_true_when_only_a_split_group_spans_splits():
    split = _split([("s1", "0", "train"), ("s3", "0", "val")])
    assert check_leakage(split, _images()).has_leakage


def _leakage_argv(
    fixture_cache: FixtureCache,
    split_file: Path,
    out_dir: Path,
    rows: list[tuple[str, str, str | None, int]] = IMAGE_ROWS,
) -> list[str]:
    pin = fixture_cache.dataset("coralnet", "20261120-aaaaaaa", images=images_table(rows))
    return [
        "split",
        "leakage",
        "--split-file",
        str(split_file),
        "--version",
        pin.version,
        "--manifest-sha256",
        pin.manifest_sha256,
        "--cache-dir",
        str(fixture_cache.cache_dir),
        "--bucket",
        BUCKET,
        "--out-dir",
        str(out_dir),
    ]


def test_cli_exits_1_on_leakage(tmp_path, fixture_cache):
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns1,train\ns2,val\ns3,val\ns4,test\n")
    assert cli.main(_leakage_argv(fixture_cache, split_file, tmp_path / "out")) == 1


def test_cli_refuses_a_listed_source_with_a_null_crc(tmp_path, fixture_cache):
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns1,train\ns2,train\ns3,train\ns4,test\n")
    rows = [*IMAGE_ROWS, ("x2", "s4", None, 1), ("x3", "s2", None, 1)]
    argv = _leakage_argv(fixture_cache, split_file, tmp_path / "out", rows)
    with pytest.raises(ValueError, match=r"2 images.*\['s2', 's4'\]"):
        cli.main(argv)


def test_cli_ignores_a_null_crc_in_a_source_the_split_file_does_not_list(tmp_path, fixture_cache):
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns1,train\ns2,train\ns3,train\ns4,test\n")
    rows = [*IMAGE_ROWS, ("y1", "s9", None, 1)]
    assert cli.main(_leakage_argv(fixture_cache, split_file, tmp_path / "out", rows)) == 0


def test_cli_exits_0_when_clean(tmp_path, fixture_cache):
    split_file = tmp_path / "coralnet.csv"
    split_file.write_text("group,split\ns1,train\ns2,train\ns3,train\ns4,test\n")
    assert cli.main(_leakage_argv(fixture_cache, split_file, tmp_path / "out")) == 0
