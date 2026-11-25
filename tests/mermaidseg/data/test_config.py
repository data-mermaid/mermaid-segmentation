"""sync_plan: the pins, group selection and strictness derived from the §8 data section."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mermaidseg.data.config import TAXONOMY_NAME, DataSection, entry_split_groups, sync_plan
from segdata_spec.resize import ResizeSpec

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_T = "c" * 64


def _entry(**kwargs: Any) -> dict[str, Any]:
    return {"version": "20261120-ab12cd3", "manifest_sha256": SHA_A, **kwargs}


def _write_split_file(root: Path) -> None:
    path = root / "splits" / "coralnet.csv"
    path.parent.mkdir()
    path.write_text("group,split,n_images\ng1,train,10\ng2,val,5\ng3,train,7\n")


def test_pins_carry_dataset_names_and_taxonomy_name(tmp_path: Path) -> None:
    cfg = {
        "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": SHA_T},
        "datasets": {
            "coralnet": _entry(train={"groups": ["a"]}),
            "mermaid": _entry(version="20261122-ab12cd3", manifest_sha256=SHA_B),
        },
    }
    plan = sync_plan(cfg, tmp_path)
    assert {n: (p.name, p.version, p.manifest_sha256) for n, p in plan.pins.items()} == {
        "coralnet": ("coralnet", "20261120-ab12cd3", SHA_A),
        "mermaid": ("mermaid", "20261122-ab12cd3", SHA_B),
    }
    assert plan.taxonomy is not None
    assert (plan.taxonomy.name, plan.taxonomy.version, plan.taxonomy.manifest_sha256) == (
        TAXONOMY_NAME,
        "20261121-ab12cd3",
        SHA_T,
    )
    assert TAXONOMY_NAME == "mermaid-benthic"


def test_no_taxonomy_gives_none(tmp_path: Path) -> None:
    assert (
        sync_plan({"datasets": {"d": _entry(train={"groups": ["a"]})}}, tmp_path).taxonomy is None
    )


def test_groups_are_unioned_across_blocks(tmp_path: Path) -> None:
    entry = _entry(train={"groups": ["a", "b"]}, val={"groups": ["c"]})
    plan = sync_plan({"datasets": {"d": entry}}, tmp_path)
    assert plan.selection == {"d": {"a", "b", "c"}}


def test_groups_all_in_any_block_selects_everything(tmp_path: Path) -> None:
    entry = _entry(train={"groups": ["a"]}, val={"groups": "all"})
    assert sync_plan({"datasets": {"d": entry}}, tmp_path).selection == {"d": None}


@pytest.mark.parametrize(
    ("use", "expected"),
    [(["train"], {"g1", "g3"}), (["train", "val"], {"g1", "g2", "g3"})],
    ids=["train", "trainval"],
)
def test_use_reads_split_file_relative_to_repo_root(
    tmp_path: Path, use: list[str], expected: set[str]
) -> None:
    _write_split_file(tmp_path)
    entry = _entry(split_file="splits/coralnet.csv", train={"use": use})
    assert sync_plan({"datasets": {"d": entry}}, tmp_path).selection == {"d": expected}


def test_use_blocks_union_with_group_blocks(tmp_path: Path) -> None:
    _write_split_file(tmp_path)
    entry = _entry(split_file="splits/coralnet.csv", train={"use": ["val"]}, val={"groups": ["x"]})
    assert sync_plan({"datasets": {"d": entry}}, tmp_path).selection == {"d": {"g2", "x"}}


def test_split_files_map_datasets_that_have_one_to_repo_root_paths(tmp_path: Path) -> None:
    cfg = {
        "datasets": {
            "with": _entry(split_file="splits/coralnet.csv", train={"groups": "all"}),
            "without": _entry(train={"groups": "all"}),
        }
    }
    assert sync_plan(cfg, tmp_path).split_files == {"with": tmp_path / "splits/coralnet.csv"}


def test_on_missing_images_defaults_to_fail_and_carries_drop(tmp_path: Path) -> None:
    cfg = {
        "datasets": {
            "a": _entry(train={"groups": "all"}),
            "b": _entry(train={"groups": "all"}, on_missing_images="drop"),
        }
    }
    assert sync_plan(cfg, tmp_path).on_missing_images == {"a": "fail", "b": "drop"}


@pytest.mark.parametrize(
    "entry",
    [
        _entry(train={"use": ["train"]}),
        _entry(split_file="s.csv", train={"use": ["train"], "groups": "all"}),
        _entry(train={}),
        _entry(train={"groups": "all", "bogus": 1}),
        _entry(train={"groups": "all"}, bogus=1),
        _entry(train={"groups": "all"}, on_missing_images="skip"),
        _entry(train={"groups": "all", "repeat": 0}),
        _entry(train={"groups": "all"}, resize={"max_edge": 100, "bogus": 1}),
    ],
    ids=[
        "use-without-split-file",
        "use-and-groups",
        "neither",
        "unknown-block-key",
        "unknown-entry-key",
        "bad-missing-mode",
        "repeat-below-one",
        "unknown-resize-key",
    ],
)
def test_invalid_entries_raise(tmp_path: Path, entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sync_plan({"datasets": {"d": entry}}, tmp_path)


def _plan_with_split_rows(root: Path, rows: list[str]) -> None:
    path = root / "s.csv"
    path.write_text("group,split\n" + "".join(f"{r}\n" for r in rows))
    cfg = {"datasets": {"d": _entry(split_file="s.csv", train={"use": ["train"]})}}
    sync_plan(cfg, root)


@pytest.mark.parametrize("bad", ["Val", "holdout", ""])
def test_split_value_outside_known_splits_raises(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match=f"'{bad}'"):
        _plan_with_split_rows(tmp_path, ["g1,train", f"g2,{bad}"])


@pytest.mark.parametrize(
    "rows",
    [["g1,train", "g1,val"], ["g1,train", "g1,train"]],
    ids=["across-splits", "within-split"],
)
def test_group_on_more_than_one_row_raises(tmp_path: Path, rows: list[str]) -> None:
    with pytest.raises(ValueError, match="g1"):
        _plan_with_split_rows(tmp_path, rows)


def test_unknown_section_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sync_plan({"datasets": {}, "bucket": "other"}, tmp_path)


def test_section_accepts_resize_block_keys_and_default() -> None:
    section = DataSection.model_validate(
        {
            "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
            "datasets": {
                "d": _entry(
                    resize={"max_edge": 1024, "filter": "bicubic", "jpeg_quality": 90},
                    train={
                        "groups": "all",
                        "transform": {"VerticalFlip": {"p": 1.0}},
                        "repeat": 3,
                        "annotators": ["a", "b"],
                    },
                )
            },
        }
    )
    entry = section.datasets["d"]
    assert entry.resize == ResizeSpec(max_edge=1024, filter="bicubic", jpeg_quality=90)
    assert entry.train is not None
    assert (entry.train.transform, entry.train.repeat, entry.train.annotators) == (
        {"VerticalFlip": {"p": 1.0}},
        3,
        ["a", "b"],
    )


def test_default_train_transform_fills_a_train_block_without_one() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "datasets": {"d": _entry(train={"groups": "all"})},
    }
    train = DataSection.model_validate(cfg).datasets["d"].train
    assert train is not None
    assert train.transform == {"HorizontalFlip": {"p": 0.5}}


def test_a_blocks_own_transform_wins_over_the_default() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "datasets": {"d": _entry(train={"groups": "all", "transform": {}})},
    }
    train = DataSection.model_validate(cfg).datasets["d"].train
    assert train is not None
    assert train.transform == {}


def test_default_never_creates_a_train_block() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "datasets": {"d": _entry(val={"groups": "all"})},
    }
    entry = DataSection.model_validate(cfg).datasets["d"]
    assert entry.train is None
    assert entry.val is not None
    assert entry.val.transform is None


@pytest.mark.parametrize("split", ["val", "test"])
def test_repeat_on_an_eval_block_raises(split: str) -> None:
    with pytest.raises(ValidationError, match="repeat"):
        DataSection.model_validate(
            {"datasets": {"d": _entry(**{split: {"groups": "all", "repeat": 2}})}}
        )


@pytest.mark.parametrize("split", ["val", "test"])
def test_transform_on_an_eval_block_raises(split: str) -> None:
    with pytest.raises(ValidationError, match="transform"):
        DataSection.model_validate(
            {"datasets": {"d": _entry(**{split: {"groups": "all", "transform": {}}})}}
        )


def test_default_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        DataSection.model_validate(
            {
                "default": {"val": {"transform": {}}},
                "datasets": {"d": _entry(train={"groups": "all"})},
            }
        )


def test_entry_split_groups_resolves_one_split_block(tmp_path: Path) -> None:
    _write_split_file(tmp_path)
    entry = DataSection.model_validate(
        {
            "datasets": {
                "d": _entry(
                    split_file="splits/coralnet.csv",
                    train={"use": ["train"]},
                    val={"groups": "all"},
                    test={"groups": ["z"]},
                )
            }
        }
    ).datasets["d"]
    assert entry_split_groups(entry, "train", tmp_path) == {"g1", "g3"}
    assert entry_split_groups(entry, "val", tmp_path) is None
    assert entry_split_groups(entry, "test", tmp_path) == {"z"}
