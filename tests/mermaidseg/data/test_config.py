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
TAXONOMY = {"version": "20261121-ab12cd3", "manifest_sha256": SHA_T}


def _entry(**kwargs: Any) -> dict[str, Any]:
    return {"version": "20261120-ab12cd3", "manifest_sha256": SHA_A, **kwargs}


def _write_split_file(root: Path) -> None:
    path = root / "splits" / "coralnet.csv"
    path.parent.mkdir()
    path.write_text("group,split,n_images\ng1,train,10\ng2,val,5\ng3,train,7\n")


def test_pins_carry_dataset_names_and_taxonomy_name(tmp_path: Path) -> None:
    cfg = {
        "taxonomy": TAXONOMY,
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
    assert (plan.taxonomy.name, plan.taxonomy.version, plan.taxonomy.manifest_sha256) == (
        TAXONOMY_NAME,
        "20261121-ab12cd3",
        SHA_T,
    )


def test_groups_are_unioned_across_blocks(tmp_path: Path) -> None:
    entry = _entry(train={"groups": ["a", "b"]}, val={"groups": ["c"]})
    plan = sync_plan({"taxonomy": TAXONOMY, "datasets": {"d": entry}}, tmp_path)
    assert plan.selection == {"d": {"a", "b", "c"}}


def test_groups_all_in_any_block_selects_everything(tmp_path: Path) -> None:
    entry = _entry(train={"groups": ["a"]}, val={"groups": "all"})
    assert sync_plan({"taxonomy": TAXONOMY, "datasets": {"d": entry}}, tmp_path).selection == {
        "d": None
    }


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
    assert sync_plan({"taxonomy": TAXONOMY, "datasets": {"d": entry}}, tmp_path).selection == {
        "d": expected
    }


def test_use_blocks_union_with_group_blocks(tmp_path: Path) -> None:
    _write_split_file(tmp_path)
    entry = _entry(split_file="splits/coralnet.csv", train={"use": ["val"]}, val={"groups": ["x"]})
    assert sync_plan({"taxonomy": TAXONOMY, "datasets": {"d": entry}}, tmp_path).selection == {
        "d": {"g2", "x"}
    }


def test_split_files_map_datasets_that_have_one_to_repo_root_paths(tmp_path: Path) -> None:
    cfg = {
        "taxonomy": TAXONOMY,
        "datasets": {
            "with": _entry(split_file="splits/coralnet.csv", train={"groups": "all"}),
            "without": _entry(train={"groups": "all"}),
        },
    }
    assert sync_plan(cfg, tmp_path).split_files == {"with": tmp_path / "splits/coralnet.csv"}


@pytest.mark.parametrize(
    "entry",
    [
        _entry(train={"use": ["train"]}),
        _entry(split_file="s.csv", train={"use": ["train"], "groups": "all"}),
        _entry(train={}),
        _entry(train={"groups": "all"}, bogus=1),
    ],
    ids=[
        "use-without-split-file",
        "use-and-groups",
        "neither",
        "unknown-entry-key",
    ],
)
def test_invalid_entries_raise(tmp_path: Path, entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sync_plan({"taxonomy": TAXONOMY, "datasets": {"d": entry}}, tmp_path)


def test_default_train_transform_fills_a_train_block_without_one() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "taxonomy": TAXONOMY,
        "datasets": {"d": _entry(train={"groups": "all"})},
    }
    train = DataSection.model_validate(cfg).datasets["d"].train
    assert train is not None
    assert train.transform == {"HorizontalFlip": {"p": 0.5}}


def test_a_blocks_own_transform_wins_over_the_default() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "taxonomy": TAXONOMY,
        "datasets": {"d": _entry(train={"groups": "all", "transform": {}})},
    }
    train = DataSection.model_validate(cfg).datasets["d"].train
    assert train is not None
    assert train.transform == {}


def test_default_never_creates_a_train_block() -> None:
    cfg = {
        "default": {"train": {"transform": {"HorizontalFlip": {"p": 0.5}}}},
        "taxonomy": TAXONOMY,
        "datasets": {"d": _entry(val={"groups": "all"})},
    }
    entry = DataSection.model_validate(cfg).datasets["d"]
    assert entry.train is None
    assert entry.val is not None
    assert entry.val.transform is None


def test_entry_split_groups_resolves_one_split_block(tmp_path: Path) -> None:
    _write_split_file(tmp_path)
    entry = DataSection.model_validate(
        {
            "taxonomy": TAXONOMY,
            "datasets": {
                "d": _entry(
                    split_file="splits/coralnet.csv",
                    train={"use": ["train"]},
                    val={"groups": "all"},
                    test={"groups": ["z"]},
                )
            },
        }
    ).datasets["d"]
    assert entry_split_groups(entry, "train", tmp_path) == {"g1", "g3"}
    assert entry_split_groups(entry, "val", tmp_path) is None
    assert entry_split_groups(entry, "test", tmp_path) == {"z"}


def test_sync_plan_carries_each_entrys_resize(tmp_path: Path) -> None:
    data_cfg = {
        "taxonomy": TAXONOMY,
        "datasets": {
            "a": _entry(resize={"max_edge": 100}, train={"groups": "all"}),
            "b": _entry(train={"groups": "all"}),
        },
    }
    assert sync_plan(data_cfg, tmp_path).resize == {"a": ResizeSpec(max_edge=100), "b": None}


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize(
    ("key", "value"), [("repeat", 2), ("transform", {})], ids=["repeat", "transform"]
)
def test_train_only_keys_on_an_eval_block_raise(split: str, key: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=key):
        DataSection.model_validate(
            {
                "taxonomy": TAXONOMY,
                "datasets": {"d": _entry(**{split: {"groups": "all", key: value}})},
            }
        )


def test_a_data_section_without_a_taxonomy_pin_raises() -> None:
    with pytest.raises(ValidationError, match="taxonomy"):
        DataSection.model_validate({"datasets": {"d": _entry(train={"groups": "all"})}})
