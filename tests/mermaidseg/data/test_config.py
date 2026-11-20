"""sync_plan: the pins, group selection and strictness derived from the §8 data section."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mermaidseg.data.config import TAXONOMY_NAME, sync_plan

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
    ],
    ids=[
        "use-without-split-file",
        "use-and-groups",
        "neither",
        "unknown-block-key",
        "unknown-entry-key",
        "bad-missing-mode",
    ],
)
def test_invalid_entries_raise(tmp_path: Path, entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sync_plan({"datasets": {"d": entry}}, tmp_path)


def test_unknown_section_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sync_plan({"datasets": {}, "bucket": "other"}, tmp_path)
