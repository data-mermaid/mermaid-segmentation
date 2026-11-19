"""build_trainval_data_config: the committed trainval config is the base config, rebuilt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.build_trainval_data_config import (
    HEADER,
    _check,
    _whitelist_ids,
    build_final_data_config,
)

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_trainval_config_is_rebuilt_from_the_base_config():
    rebuilt = build_final_data_config((_CONFIGS / "data_config.yaml").read_text(), header=HEADER)
    assert rebuilt == (_CONFIGS / "data_config_trainval.yaml").read_text()


_CORALNET = """\
  coralnet:
    train:
      whitelist_sources: [{train}]
    val:
      whitelist_sources: [{val}]
      image_blacklist: null
      annotation_overrides: null
    test:
      whitelist_sources: [9]
"""
_CORALSCAPES = """\
  coralscapes_v2:
    train:
      split: train
    val:
      split: validation
    test:
      split: test
"""


def _source_config(
    *,
    train: str = "1, 2",
    val: str = "2, 3",
    coralnet: str | None = None,
    scapes: str | None = None,
) -> str:
    coralnet = _CORALNET.format(train=train, val=val) if coralnet is None else coralnet
    scapes = _CORALSCAPES if scapes is None else scapes
    return f"data:\n  local_cache_write_through: true\n{coralnet}{scapes}"


def _build(text: str) -> dict:
    return yaml.safe_load(build_final_data_config(text, header="# header\n"))["data"]


def test_validation_sources_join_the_train_whitelist_without_repeating_shared_ids():
    data = _build(_source_config(train="1, 2", val="2, 3"))

    assert data["coralnet"]["train"]["whitelist_sources"] == [1, 2, 3]
    assert data["coralnet"]["val"] is None
    assert data["coralnet"]["test"] == {"whitelist_sources": [9]}


def test_coralscapes_validation_split_joins_train_and_test_stays_held_out():
    data = _build(_source_config())

    assert data["coralscapes_v2"]["train"]["split"] == ["train", "validation"]
    assert data["coralscapes_v2"]["val"] is None
    assert data["coralscapes_v2"]["test"]["split"] == "test"


def test_the_header_is_prepended_verbatim():
    out = build_final_data_config(_source_config(), header="# header\n")

    assert out.startswith("# header\ndata:\n")


@pytest.mark.parametrize(
    ("missing", "text"),
    [
        ("coralnet", _source_config(coralnet="")),
        ("coralscapes_v2", _source_config(scapes="")),
    ],
)
def test_a_missing_dataset_section_is_rejected(missing, text):
    with pytest.raises(ValueError, match=f"section '{missing}' not found"):
        build_final_data_config(text, header="")


def test_a_coralnet_section_without_a_val_block_is_rejected():
    coralnet = "  coralnet:\n    train:\n      whitelist_sources: [1]\n"

    with pytest.raises(ValueError, match="coralnet val block not found"):
        build_final_data_config(_source_config(coralnet=coralnet), header="")


def test_a_duplicate_id_in_the_train_whitelist_is_rejected():
    with pytest.raises(ValueError, match="duplicate sources"):
        build_final_data_config(_source_config(train="1, 1"), header="")


@pytest.mark.parametrize(
    ("scapes", "message"),
    [
        (
            "  coralscapes_v2:\n    val:\n      split: validation\n",
            "coralscapes_v2 train split not found",
        ),
        (
            "  coralscapes_v2:\n    train:\n      split: train\n",
            "coralscapes_v2 val block not found",
        ),
    ],
)
def test_a_coralscapes_section_missing_its_train_or_val_split_is_rejected(scapes, message):
    with pytest.raises(ValueError, match=message):
        build_final_data_config(_source_config(scapes=scapes), header="")


def test_a_source_config_that_already_has_a_banner_is_rejected():
    with pytest.raises(ValueError, match="starts with a comment"):
        build_final_data_config("# banner\n" + _source_config(), header="")


def test_a_whitelist_missing_from_a_body_is_rejected():
    with pytest.raises(ValueError, match="whitelist_sources not found"):
        _whitelist_ids("    train:\n      other: [1]\n")


def test_whitelist_ids_are_read_across_wrapped_lines():
    assert _whitelist_ids("      whitelist_sources: [1, 2,\n                          3]\n") == [
        1,
        2,
        3,
    ]


def _write_derived(tmp_path: Path, **overrides) -> Path:
    data: dict[str, dict[str, Any]] = {
        "coralnet": {"val": None},
        "coralscapes_v2": {
            "train": {"split": ["train", "validation"]},
            "val": None,
            "test": {"split": "test"},
        },
    }
    data["coralnet"].update(overrides.get("coralnet", {}))
    data["coralscapes_v2"].update(overrides.get("coralscapes_v2", {}))
    path = tmp_path / "derived.yaml"
    path.write_text(yaml.safe_dump({"data": data}))
    return path


def test_check_accepts_a_correctly_derived_config(tmp_path):
    _check(_write_derived(tmp_path))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"coralnet": {"val": {"whitelist_sources": [3]}}}, "coralnet.val was not cleared"),
        (
            {"coralscapes_v2": {"train": {"split": "train"}}},
            "coralscapes_v2 train split was not merged",
        ),
        (
            {"coralscapes_v2": {"val": {"split": "validation"}}},
            "coralscapes_v2.val was not cleared",
        ),
        (
            {"coralscapes_v2": {"test": {"split": "validation"}}},
            "coralscapes_v2 test split changed",
        ),
    ],
)
def test_check_rejects_a_config_that_misses_a_derivation_step(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _check(_write_derived(tmp_path, **overrides))
