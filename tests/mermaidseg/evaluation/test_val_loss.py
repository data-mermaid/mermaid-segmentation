"""The validation-loss helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mermaidseg.evaluation.val_loss import (
    _Accumulator,
    _check_layout,
    binary_concept_names,
    resolve_concept_mapping_path,
)
from mermaidseg.run_setup import REPO_ROOT

NAMES = ["bleached", "anthropogenic", "sand"]


def test_binary_names_follow_the_taxonomic_channels():
    value2id = {"kingdom": {"animalia": [1, 0], "plantae": [0, 1]}, "phylum": {"cnidaria": [1]}}
    channels = ["kingdom__animalia", "kingdom__plantae", "phylum__cnidaria", *NAMES]
    assert binary_concept_names(value2id, channels) == NAMES


def test_accumulator_averages_each_key_over_the_batches_it_appears_in():
    first, second = _Accumulator(), _Accumulator()
    first.add({"loss": 1.0, "multi_hot/bleached": 0.2})
    first.add({"loss": 3.0})
    second.add({"loss": 5.0, "multi_hot/bleached": 0.4})
    first.merge(second)
    assert first.means() == pytest.approx({"loss": 3.0, "multi_hot/bleached": 0.3})


def test_concept_mapping_path_resolves_to_the_repo_file_from_any_directory(monkeypatch, tmp_path):
    from mermaidseg.io import ConfigDict

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "class_to_concepts.csv").write_text("decoy\n")
    monkeypatch.chdir(tmp_path)
    cfg = ConfigDict({"concept_mapping_path": "configs/class_to_concepts.csv"})
    resolve_concept_mapping_path(cfg)
    assert cfg.concept_mapping_path == str(REPO_ROOT / "configs" / "class_to_concepts.csv")


def test_layout_check_uses_the_files_next_to_the_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "run" / "model_step100000"
    checkpoint.parent.mkdir()
    registry = SimpleNamespace(target_id2label={1: "acropora", 2: "sand"})
    _check_layout(checkpoint, registry, NAMES)  # no files: nothing to compare

    (checkpoint.parent / "id2label.json").write_text(
        json.dumps({"0": "ignore", "1": "acropora", "2": "sand"})
    )
    (checkpoint.parent / "concept_id2name.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(NAMES)})
    )
    _check_layout(checkpoint, registry, NAMES)
    with pytest.raises(ValueError, match="concept channels differ"):
        _check_layout(checkpoint, registry, NAMES[::-1])
    with pytest.raises(ValueError, match="target classes differ"):
        _check_layout(checkpoint, SimpleNamespace(target_id2label={1: "sand"}), NAMES)
