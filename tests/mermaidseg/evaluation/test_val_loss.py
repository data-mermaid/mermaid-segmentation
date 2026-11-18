"""The validation-loss helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from mermaidseg.evaluation.val_loss import (
    _Accumulator,
    _bundle_config,
    _check_layout,
    binary_concept_names,
)

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


def _bundle(id2label: dict[int, str], concept_id2name: dict[int, str]) -> SimpleNamespace:
    return SimpleNamespace(id2label=id2label, concept_id2name=concept_id2name)


def test_layout_check_accepts_a_bundle_that_matches_the_registry():
    bundle = _bundle({0: "ignore", 1: "acropora", 2: "sand"}, dict(enumerate(NAMES, start=1)))
    registry = SimpleNamespace(target_id2label={1: "acropora", 2: "sand"})
    _check_layout(bundle, registry, NAMES)


def test_layout_check_names_a_differing_class():
    bundle = _bundle({0: "ignore", 1: "acropora", 2: "sand"}, dict(enumerate(NAMES, start=1)))
    registry = SimpleNamespace(target_id2label={1: "acropora", 2: "rubble"})
    with pytest.raises(ValueError, match="rubble"):
        _check_layout(bundle, registry, NAMES)


def test_layout_check_names_a_differing_concept():
    bundle = _bundle({0: "ignore", 1: "sand"}, dict(enumerate(NAMES, start=1)))
    registry = SimpleNamespace(target_id2label={1: "sand"})
    with pytest.raises(ValueError, match="'anthropogenic'"):
        _check_layout(bundle, registry, ["bleached", "other", "sand"])


def test_bundle_config_takes_run_sections_from_the_bundle_and_data_from_the_file(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"data": {"local_cache_write_through": False, "local_cache_dir": "x"}})
    )
    run = {
        "model": {"name": "dpt", "input_size": [8, 8]},
        "training": {"batch_size": 2, "taxonomic_loss": {"kl_weight": 1.0}, "loss": {"a": 1}},
    }
    bundle = SimpleNamespace(config=run, input_size=(32, 64))
    cfg = _bundle_config(data_yaml, bundle)
    assert cfg.model.name == "dpt"
    assert cfg.model.input_size == [32, 64]
    assert cfg.training.batch_size == 2
    assert cfg.data.local_cache_dir == "x"
    cfg.training.pop("taxonomic_loss")
    cfg.training.loss.a = 9
    assert "taxonomic_loss" in run["training"]
    assert run["training"]["loss"] == {"a": 1}
    assert run["model"]["input_size"] == [8, 8]
