"""The validation-loss helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import TensorDataset

from mermaidseg.data.config import DataSection
from mermaidseg.evaluation import val_loss
from mermaidseg.evaluation.val_loss import (
    _Accumulator,
    _bundle_config,
    _check_layout,
    evaluate_val_loss,
    val_blocks_only,
)
from mermaidseg.io import ConfigDict

NAMES = ["bleached", "anthropogenic", "sand"]


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
        yaml.safe_dump(
            {
                "data": {
                    "datasets": {
                        "ds": {"version": "v1", "manifest_sha256": "x", "val": {"groups": "all"}}
                    }
                }
            }
        )
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
    assert cfg.data.datasets.ds.version == "v1"
    cfg.training.pop("taxonomic_loss")
    cfg.training.loss.a = 9
    assert "taxonomic_loss" in run["training"]
    assert run["training"]["loss"] == {"a": 1}
    assert run["model"]["input_size"] == [8, 8]


def test_val_blocks_only_keeps_the_val_block_of_each_dataset_that_has_one():
    section = DataSection.model_validate(
        {
            "datasets": {
                "a": {
                    "version": "v",
                    "manifest_sha256": "x",
                    "train": {"groups": "all"},
                    "val": {"groups": ["g1"]},
                    "test": {"groups": ["g2"]},
                },
                "b": {"version": "v", "manifest_sha256": "x", "train": {"groups": "all"}},
            }
        }
    )

    kept = val_blocks_only(section)

    assert list(kept.datasets) == ["a"]
    entry = kept.datasets["a"]
    assert (entry.train, entry.test) == (None, None)
    assert entry.val is not None and entry.val.groups == ["g1"]


def test_layout_check_reports_a_length_difference_when_one_list_extends_the_other():
    bundle = _bundle({0: "ignore", 1: "sand"}, dict(enumerate(NAMES, start=1)))
    registry = SimpleNamespace(target_id2label={1: "sand"})
    with pytest.raises(ValueError, match="lengths: bundle 3 vs registry 4"):
        _check_layout(bundle, registry, [*NAMES, "extra"])


class _Harness:
    """Stubs of the collaborators that read configs and datasets; the loop and outputs are real."""

    def __init__(self, monkeypatch, datasets: dict):
        self.cfg = ConfigDict(
            {
                "data": {"datasets": {}},
                "model": {"input_size": [8, 8]},
                "training": {"padding": 0, "taxonomic_loss": {"w": 1}, "loss": {}},
            }
        )
        self.datasets = datasets
        self.meta_cfg = None
        self.build_args: dict = {}
        registry = SimpleNamespace(
            target_id2label={1: "sand"}, concept_value2id={"bleached": {"not_given": 0}}
        )
        schema = SimpleNamespace(
            channel_id2name=lambda: {1: "bleached"}, binary_names=lambda: ["bleached"]
        )
        meta = SimpleNamespace(
            model=nn.Identity(),
            loss=object(),
            _to_target_labels=lambda labels: labels,
            _to_concept_labels=lambda labels: labels,
        )

        def build_meta_model(cfg, _registry, _device, model):
            self.meta_cfg = cfg
            return meta

        def loss(outputs, _loss_fn, _none, _target_labels, _target_concepts):
            value = outputs.mean()
            return value, None, None, {"classification": value * 2}

        monkeypatch.setattr(val_loss, "_bundle_config", lambda *_a: self.cfg)

        def build_datasets(section, cache_dir, **kwargs):
            self.build_args = {"cache_dir": cache_dir, **kwargs}
            return self.datasets

        monkeypatch.setattr(val_loss, "build_datasets", build_datasets)
        monkeypatch.setattr(val_loss, "build_registry", lambda *_a, **_k: (registry, schema))
        monkeypatch.setattr(val_loss, "build_meta_model", build_meta_model)
        monkeypatch.setattr(val_loss, "apply_concept_bottleneck_loss", loss)
        self.bundle = SimpleNamespace(
            id2label={0: "ignore", 1: "sand"},
            concept_id2name={1: "bleached"},
            input_size=(8, 8),
            path="bundle/path",
            model=nn.Identity(),
        )

    def run(self, tmp_path, **kwargs):
        return evaluate_val_loss(
            bundle=self.bundle,  # type: ignore[arg-type]
            data_config="data.yaml",
            cache_dir=kwargs.pop("cache_dir", "/cache"),
            output_dir=tmp_path,
            device=torch.device("cpu"),
            batch_size=1,
            num_workers=0,
            **kwargs,
        )


def _images(*values: float) -> TensorDataset:
    """One 3x2x2 image per value, filled with it, with a 2x2 label map."""
    images = torch.stack([torch.full((3, 2, 2), v) for v in values])
    return TensorDataset(images, torch.zeros(len(values), 2, 2))


def test_val_loss_is_averaged_per_dataset_and_pooled_over_all_batches(tmp_path, monkeypatch):
    harness = _Harness(
        monkeypatch, {("b_ds", "val"): _images(8.0), ("a_ds", "val"): _images(1.0, 3.0)}
    )
    result = harness.run(tmp_path, max_images_per_dataset=None)
    assert result["per_dataset"] == {
        "a_ds": {"classification": 4.0, "loss": 2.0},
        "b_ds": {"classification": 16.0, "loss": 8.0},
    }
    # Pooled over the three batches, not the mean of the two dataset means (5.0).
    assert result["overall"] == {"classification": 8.0, "loss": 4.0}
    assert result["n_images"] == {"a_ds": 2, "b_ds": 1}
    assert list(result["per_dataset"]) == ["a_ds", "b_ds"]
    on_disk = json.loads((tmp_path / "val_loss" / "metrics.json").read_text())
    assert on_disk["overall"] == result["overall"]
    assert on_disk["n_images"] == {"a_ds": 2, "b_ds": 1}


def test_val_loss_scores_a_seeded_ascending_subset_of_each_dataset(tmp_path, monkeypatch):
    seen: list[float] = []

    def loss(outputs, *_rest):
        seen.append(float(outputs.mean()))
        return outputs.mean(), None, None, {}

    harness = _Harness(monkeypatch, {("ds", "val"): _images(*map(float, range(8)))})
    monkeypatch.setattr(val_loss, "apply_concept_bottleneck_loss", loss)
    result = harness.run(tmp_path, max_images_per_dataset=4, seed=3)
    assert result["n_images"] == {"ds": 4}
    assert seen == sorted(seen)
    assert len(set(seen)) == 4
    first = list(seen)
    seen.clear()
    harness.run(tmp_path, max_images_per_dataset=4, seed=3)
    assert seen == first
    subsets = set()
    for other in range(5):
        seen.clear()
        harness.run(tmp_path, max_images_per_dataset=4, seed=other)
        subsets.add(tuple(seen))
    assert len(subsets) > 1


def test_val_loss_overrides_the_run_config_for_scoring(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, {("ds", "val"): _images(1.0)})
    result = harness.run(
        tmp_path,
        name="scored",
        cache_dir="/the/cache",
        image_ids={("ds", "val"): ["i1"]},
        per_pixel_loss_weight=0.25,
        per_image_loss_weight=0.75,
        seed=5,
    )
    cfg = harness.meta_cfg
    assert cfg.run_name == "scored"
    assert "taxonomic_loss" not in cfg.training
    assert cfg.training.distributed is False
    assert (cfg.training.loss.per_pixel_loss_weight, cfg.training.loss.per_image_loss_weight) == (
        0.25,
        0.75,
    )
    assert cfg.training.loss.binary_concept_names == ["bleached"]
    assert harness.build_args["cache_dir"] == Path("/the/cache")
    assert harness.build_args["image_ids"] == {("ds", "val"): ["i1"]}
    assert result["settings"] == {
        "data_config": "data.yaml",
        "bundle": "bundle/path",
        "resolution": 8,
        "max_images_per_dataset": 1000,
        "per_pixel_loss_weight": 0.25,
        "per_image_loss_weight": 0.75,
        "seed": 5,
        "concept_source_mask": None,
    }
    assert (tmp_path / "scored" / "metrics.json").is_file()


def test_val_loss_raises_when_the_data_config_has_no_val_split(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="data.yaml has no val split"):
        _Harness(monkeypatch, {}).run(tmp_path)
