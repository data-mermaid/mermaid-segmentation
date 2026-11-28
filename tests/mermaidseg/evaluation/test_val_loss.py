"""The validation-loss helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mermaidseg.data.config import DataSection, SplitSelection
from mermaidseg.evaluation import val_loss
from mermaidseg.evaluation.val_loss import (
    _Accumulator,
    _bundle_config,
    evaluate_val_loss,
    split_blocks_only,
)
from mermaidseg.io import ConfigDict
from mermaidseg.model.loss import build_loss
from mermaidseg.model.models import ConceptBottleneckOutput
from tests.conftest import IMAGE_SIZE, NUM_CLASSES
from tests.mermaidseg.conftest import cbm_mask_config, cbm_meta, concept_keep, source_lookup

NAMES = ["bleached", "anthropogenic", "sand"]
_VAL_SECTION = DataSection.model_validate(
    {
        "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "c" * 64},
        "datasets": {"ds": {"version": "v", "manifest_sha256": "x", "val": {"groups": "all"}}},
    }
)


def test_accumulator_averages_each_key_over_the_batches_it_appears_in():
    first, second = _Accumulator(), _Accumulator()
    first.add({"loss": 1.0, "multi_hot/bleached": 0.2})
    first.add({"loss": 3.0})
    second.add({"loss": 5.0, "multi_hot/bleached": 0.4})
    first.merge(second)
    assert first.means() == pytest.approx({"loss": 3.0, "multi_hot/bleached": 0.3})


def test_bundle_config_takes_the_model_and_training_sections_from_the_bundle():
    run = {
        "data": {"datasets": {}},
        "model": {"name": "dpt", "input_size": [8, 8]},
        "training": {"batch_size": 2, "taxonomic_loss": {"kl_weight": 1.0}, "loss": {"a": 1}},
    }
    bundle = SimpleNamespace(config=run, input_size=(32, 64))
    cfg = _bundle_config(bundle)
    assert sorted(cfg) == ["model", "training"]
    assert cfg.model.name == "dpt"
    assert cfg.model.input_size == [32, 64]
    assert cfg.training.batch_size == 2
    cfg.training.pop("taxonomic_loss")
    cfg.training.loss.a = 9
    assert "taxonomic_loss" in run["training"]
    assert run["training"]["loss"] == {"a": 1}
    assert run["model"]["input_size"] == [8, 8]


def _split_section() -> DataSection:
    """``a`` has all three blocks; ``b`` only a train block, with transform and repeat."""
    return DataSection.model_validate(
        {
            "taxonomy": {"version": "20261121-ab12cd3", "manifest_sha256": "c" * 64},
            "datasets": {
                "a": {
                    "version": "v",
                    "manifest_sha256": "x",
                    "train": {"groups": "all"},
                    "val": {"groups": ["g1"], "annotators": ["host"]},
                    "test": {"groups": ["g2"]},
                },
                "b": {
                    "version": "v",
                    "manifest_sha256": "x",
                    "train": {
                        "groups": ["g3"],
                        "annotators": ["archived"],
                        "transform": {"HorizontalFlip": {}},
                        "repeat": 2,
                    },
                },
                "c": {"version": "v", "manifest_sha256": "x", "test": {"groups": ["g4"]}},
            },
        }
    )


def test_split_blocks_only_keeps_the_val_block_of_each_dataset_that_has_one():
    kept = split_blocks_only(_split_section(), "val")

    assert list(kept.datasets) == ["a"]
    entry = kept.datasets["a"]
    assert (entry.train, entry.test) == (None, None)
    assert entry.val == SplitSelection(groups=["g1"], annotators=["host"])


def test_split_blocks_only_moves_the_train_block_into_val_without_transform_or_repeat():
    kept = split_blocks_only(_split_section(), "train")

    assert list(kept.datasets) == ["a", "b"]
    entry = kept.datasets["b"]
    assert (entry.train, entry.test) == (None, None)
    assert entry.val == SplitSelection(groups=["g3"], annotators=["archived"])
    assert kept.datasets["a"].val == SplitSelection(groups="all")


def test_val_loss_raises_when_no_dataset_has_the_split(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, {("ds", "val"): _images(1.0)})
    with pytest.raises(ValueError, match="data section has no train split"):
        harness.run(
            tmp_path, section=_split_section().model_copy(update={"datasets": {}}), split="train"
        )


class _EchoModel(nn.Module):
    """Bundle model whose class logits, concept logits and concept scores are the image itself."""

    def forward(self, images: torch.Tensor) -> ConceptBottleneckOutput:
        return ConceptBottleneckOutput(logits=images, concept_logits=images, concept_probs=images)


class _MeanLoss(nn.Module):
    """Loss = mean output, classification = twice that; ``record`` sees each batch's arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.record: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, outputs, target_labels, _concept_logits, target_concepts):
        self.record.append((outputs, target_labels, target_concepts))
        value = outputs.mean()
        return value, {"classification": value * 2}


class _Harness:
    """Stubs of the collaborators that read configs and datasets; the loop, step and outputs are real."""

    def __init__(self, monkeypatch, datasets: dict):
        self.cfg = ConfigDict(
            {
                "model": {"input_size": [8, 8]},
                "training": {"padding": 0, "taxonomic_loss": {"w": 1}, "loss": {}},
            }
        )
        self.datasets = datasets
        self.build_args: dict = {}
        self.loss = _MeanLoss()
        self.loss_args: tuple = ()
        # Source ids 0, 1, 2 -> target classes 0, 1, 1 and concept values 0, 1, 0.5.
        registry = SimpleNamespace(
            target_id2label={1: "sand"},
            concept_value2id={"bleached": {"not_given": 0}},
            source_to_target=torch.tensor([0, 1, 1]),
            source_to_concepts=torch.tensor([[0.0], [1.0], [0.5]]),
        )
        schema = SimpleNamespace(
            channel_id2name=lambda: {1: "bleached"}, binary_names=lambda: ["bleached"]
        )

        def build_loss(loss_cfg, concept_value2id):
            self.loss_args = (loss_cfg, concept_value2id)
            return self.loss

        monkeypatch.setattr(val_loss, "_bundle_config", lambda *_a: self.cfg)

        def build_datasets(section, cache_dir, **kwargs):
            self.build_args = {"cache_dir": cache_dir, **kwargs}
            return self.datasets

        monkeypatch.setattr(val_loss, "build_datasets", build_datasets)
        self.registry_kwargs: dict = {}

        def build_registry(*_a, **kwargs):
            self.registry_kwargs = kwargs
            return registry, schema

        monkeypatch.setattr(val_loss, "build_registry", build_registry)
        monkeypatch.setattr(val_loss, "build_loss", build_loss)
        self.bundle = SimpleNamespace(
            id2label={0: "ignore", 1: "sand"},
            concept_id2name={1: "bleached"},
            input_size=(8, 8),
            path="bundle/path",
            model=_EchoModel(),
            metadata={"label_space": {}},
        )

    def run(self, tmp_path, *, label_space: bool = True, **kwargs):
        self.bundle.path = tmp_path / "bundle"
        if label_space:
            (self.bundle.path / "label_space").mkdir(parents=True, exist_ok=True)
        return evaluate_val_loss(
            bundle=self.bundle,  # type: ignore[arg-type]
            section=kwargs.pop("section", _VAL_SECTION),
            split=kwargs.pop("split", "val"),
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
    harness = _Harness(monkeypatch, {("ds", "val"): _images(*map(float, range(8)))})
    record = harness.loss.record

    def seen() -> list[float]:
        return [float(outputs.mean()) for outputs, _, _ in record]

    result = harness.run(tmp_path, max_images_per_dataset=4, seed=3)
    assert result["n_images"] == {"ds": 4}
    first = seen()
    assert first == sorted(first)
    assert len(set(first)) == 4
    record.clear()
    harness.run(tmp_path, max_images_per_dataset=4, seed=3)
    assert seen() == first
    subsets = set()
    for other in range(5):
        record.clear()
        harness.run(tmp_path, max_images_per_dataset=4, seed=other)
        subsets.add(tuple(seen()))
    assert len(subsets) > 1


def test_val_loss_scores_the_targets_the_registry_lookups_give(tmp_path, monkeypatch):
    labels = torch.tensor([[[1.0, 2.0], [2.0, 2.0]]])
    harness = _Harness(
        monkeypatch, {("ds", "val"): TensorDataset(torch.full((1, 3, 2, 2), 1.0), labels)}
    )

    harness.run(tmp_path)

    # Source ids 1, 2, 2, 2: target classes 1, 1, 1, 1 and concept values 1, 0.5, 0.5, 0.5.
    ((_, target_labels, target_concepts),) = harness.loss.record
    assert target_labels.tolist() == [[[1, 1], [1, 1]]]
    assert target_concepts.tolist() == [[[[1.0, 0.5], [0.5, 0.5]]]]


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
    cfg = harness.cfg
    assert cfg.run_name == "scored"
    assert cfg.training.distributed is False
    assert (cfg.training.loss.per_pixel_loss_weight, cfg.training.loss.per_image_loss_weight) == (
        0.25,
        0.75,
    )
    assert cfg.training.loss.binary_concept_names == ["bleached"]
    assert harness.loss_args == (cfg.training.loss, {"bleached": {"not_given": 0}})
    assert harness.build_args["cache_dir"] == Path("/the/cache")
    assert harness.build_args["image_ids"] == {("ds", "val"): ["i1"]}
    assert result["settings"] == {
        "split": "val",
        "bundle": str(tmp_path / "bundle"),
        "resolution": 8,
        "max_images_per_dataset": 1000,
        "per_pixel_loss_weight": 0.25,
        "per_image_loss_weight": 0.75,
        "seed": 5,
        "concept_source_mask": None,
    }
    assert (tmp_path / "scored" / "metrics.json").is_file()


def test_val_loss_reads_the_label_space_of_the_bundle_not_the_repo_path_of_its_config(
    tmp_path, monkeypatch
):
    harness = _Harness(monkeypatch, {("ds", "val"): _images(1.0)})
    harness.cfg.training.label_space = "configs/label_spaces/gone"

    harness.run(tmp_path)

    assert harness.registry_kwargs["label_space_dir"] == tmp_path / "bundle" / "label_space"


@pytest.mark.parametrize(
    ("listed", "make_dir"),
    [("classes.csv", True), ("../outside.csv", True), ("classes.csv", False)],
    ids=["absent-file", "outside-the-directory", "absent-directory"],
)
def test_val_loss_refuses_a_listed_label_space_file_that_is_not_in_the_bundle(
    tmp_path, monkeypatch, listed, make_dir
):
    harness = _Harness(monkeypatch, {("ds", "val"): _images(1.0)})
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "outside.csv").write_text("x\n")
    harness.bundle.metadata = {"label_space": {listed: hashlib.sha256(b"x\n").hexdigest()}}

    with pytest.raises(ValueError, match="is missing"):
        harness.run(tmp_path, label_space=make_dir)


def test_val_loss_refuses_a_bundle_whose_label_space_was_edited_after_export(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, {("ds", "val"): _images(1.0)})
    space = tmp_path / "bundle" / "label_space"
    space.mkdir(parents=True)
    (space / "classes.csv").write_text("exported\n")
    harness.bundle.metadata = {
        "label_space": {"classes.csv": hashlib.sha256(b"exported\n").hexdigest()}
    }
    (space / "classes.csv").write_text("edited\n")

    with pytest.raises(ValueError, match="differs from the sha256"):
        harness.run(tmp_path)


def _cbm_scoring_config(schema) -> ConfigDict:
    """The CBM run config as ``evaluate_val_loss`` scores it: 0.5/0.5 reduction, named binaries."""
    cfg = cbm_mask_config()
    cfg.training.loss.binary_concept_names = schema.binary_names()
    cfg.training.padding = 0
    return cfg


def _run_real_val_loss(monkeypatch, tmp_path, schema, model, dataset, **kwargs):
    """``evaluate_val_loss`` on one val dataset with the real loss, step and lookups.

    Returns the result and the concept targets the loss saw, one entry per batch.
    """
    cfg = _cbm_scoring_config(schema)
    registry = SimpleNamespace(
        target_id2label={1: "acropora", 2: "porites"},
        concept_value2id=schema.concept_value2id,
        source_to_target=torch.arange(NUM_CLASSES),
        source_to_concepts=source_lookup(schema),
    )
    seen: list[torch.Tensor] = []

    def recording_build_loss(loss_cfg, concept_value2id):
        loss = build_loss(loss_cfg, concept_value2id)
        forward = loss.forward

        def record(outputs, target_labels, concept_logits, target_concepts):
            seen.append(target_concepts)
            return forward(outputs, target_labels, concept_logits, target_concepts)

        loss.forward = record  # type: ignore[method-assign]
        return loss

    monkeypatch.setattr(val_loss, "_bundle_config", lambda *_a: cfg)
    monkeypatch.setattr(val_loss, "build_datasets", lambda *_a, **_k: {("ds", "val"): dataset})
    monkeypatch.setattr(val_loss, "build_registry", lambda *_a, **_k: (registry, schema))
    monkeypatch.setattr(val_loss, "build_loss", recording_build_loss)
    bundle = SimpleNamespace(
        id2label={0: "ignore", **registry.target_id2label},
        concept_id2name=schema.channel_id2name(),
        input_size=tuple(IMAGE_SIZE),
        path=tmp_path / "bundle",
        model=model,
        metadata={"label_space": {}},
    )
    (tmp_path / "bundle" / "label_space").mkdir(parents=True)
    result = evaluate_val_loss(
        bundle=bundle,  # type: ignore[arg-type]
        section=_VAL_SECTION,
        split="val",
        cache_dir="/cache",
        output_dir=tmp_path,
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        **kwargs,
    )
    return result, seen


def test_val_loss_matches_the_meta_model_validation_loss_on_the_same_batch(
    tmp_path, monkeypatch, make_meta_model, cbm_schema
):
    # Same 0.5/0.5 reduction as the scoring defaults; training sets no binary_concept_names.
    meta = cbm_meta(make_meta_model, cbm_schema)
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(2, 3, *IMAGE_SIZE, generator=generator)
    labels = torch.randint(0, 3, (2, *IMAGE_SIZE), generator=generator)
    dataset = TensorDataset(images, labels)
    expected_loss, expected = meta.validation_epoch(DataLoader(dataset, batch_size=2))

    result, _ = _run_real_val_loss(monkeypatch, tmp_path, cbm_schema, meta.model, dataset)

    overall = dict(result["overall"])
    assert overall.pop("loss") == pytest.approx(expected_loss, rel=1e-6)
    shared = {key.removeprefix("loss/"): value for key, value in expected.items()}
    assert {key: overall[key] for key in shared} == pytest.approx(shared, rel=1e-6)
    # Scoring adds one multi_hot/<name> term per binary channel on top of them.
    assert {key for key in overall if key not in shared} == {
        f"multi_hot/{name}" for name in cbm_schema.binary_names()
    }


def test_val_loss_with_the_source_mask_zeroes_the_masked_concept_targets(
    tmp_path, monkeypatch, make_meta_model, cbm_schema
):
    images = torch.randn(2, 3, *IMAGE_SIZE)
    labels = torch.ones(2, *IMAGE_SIZE, dtype=torch.long)  # acropora: bleached False
    rows = torch.tensor([1, 0])
    mask = SimpleNamespace(
        keep=concept_keep(cbm_schema), path="mask.csv", num_masked=1, dataset="coralnet"
    )
    mask_kwargs: dict = {}

    def load_mask(*_args, **kwargs):
        mask_kwargs.update(kwargs)
        return mask

    monkeypatch.setattr(val_loss, "load_run_concept_source_mask", load_mask)
    monkeypatch.setattr(
        val_loss,
        "attach_source_rows",
        lambda datasets, _mask: [TensorDataset(images, labels, rows) for _ in datasets],
    )
    model = cbm_meta(make_meta_model, cbm_schema).model

    result, seen = _run_real_val_loss(
        monkeypatch,
        tmp_path,
        cbm_schema,
        model,
        TensorDataset(images, labels),
        apply_source_mask=True,
    )

    # Keep-table row 1 masks bleached for sample 0; sample 1 (row 0) keeps its False target 1.
    bleached = cbm_schema.channel_names.index("bleached")
    (targets,) = seen
    assert (targets[0, bleached] == 0).all()
    assert (targets[1, bleached] == 1).all()
    assert result["settings"]["concept_source_mask"] == "mask.csv"
    assert mask_kwargs["label_space_dir"] == tmp_path / "bundle" / "label_space"
