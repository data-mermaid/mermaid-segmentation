"""Checks for the per-shard label lookup and loss, the batch split guard, and the
training-metric cadence."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.registry import source_labels_to_target_labels
from mermaidseg.io import ConfigDict
from mermaidseg.model.data_parallel import (
    ShardStep,
    apply_concept_bottleneck_loss,
    require_divisible_batch,
    shard_targets,
)
from tests.conftest import CBM_CONCEPT_VALUE2ID, IMAGE_SIZE, NUM_CLASSES
from tests.mermaidseg.conftest import cbm_meta, concept_keep, source_lookup

_NUM_SOURCE = 9
_NUM_CONCEPTS = 5
_NUM_CLASSES = 4


def _lookups(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randint(0, _NUM_CLASSES, (_NUM_SOURCE + 1,), generator=generator)
    target[0] = 0
    concepts = torch.rand(_NUM_SOURCE + 1, _NUM_CONCEPTS, generator=generator)
    concepts[0] = 0
    return target, concepts


class _TinyCBM(nn.Module):
    """Stand-in for the concept-bottleneck model: a frozen stem and two trainable heads."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 1)
        self.concept_head = nn.Conv2d(6, _NUM_CONCEPTS, 1)
        self.classifier = nn.Conv2d(_NUM_CONCEPTS, _NUM_CLASSES, 1)
        self.register_buffer("scale", torch.ones(1))
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        concept_logits = self.concept_head(torch.relu(self.stem(x))) * self.scale
        concept_outputs = torch.sigmoid(concept_logits)
        return SimpleNamespace(
            logits=self.classifier(concept_outputs),
            concept_logits=concept_logits,
            hidden_states=concept_outputs,
        )


class _TinyLoss(nn.Module):
    def forward(self, outputs, target_labels, concept_logits, target_concepts):
        class_loss = nn.functional.cross_entropy(outputs, target_labels)
        concept_loss = nn.functional.binary_cross_entropy_with_logits(
            concept_logits, target_concepts
        )
        loss = class_loss + concept_loss
        return loss, {"concepts": float(concept_loss.detach())}


def _cpu_step() -> ShardStep:
    target, concepts = _lookups()
    return ShardStep(_TinyCBM(), _TinyLoss(), None, target, concepts, False, torch.bfloat16)


def _batch(batch_size: int = 12, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch_size, 3, 8, 8, generator=generator)
    source_labels = torch.randint(0, _NUM_SOURCE + 1, (batch_size, 8, 8), generator=generator)
    return images, source_labels


def test_batch_must_split_evenly_across_the_given_gpus():
    assert require_divisible_batch(16, 4) == 4
    assert require_divisible_batch(8, 4) == 2
    with pytest.raises(RuntimeError, match="not divisible"):
        require_divisible_batch(15, 4)


def test_shard_targets_match_the_whole_batch_lookup():
    target, concepts = _lookups()
    _, source_labels = _batch()
    whole_labels = source_labels_to_target_labels(source_labels, target)
    whole_concepts = source_labels_to_concepts(source_labels, concepts)
    for start, end in [(0, 3), (3, 6), (6, 9), (9, 12)]:
        labels, shard_concepts = shard_targets(source_labels[start:end], target, concepts)
        assert torch.equal(labels, whole_labels[start:end])
        assert torch.equal(shard_concepts, whole_concepts[start:end])


def test_shard_step_matches_the_single_gpu_loss():
    step = _cpu_step()
    images, source_labels = _batch()
    loss, outputs, concept_outputs, labels, concepts, _ = step(images, source_labels)

    target, concept_lookup = _lookups()
    expected_labels = source_labels_to_target_labels(source_labels, target)
    expected_concepts = source_labels_to_concepts(source_labels, concept_lookup)
    expected_loss, expected_outputs, expected_concept_outputs, _ = apply_concept_bottleneck_loss(
        step.model(images),
        step.loss_fn,
        None,
        expected_labels,
        expected_concepts,
    )
    assert torch.equal(loss, expected_loss)
    assert torch.equal(outputs, expected_outputs)
    assert torch.equal(concept_outputs, expected_concept_outputs)
    assert torch.equal(labels, expected_labels)
    assert torch.equal(concepts, expected_concepts)

    no_maps = step(images, source_labels, return_maps=False)
    assert torch.equal(no_maps[0], expected_loss)
    assert no_maps[1:5] == (None, None, None, None)


def test_train_epoch_scores_every_nth_microbatch(
    minimal_config, make_meta_model, tiny_loader, monkeypatch
):
    from mermaidseg.model.eval import Evaluator

    cfg = copy.deepcopy(minimal_config)
    cfg.training = ConfigDict(
        {
            **dict(cfg.training),
            "iterations_per_train_epoch": 3,
            "grad_accum_steps": 2,
            "train_metrics_every": 4,
        }
    )
    meta = make_meta_model(cfg)
    evaluator = Evaluator(num_classes=4, device="cpu", concept_value2id=CBM_CONCEPT_VALUE2ID)
    metric = evaluator.metric_dict["accuracy"]
    calls = []
    original_update = metric.update

    def _counting_update(*args, **kwargs):
        calls.append(1)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(metric, "update", _counting_update)
    meta.train_epoch(tiny_loader, evaluator)
    # Six microbatches, indices 0..5; every 4th is scored.
    assert len(calls) == 2
    assert meta._train_microbatches_seen == 6


@pytest.mark.parametrize("use_amp", [True, False])
def test_shard_step_autocasts_on_the_input_device_type(use_amp):
    target, concepts = _lookups()
    model = _TinyCBM()
    seen: dict[str, object] = {}

    def _record(_module, _args):
        seen["enabled"] = torch.is_autocast_enabled("cpu")
        seen["dtype"] = torch.get_autocast_dtype("cpu")

    model.stem.register_forward_pre_hook(_record)
    step = ShardStep(model, _TinyLoss(), None, target, concepts, use_amp, torch.bfloat16)
    images, source_labels = _batch()
    step(images, source_labels)
    assert seen["enabled"] is use_amp
    if use_amp:
        assert seen["dtype"] == torch.bfloat16


def test_shard_step_masks_its_shard(make_meta_model, cbm_schema):
    """The data_parallel / distributed per-GPU step masks with the rows it is given."""
    meta = cbm_meta(make_meta_model, cbm_schema)
    keep = concept_keep(cbm_schema)
    lookup = source_lookup(cbm_schema)
    bleached = cbm_schema.channel_names.index("bleached")
    target = torch.arange(NUM_CLASSES)
    step = ShardStep(meta.model, meta.loss, None, target, lookup, False, torch.bfloat16, keep)
    images = torch.randn(2, 3, *IMAGE_SIZE)
    labels = torch.ones(2, *IMAGE_SIZE, dtype=torch.long)
    *_, concepts, _ = step(images, labels, return_maps=True, source_rows=torch.tensor([1, 0]))
    assert (concepts[0, bleached] == 0).all() and (concepts[1, bleached] == 1).all()
    with pytest.raises(RuntimeError, match="no source rows"):
        shard_targets(labels, target, lookup, keep, None)
