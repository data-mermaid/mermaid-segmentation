"""Per-binary-concept loss breakdown and the validation-loss helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mermaidseg.evaluation.val_loss import _Accumulator, _check_layout, binary_concept_names
from mermaidseg.model.concept_metrics import calculate_multi_hot_concept_loss
from mermaidseg.model.loss import ConceptBottleneckLoss, binary_concept_losses

NAMES = ["bleached", "anthropogenic", "sand"]
WEIGHTS = {"per_pixel_loss_weight": 0.5, "per_image_loss_weight": 0.5}


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, len(NAMES), 4, 4, generator=generator)
    labels = torch.randint(0, 3, (3, len(NAMES), 4, 4), generator=generator)
    labels[0, 0] = 0  # image 0 has no bleached labels
    labels[:, 2] = 0  # no sand labels anywhere
    return logits, labels


def test_each_channel_matches_the_multi_hot_loss_on_that_channel():
    logits, labels = _batch()
    losses = binary_concept_losses(logits, labels, NAMES, **WEIGHTS)
    assert set(losses) == {"multi_hot/bleached", "multi_hot/anthropogenic"}
    for channel, name in enumerate(NAMES[:2]):
        expected = calculate_multi_hot_concept_loss(
            logits[:, channel : channel + 1],
            labels[:, channel : channel + 1],
            from_logits=True,
            **WEIGHTS,
        )
        assert losses[f"multi_hot/{name}"] == pytest.approx(expected.item(), rel=1e-5)


def test_name_count_must_match_the_channels():
    logits, labels = _batch()
    with pytest.raises(ValueError, match="2 binary concept names for 3 channels"):
        binary_concept_losses(logits, labels, NAMES[:2], **WEIGHTS)


def test_concept_bottleneck_loss_reports_the_breakdown_only_when_named():
    logits, labels = _batch()
    outputs = torch.randn(3, 5, 4, 4)
    targets = torch.randint(0, 5, (3, 4, 4))
    plain = ConceptBottleneckLoss({}, **WEIGHTS)
    named = ConceptBottleneckLoss({}, binary_concept_names=NAMES, **WEIGHTS)
    total_plain, plain_components = plain(outputs, targets, logits, labels)
    total_named, named_components = named(outputs, targets, logits, labels)

    assert not any(key.startswith("multi_hot/") for key in plain_components)
    assert {"multi_hot/bleached", "multi_hot/anthropogenic"} <= named_components.keys()
    assert "multi_hot/sand" not in named_components
    assert total_named.item() == pytest.approx(total_plain.item())
    assert named_components["multi_hot"] == pytest.approx(plain_components["multi_hot"])


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
