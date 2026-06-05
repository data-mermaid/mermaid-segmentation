"""Tests for per-component loss decomposition."""

from __future__ import annotations

import pytest
import torch

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.loss import BCEWithLogitsLoss, ConceptBottleneckLoss, CrossEntropyLoss


def _tiny_concept_value2id() -> dict[str, dict[str, list[int]]]:
    mapping = {rank: {"a": [2, 1], "b": [1, 2]} for rank in TAXONOMIC_CONCEPTS}
    mapping["branching"] = {"true": 1, "false": 0}
    mapping["encrusting"] = {"true": 1, "false": 0}
    return mapping


def _concept_channel_count(concept_value2id: dict[str, dict[str, list[int]]]) -> int:
    total = 0
    for rank in TAXONOMIC_CONCEPTS:
        total += len(next(iter(concept_value2id[rank].values())))
    total += len([key for key in concept_value2id if key not in TAXONOMIC_CONCEPTS])
    return total


def _realistic_concept_labels(
    concept_value2id: dict[str, dict[str, list[int]]],
    batch: int,
    height: int,
    width: int,
) -> torch.Tensor:
    num_concepts = _concept_channel_count(concept_value2id)
    labels = torch.zeros(batch, num_concepts, height, width)
    offset = 0
    for rank in TAXONOMIC_CONCEPTS:
        rank_width = len(next(iter(concept_value2id[rank].values())))
        encoding = torch.tensor(concept_value2id[rank]["a"], dtype=torch.float32)
        labels[:, offset : offset + rank_width, :, :] = encoding.view(1, rank_width, 1, 1)
        offset += rank_width
    labels[:, offset:, :, :] = 2  # multi-hot true
    return labels


def test_concept_bottleneck_loss_returns_all_components() -> None:
    concept_value2id = _tiny_concept_value2id()
    num_concepts = _concept_channel_count(concept_value2id)
    loss_fn = ConceptBottleneckLoss(concept_value2id=concept_value2id)

    outputs = torch.randn(2, 5, 4, 4)
    target_labels = torch.randint(1, 5, (2, 4, 4))
    concept_outputs = torch.rand(2, num_concepts, 4, 4)
    offset = 0
    for rank in TAXONOMIC_CONCEPTS:
        rank_width = len(next(iter(concept_value2id[rank].values())))
        chunk = concept_outputs[:, offset : offset + rank_width, ...]
        concept_outputs[:, offset : offset + rank_width, ...] = torch.softmax(chunk, dim=1)
        offset += rank_width
    concept_outputs[:, offset:, ...] = torch.sigmoid(concept_outputs[:, offset:, ...])
    concept_labels = _realistic_concept_labels(concept_value2id, batch=2, height=4, width=4)

    total_loss, components = loss_fn(outputs, target_labels, concept_outputs, concept_labels)

    assert isinstance(total_loss, torch.Tensor)
    expected_keys = {"classification", "concepts", "multi_hot", *TAXONOMIC_CONCEPTS}
    assert expected_keys.issubset(set(components))
    assert components["concepts"] == pytest.approx(
        sum(components[rank] for rank in TAXONOMIC_CONCEPTS) + components["multi_hot"]
    )
    for rank in TAXONOMIC_CONCEPTS:
        assert components[rank] > 0.0


def test_bce_with_logits_loss_returns_rank_components() -> None:
    concept_value2id = _tiny_concept_value2id()
    num_concepts = _concept_channel_count(concept_value2id)
    loss_fn = BCEWithLogitsLoss(concept_value2id=concept_value2id)

    concept_outputs = torch.randn(2, num_concepts, 4, 4)
    concept_labels = _realistic_concept_labels(concept_value2id, batch=2, height=4, width=4)
    target_labels = torch.ones(2, 4, 4, dtype=torch.long)

    total_loss, components = loss_fn(concept_outputs, concept_labels, target_labels)

    assert isinstance(total_loss, torch.Tensor)
    assert "concepts" in components
    assert "multi_hot" in components
    for rank in TAXONOMIC_CONCEPTS:
        assert rank in components
        assert components[rank] > 0.0


def test_cross_entropy_loss_returns_classification_component() -> None:
    loss_fn = CrossEntropyLoss(ignore_index=0)
    outputs = torch.randn(2, 5, 4, 4)
    target_labels = torch.randint(0, 5, (2, 4, 4))

    total_loss, components = loss_fn(outputs, target_labels)

    assert isinstance(total_loss, torch.Tensor)
    assert components == {"classification": pytest.approx(total_loss.item())}
