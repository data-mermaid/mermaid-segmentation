"""Tests for frozen run-scoped concept channel layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mermaidseg.dataset_reconciliation import (
    ConceptSchema,
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.dataset_reconciliation.concepts import (
    DEFAULT_CLASS_TO_CONCEPTS_CSV,
    source_labels_to_concepts,
)
from tests._dataset_stubs import make_coralnet_stub
from tests.test_loss_components import _concept_channel_count


def _coralnet_maps(ds) -> dict[str, dict[str, str]]:
    return {
        "coralnet": {name: name for name in ds.source_name2id},
    }


def test_schema_channel_count_matches_concept_value2id() -> None:
    schema = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources={"coralnet"})
    assert schema.num_channels == len(schema.channel_names)
    assert schema.num_channels == _concept_channel_count(schema.concept_value2id)


def test_schema_stable_across_registry_composition(tmp_path: Path) -> None:
    schema = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources={"coralnet"})

    train = make_coralnet_stub(
        image_to_classes={"a": ["sand"]},
        image_to_source={"a": 1},
        source_id2name={1: "sand"},
    )
    train.split = "train"
    val = make_coralnet_stub(
        image_to_classes={"b": ["coral"]},
        image_to_source={"b": 2},
        source_id2name={1: "coral"},
    )
    val.split = "val"
    dataset_dict = {("coralnet", "train"): train, ("coralnet", "val"): val}
    _, registry_datasets = prepare_splits_for_registry(dataset_dict)

    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        source_to_target_name_maps=_coralnet_maps(registry_datasets[0]),
        compute_concepts=True,
        concept_schema=schema,
    )

    assert registry.num_concepts == schema.num_channels
    assert registry.concept_id2name == schema.channel_id2name()


def test_lookup_matches_schema_rows() -> None:
    schema = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources={"coralnet"})

    train = make_coralnet_stub(
        image_to_classes={"a": ["sand"], "b": ["coral"]},
        image_to_source={"a": 1, "b": 2},
        source_id2name={1: "sand", 2: "coral"},
    )
    train.split = "train"
    dataset_dict = {("coralnet", "train"): train}
    _, registry_datasets = prepare_splits_for_registry(dataset_dict)

    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        source_to_target_name_maps=_coralnet_maps(registry_datasets[0]),
        compute_concepts=True,
        concept_schema=schema,
    )

    for gid, (source, label) in registry.global_id2source.items():
        expected = schema.row_for(source, label)
        actual = registry.source_to_concepts[gid].cpu().numpy()
        assert np.allclose(expected, actual), f"gid={gid} {source}/{label}"


def test_schema_independent_of_registered_label_subset(tmp_path: Path) -> None:
    csv_path = tmp_path / "concepts.csv"
    pd.DataFrame(
        [
            {
                "source_label_class_name": "alpha",
                "source_dataset_source": "coralnet",
                "kingdom": "animalia",
                "phylum": "cnidaria",
                "class": "not_given",
                "order": "not_given",
                "family": "not_given",
                "genus": "not_given",
                "branching": "TRUE",
                "dead": "FALSE",
            },
            {
                "source_label_class_name": "beta",
                "source_dataset_source": "coralnet",
                "kingdom": "plantae",
                "phylum": "not_given",
                "class": "not_given",
                "order": "not_given",
                "family": "not_given",
                "genus": "not_given",
                "branching": "FALSE",
                "dead": "FALSE",
            },
        ]
    ).to_csv(csv_path, index=False)

    schema = ConceptSchema.from_csv(csv_path, sources={"coralnet"})
    assert schema.num_channels > 0

    train_only = make_coralnet_stub(
        image_to_classes={"a": ["alpha"]},
        image_to_source={"a": 1},
        source_id2name={1: "alpha"},
    )
    train_only.split = "train"
    _, reps_train = prepare_splits_for_registry({("coralnet", "train"): train_only})
    reg_train = SourceLabelRegistry(
        reps_train,
        fetch_remote=False,
        source_to_target_name_maps={"coralnet": {"alpha": "alpha"}},
        compute_concepts=True,
        concept_schema=schema,
    )

    full_dict = {
        ("coralnet", "train"): train_only,
        ("coralnet", "val"): make_coralnet_stub(
            image_to_classes={"b": ["beta"]},
            image_to_source={"b": 2},
            source_id2name={1: "beta"},
        ),
    }
    full_dict[("coralnet", "val")].split = "val"
    _, reps_full = prepare_splits_for_registry(full_dict)
    reg_full = SourceLabelRegistry(
        reps_full,
        fetch_remote=False,
        source_to_target_name_maps={"coralnet": {"alpha": "alpha", "beta": "beta"}},
        compute_concepts=True,
        concept_schema=schema,
    )

    assert reg_train.num_concepts == reg_full.num_concepts == schema.num_channels
    assert reg_train.concept_value2id == reg_full.concept_value2id == schema.concept_value2id


def test_val_concept_labels_match_schema_round_trip() -> None:
    schema = ConceptSchema.from_csv(DEFAULT_CLASS_TO_CONCEPTS_CSV, sources={"coralnet"})

    train = make_coralnet_stub(
        image_to_classes={"a": ["sand"]},
        image_to_source={"a": 1},
        source_id2name={1: "sand"},
    )
    train.split = "train"
    val = make_coralnet_stub(
        image_to_classes={"b": ["coral"]},
        image_to_source={"b": 2},
        source_id2name={1: "coral"},
    )
    val.split = "val"
    dataset_dict = {("coralnet", "train"): train, ("coralnet", "val"): val}
    _, registry_datasets = prepare_splits_for_registry(dataset_dict)

    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        source_to_target_name_maps=_coralnet_maps(registry_datasets[0]),
        compute_concepts=True,
        concept_schema=schema,
    )
    attach_registry(registry, dataset_dict.values())

    offset = registry.dataset_offsets["coralnet"]
    local_id = val.source_name2id["coral"]
    global_id = local_id + offset
    source_labels = torch.tensor([[[global_id]]], dtype=torch.long)
    concept_labels = source_labels_to_concepts(source_labels, registry.source_to_concepts)

    source, label = registry.global_id2source[global_id]
    expected = torch.from_numpy(schema.row_for(source, label)).view(1, -1, 1, 1).float()
    assert torch.allclose(concept_labels, expected)
