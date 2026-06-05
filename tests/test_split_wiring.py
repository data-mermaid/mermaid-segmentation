"""Tests for split vocabulary canonicalization and registry attachment."""

from __future__ import annotations

import numpy as np

from mermaidseg.dataset_reconciliation import (
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from tests._dataset_stubs import make_coralnet_stub


def _coralnet_maps(ds) -> dict[str, dict[str, str]]:
    return {
        "coralnet": {name: name for name in ds.source_name2id},
    }


def test_union_vocab_aligns_train_and_val_local_ids() -> None:
    train = make_coralnet_stub(
        image_to_classes={"a": ["sand"], "b": ["coral"]},
        image_to_source={"a": 1, "b": 2},
        source_id2name={1: "sand", 2: "coral"},
    )
    train.split = "train"
    val = make_coralnet_stub(
        image_to_classes={"c": ["coral"], "d": ["rubble"]},
        image_to_source={"c": 3, "d": 4},
        source_id2name={1: "coral", 2: "rubble"},
    )
    val.split = "val"
    dataset_dict = {("coralnet", "train"): train, ("coralnet", "val"): val}

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        source_to_target_name_maps=_coralnet_maps(registry_datasets[0]),
        compute_concepts=False,
    )
    attach_registry(registry, dataset_dict.values())

    assert train.source_name2id["coral"] == val.source_name2id["coral"]
    assert train.source_name2id["sand"] == val.source_name2id["sand"]
    assert train.source_name2id["rubble"] == val.source_name2id["rubble"]
    assert train.global_offset == val.global_offset == registry.dataset_offsets["coralnet"]


def test_val_global_id_maps_to_same_registry_row_as_train() -> None:
    train = make_coralnet_stub(
        image_to_classes={"a": ["sand"], "b": ["coral"]},
        image_to_source={"a": 1, "b": 2},
        source_id2name={1: "sand", 2: "coral"},
    )
    train.split = "train"
    val = make_coralnet_stub(
        image_to_classes={"c": ["coral"]},
        image_to_source={"c": 3},
        source_id2name={1: "coral"},
    )
    val.split = "val"
    dataset_dict = {("coralnet", "train"): train, ("coralnet", "val"): val}

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        source_to_target_name_maps=_coralnet_maps(registry_datasets[0]),
        compute_concepts=False,
    )
    attach_registry(registry, dataset_dict.values())

    offset = registry.dataset_offsets["coralnet"]
    train_gid = train.source_name2id["coral"] + offset
    val_gid = val.source_name2id["coral"] + offset
    assert train_gid == val_gid
    assert registry.source_to_target[train_gid] == registry.source_to_target[val_gid]
