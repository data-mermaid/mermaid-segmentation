"""Tests for split vocabulary canonicalization and registry attachment."""

from __future__ import annotations

import numpy as np
import torch

from mermaidseg.dataset_reconciliation import (
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.datasets.benthos_yuval.benthos_yuval_corals_dataset import (
    BenthosYuvalCoralsDataset,
)
from mermaidseg.datasets.coralscapes.coralscapes_dataset import (
    CORALSCAPES_ID2NAME,
    CoralscapesDataset,
)
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


def _make_coralscapes_stub(*, source_id2name: dict[int, str]) -> CoralscapesDataset:
    """CoralscapesDataset shell without loading the HuggingFace corpus."""
    ds = CoralscapesDataset.__new__(CoralscapesDataset)
    ds.split = "train"
    ds.transform = None
    ds.class_subset = None
    ds._global_offset = 0
    ds.source_id2name = dict(source_id2name)
    ds.source_name2id = {name: local_id for local_id, name in source_id2name.items()}
    ds.num_source_classes = len(source_id2name) + 1
    ds._native_to_local = ds._build_native_to_local()
    return ds


def test_coralscapes_native_lookup_matches_sorted_registry_vocab() -> None:
    """Emitted local ids must match registry rows after vocabulary canonicalization."""
    native_order = {i: CORALSCAPES_ID2NAME[i] for i in sorted(CORALSCAPES_ID2NAME)}
    train = _make_coralscapes_stub(source_id2name=native_order)
    val = _make_coralscapes_stub(source_id2name=native_order)
    dataset_dict = {("coralscapes", "train"): train, ("coralscapes", "val"): val}

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    registry = SourceLabelRegistry(
        registry_datasets,
        fetch_remote=False,
        compute_concepts=True,
        concept_schema=ConceptSchema.from_csv(sources={"coralscapes"}),
    )
    attach_registry(registry, dataset_dict.values())

    human_native_id = next(
        native_id for native_id, name in CORALSCAPES_ID2NAME.items() if name == "human"
    )
    background_native_id = next(
        native_id for native_id, name in CORALSCAPES_ID2NAME.items() if name == "background"
    )

    human_local = int(train._native_to_local[human_native_id])
    background_local = int(train._native_to_local[background_native_id])
    assert train.source_id2name[human_local] == "human"
    assert train.source_id2name[background_local] == "background"

    offset = registry.dataset_offsets["coralscapes"]
    human_gid = human_local + offset
    background_gid = background_local + offset

    schema = ConceptSchema.from_csv(sources={"coralscapes"})
    channel_names = list(schema.channel_names)
    human_row = schema.row_for("coralscapes", "human")
    background_row = schema.row_for("coralscapes", "background")

    assert int(registry.source_to_concepts[human_gid, channel_names.index("human")]) == 2
    assert int(registry.source_to_concepts[human_gid, channel_names.index("anthropogenic")]) == 2
    assert int(registry.source_to_concepts[background_gid, channel_names.index("background")]) == 2
    assert int(registry.source_to_concepts[background_gid, channel_names.index("human")]) == 1

    human_mask = np.full((1, 4, 4), human_gid, dtype=np.int64)
    background_mask = np.full((1, 4, 4), background_gid, dtype=np.int64)
    human_concepts = source_labels_to_concepts(
        torch.from_numpy(human_mask),
        registry.source_to_concepts,
    )
    background_concepts = source_labels_to_concepts(
        torch.from_numpy(background_mask),
        registry.source_to_concepts,
    )

    assert int(human_concepts[0, channel_names.index("human"), 0, 0]) == 2
    assert int(human_concepts[0, channel_names.index("anthropogenic"), 0, 0]) == 2
    assert int(background_concepts[0, channel_names.index("background"), 0, 0]) == 2
    assert int(background_concepts[0, channel_names.index("human"), 0, 0]) == 1
    np.testing.assert_array_equal(
        registry.source_to_concepts[human_gid].numpy(),
        human_row,
    )
    np.testing.assert_array_equal(
        registry.source_to_concepts[background_gid].numpy(),
        background_row,
    )


def test_benthos_yuval_global_lookup_rebuilds_on_vocab_change() -> None:
    """classes.json lookup must track registry vocabulary reassignment."""
    ds = BenthosYuvalCoralsDataset.__new__(BenthosYuvalCoralsDataset)
    ds._classes_global = {"sand": 1, "coral": 2}
    ds.source_id2name = {1: "sand", 2: "coral"}
    ds.source_name2id = {"sand": 1, "coral": 2}
    ds.num_source_classes = 3
    ds._classes_global_to_local = ds._build_classes_global_to_local()

    ds.set_source_vocabulary(
        {1: "coral", 2: "sand"},
        {"coral": 1, "sand": 2},
        3,
    )

    assert int(ds._classes_global_to_local[1]) == ds.source_name2id["sand"]
    assert int(ds._classes_global_to_local[2]) == ds.source_name2id["coral"]
