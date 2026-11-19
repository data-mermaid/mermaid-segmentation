"""Tests for label-integrity fixes in registry vocabulary wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry
from mermaidseg.dataset_reconciliation.split_wiring import (
    _root_dataset,
    attach_registry,
    group_splits,
    prepare_splits_for_registry,
    select_registry_representatives,
)
from mermaidseg.datasets.benthos_yuval.benthos_yuval_corals_dataset import (
    BenthosYuvalCoralsDataset,
)
from mermaidseg.datasets.coralscapes_v2.coralscapes_v2_dataset import (
    CORALSCAPES_V2_ID2NAME,
    CoralscapesV2Dataset,
)
from mermaidseg.datasets.ucsd_mosaics.ucsd_mosaics_dataset import UCSDMosaicsDataset


class _SourceStub(Dataset):
    """Dataset stub exposing only what split wiring and the registry read."""

    def __init__(self, source_name: str, names: list[str]) -> None:
        self.SOURCE_NAME = source_name
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {n: i for i, n in self.source_id2name.items()}
        self.num_source_classes = len(names) + 1
        self.offset: int | None = None
        self.dataset: object | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def _coralscapes_v2(monkeypatch, class_subset: list[str]) -> CoralscapesV2Dataset:
    """The real dataset class on an empty HuggingFace corpus."""
    monkeypatch.setattr("datasets.load_dataset", lambda repo_id: {"train": []})
    return CoralscapesV2Dataset(split="train", class_subset=class_subset)


# Native Coralscapes V2 ids of the names used below.
_ACROPORA_ALIVE, _FISH, _SAND, _ACANTHASTER = 2, 25, 69, 1


def test_coralscapes_v2_lookup_follows_sorted_registry_vocab_and_offset(monkeypatch) -> None:
    """Emitted ids are sorted-vocabulary local ids shifted by the registry offset."""
    mermaid = _SourceStub("mermaid", ["coral", "sand"])
    train = _coralscapes_v2(monkeypatch, ["sand", "acropora alive"])
    val = _coralscapes_v2(monkeypatch, ["fish", "sand"])
    dataset_dict = {("mermaid", "train"): mermaid, ("cs", "train"): train, ("cs", "val"): val}

    vocabularies, registry_datasets = prepare_splits_for_registry(dataset_dict)

    # The union of both splits, sorted: not the order either split was built with.
    assert vocabularies["coralscapes_v2"].source_id2name == {
        1: "acropora alive",
        2: "fish",
        3: "sand",
    }
    for ds in (train, val):
        assert ds.source_id2name == {1: "acropora alive", 2: "fish", 3: "sand"}
        assert [int(ds._native_to_local[i]) for i in (_ACROPORA_ALIVE, _FISH, _SAND)] == [1, 2, 3]
        assert int(ds._native_to_local[_ACANTHASTER]) == 0  # outside the vocabulary

    registry = SourceLabelRegistry(registry_datasets)
    attach_registry(registry, dataset_dict.values())

    assert registry.dataset_offsets == {"mermaid": 0, "coralscapes_v2": 2}
    for ds in (train, val):
        assert ds.global_offset == 2
        assert [int(ds._native_to_local[i]) for i in (_ACROPORA_ALIVE, _FISH, _SAND)] == [3, 4, 5]
        assert int(ds._native_to_local[_ACANTHASTER]) == 0
    assert registry.global_id2source[3] == ("coralscapes_v2", "acropora alive")
    assert registry.global_id2source[5] == ("coralscapes_v2", "sand")


def test_attach_registry_rejects_a_dataset_whose_source_is_not_registered() -> None:
    registry = SourceLabelRegistry([_SourceStub("mermaid", ["coral"])])

    with pytest.raises(KeyError, match="'catlin_seaview' is not registered"):
        attach_registry(registry, [_SourceStub("catlin_seaview", ["sand"])])


def _make_coralscapes_v2_stub(*, source_id2name: dict[int, str]) -> CoralscapesV2Dataset:
    """CoralscapesV2Dataset shell without loading the HuggingFace corpus."""
    ds = CoralscapesV2Dataset.__new__(CoralscapesV2Dataset)
    ds.split = "train"
    ds.transform = None
    ds.class_subset = None
    ds._global_offset = 0
    ds.source_id2name = dict(source_id2name)
    ds.source_name2id = {name: local_id for local_id, name in source_id2name.items()}
    ds.num_source_classes = len(source_id2name) + 1
    ds._native_to_local = ds._build_native_to_local()
    return ds


def test_coralscapes_v2_concepts_follow_the_registry_lookup() -> None:
    """Concept rows are reached through the global ids a Coralscapes V2 split emits."""
    native_order = {i: CORALSCAPES_V2_ID2NAME[i] for i in sorted(CORALSCAPES_V2_ID2NAME)}
    train = _make_coralscapes_v2_stub(source_id2name=native_order)
    val = _make_coralscapes_v2_stub(source_id2name=native_order)
    dataset_dict = {("coralscapes_v2", "train"): train, ("coralscapes_v2", "val"): val}

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    registry = SourceLabelRegistry(
        registry_datasets,
        compute_concepts=True,
        concept_schema=ConceptSchema.from_csv(sources={"coralscapes_v2"}),
    )
    attach_registry(registry, dataset_dict.values())

    human_native_id = next(
        native_id for native_id, name in CORALSCAPES_V2_ID2NAME.items() if name == "human"
    )
    background_native_id = next(
        native_id for native_id, name in CORALSCAPES_V2_ID2NAME.items() if name == "background"
    )

    # After attach_registry, the lookup already emits global ids (offset folded in).
    offset = registry.dataset_offsets["coralscapes_v2"]
    human_gid = int(train._native_to_local[human_native_id])
    background_gid = int(train._native_to_local[background_native_id])
    assert train.source_id2name[human_gid - offset] == "human"
    assert train.source_id2name[background_gid - offset] == "background"

    schema = ConceptSchema.from_csv(sources={"coralscapes_v2"})
    channel_names = list(schema.channel_names)
    human_row = schema.row_for("coralscapes_v2", "human")
    background_row = schema.row_for("coralscapes_v2", "background")

    assert registry.source_to_concepts is not None

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


def test_registry_representative_prefers_the_train_split_even_when_listed_last() -> None:
    train, val = _SourceStub("mermaid", ["a"]), _SourceStub("mermaid", ["b"])
    other_val = _SourceStub("catlin_seaview", ["c"])
    dataset_dict = {("m", "val"): val, ("m", "train"): Subset(train, [0]), ("c", "val"): other_val}

    reps = select_registry_representatives(dataset_dict)

    # Representatives are unwrapped datasets. catlin has no train split, so its only split stands in.
    assert reps == [train, other_val]


def test_registry_representative_falls_back_to_the_first_split_without_prefer_split() -> None:
    val, test = _SourceStub("mermaid", ["a"]), _SourceStub("mermaid", ["b"])

    reps = select_registry_representatives({("m", "val"): val, ("m", "test"): test})

    assert reps == [val]


def test_root_dataset_unwraps_nested_subsets_but_stops_at_a_dataset_with_source_name() -> None:
    inner = _SourceStub("mermaid", ["a"])
    inner.dataset = object()  # e.g. the HuggingFace table a real dataset wraps

    assert _root_dataset(Subset(Subset(inner, [0]), [0])) is inner
    assert _root_dataset(inner) is inner


def test_group_splits_requires_a_source_name() -> None:
    with pytest.raises(TypeError, match="no SOURCE_NAME"):
        group_splits({("x", "train"): Subset(Dataset(), [])})


def test_benthos_yuval_global_lookup_rebuilds_on_vocab_change() -> None:
    """classes.json lookup must track registry vocabulary reassignment."""
    ds = BenthosYuvalCoralsDataset.__new__(BenthosYuvalCoralsDataset)
    ds._global_offset = 0
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


def test_ucsd_mosaics_native_lookup_rebuilds_on_vocab_change() -> None:
    """Native-to-local lookup must track registry vocabulary reassignment."""
    ds = UCSDMosaicsDataset.__new__(UCSDMosaicsDataset)
    ds._global_offset = 0
    ds.class_table = [
        {"id": 1, "name": "sand"},
        {"id": 2, "name": "coral"},
    ]
    ds.source_id2name = {1: "sand", 2: "coral"}
    ds.source_name2id = {"sand": 1, "coral": 2}
    ds.num_source_classes = 3
    ds._native_to_local = ds._build_native_to_local()

    ds.set_source_vocabulary(
        {1: "coral", 2: "sand"},
        {"coral": 1, "sand": 2},
        3,
    )

    assert int(ds._native_to_local[1]) == ds.source_name2id["sand"]
    assert int(ds._native_to_local[2]) == ds.source_name2id["coral"]
