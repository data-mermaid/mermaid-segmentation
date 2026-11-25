"""Tests for label-integrity fixes in registry vocabulary wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset

from mermaidseg.data import DenseDataset
from mermaidseg.data.config import Split
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
from segdata_spec.read import open_version
from tests.mermaidseg.data.conftest import ImageSpec, write_version


class _SourceStub(Dataset):
    """Dataset stub exposing only what split wiring and the registry read."""

    def __init__(self, source_name: str, names: list[str]) -> None:
        self.SOURCE_NAME = source_name
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {n: i for i, n in self.source_id2name.items()}
        self.num_source_classes = len(names) + 1
        self.label_targets = {n: n for n in names}
        self.offset: int | None = None
        self.dataset: object | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def _dense_readers(tmp_path, labels: dict[int, str], masks: dict[Split, np.ndarray]) -> dict:
    """One ``coralscapes_v2`` dense reader per split, the split's image holding ``masks[split]``."""
    images = [ImageSpec(split, group=split, mask=mask) for split, mask in masks.items()]
    rows = [
        {"label_id": i, "name": name, "mermaid_benthic_attribute": name}
        for i, name in labels.items()
    ]
    cache_dir, bucket, pin = write_version(
        tmp_path, kind="dense", images=images, labels=rows, name="coralscapes_v2"
    )
    version = open_version(cache_dir, bucket, pin)
    return {split: DenseDataset(version, split=split, groups=[split]) for split in masks}


# Version label ids of the names used below.
_ACROPORA_ALIVE, _FISH, _SAND = 2, 25, 69


def test_dense_masks_follow_the_sorted_registry_vocab_and_offset(tmp_path) -> None:
    """Emitted ids are sorted-vocabulary local ids shifted by the registry offset."""
    train_mask = np.array([[_SAND, _ACROPORA_ALIVE]], np.uint8)
    val_mask = np.array([[_FISH, _SAND]], np.uint8)
    labels = {_ACROPORA_ALIVE: "acropora alive", _FISH: "fish", _SAND: "sand"}
    readers = _dense_readers(tmp_path, labels, {"train": train_mask, "val": val_mask})
    mermaid = _SourceStub("mermaid", ["coral", "sand"])
    dataset_dict = {
        ("mermaid", "train"): mermaid,
        ("cs", "train"): readers["train"],
        ("cs", "val"): readers["val"],
    }

    vocabularies, registry_datasets = prepare_splits_for_registry(dataset_dict)

    # The union of both splits, sorted: not the order either split was built with.
    assert vocabularies["coralscapes_v2"].source_id2name == {
        1: "acropora alive",
        2: "fish",
        3: "sand",
    }
    assert readers["train"][0][1].tolist() == [[3, 1]]

    registry = SourceLabelRegistry(registry_datasets)
    attach_registry(registry, dataset_dict.values())

    assert registry.dataset_offsets == {"mermaid": 0, "coralscapes_v2": 2}
    assert readers["train"][0][1].tolist() == [[5, 3]]
    assert readers["val"][0][1].tolist() == [[4, 5]]
    assert registry.global_id2source[3] == ("coralscapes_v2", "acropora alive")
    assert registry.global_id2source[5] == ("coralscapes_v2", "sand")


def test_attach_registry_rejects_a_dataset_whose_source_is_not_registered() -> None:
    registry = SourceLabelRegistry([_SourceStub("mermaid", ["coral"])])

    with pytest.raises(KeyError, match="'catlin_seaview' is not registered"):
        attach_registry(registry, [_SourceStub("catlin_seaview", ["sand"])])


def test_dense_concepts_follow_the_registry_lookup(tmp_path) -> None:
    """Concept rows are reached through the global ids a dense reader emits."""
    labels = {1: "human", 2: "background"}
    mask = np.array([[1, 2]], np.uint8)
    readers = _dense_readers(tmp_path, labels, {"train": mask, "val": mask})
    dataset_dict = {("coralscapes_v2", split): ds for split, ds in readers.items()}
    schema = ConceptSchema.from_csv(sources={"coralscapes_v2"})

    _, registry_datasets = prepare_splits_for_registry(dataset_dict)
    registry = SourceLabelRegistry(registry_datasets, compute_concepts=True, concept_schema=schema)
    attach_registry(registry, dataset_dict.values())

    assert registry.source_to_concepts is not None
    global_ids = torch.as_tensor(readers["train"][0][1]).long()[None]
    concepts = source_labels_to_concepts(global_ids, registry.source_to_concepts)
    channel_names = list(schema.channel_names)
    np.testing.assert_array_equal(
        concepts[0, :, 0, 0].numpy(), schema.row_for("coralscapes_v2", "human")
    )
    np.testing.assert_array_equal(
        concepts[0, :, 0, 1].numpy(), schema.row_for("coralscapes_v2", "background")
    )
    assert int(concepts[0, channel_names.index("human"), 0, 0]) == 2
    assert int(concepts[0, channel_names.index("human"), 0, 1]) == 1


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
