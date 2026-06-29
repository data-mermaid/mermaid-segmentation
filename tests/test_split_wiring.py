"""Tests for label-integrity fixes in registry vocabulary wiring."""

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
from mermaidseg.datasets.ucsd_mosaics.ucsd_mosaics_dataset import UCSDMosaicsDataset


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


def test_source_label_registry_normalizes_target_names_to_lowercase() -> None:
    """Lowercase static map values must match a capitalized class_subset."""

    class CoralscapesV2Stub:
        SOURCE_NAME = "coralscapes_v2"
        source_id2name = {1: "pocillopora alive", 2: "sand", 3: "background"}
        source_name2id = {"pocillopora alive": 1, "sand": 2, "background": 3}

        def set_global_offset(self, offset: int) -> None:
            self._global_offset = offset

    registry = SourceLabelRegistry(
        [CoralscapesV2Stub()],
        target_label_subset=["Pocillopora", "Sand", "Background"],
        fetch_remote=False,
    )

    assert registry.target_label2id == {"background": 1, "pocillopora": 2, "sand": 3}
    assert registry.target_id2label == {1: "background", 2: "pocillopora", 3: "sand"}

    offset = registry.dataset_offsets["coralscapes_v2"]
    pocillopora_gid = 1 + offset
    sand_gid = 2 + offset
    background_gid = 3 + offset
    assert int(registry.source_to_target[pocillopora_gid]) == 2
    assert int(registry.source_to_target[sand_gid]) == 3
    assert int(registry.source_to_target[background_gid]) == 1


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


def test_ucsd_mosaics_native_lookup_rebuilds_on_vocab_change() -> None:
    """Native-to-local lookup must track registry vocabulary reassignment."""
    ds = UCSDMosaicsDataset.__new__(UCSDMosaicsDataset)
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
