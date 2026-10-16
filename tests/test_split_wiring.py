"""Tests for label-integrity fixes in registry vocabulary wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.dataset_reconciliation import (
    SourceLabelRegistry,
    attach_registry,
    prepare_splits_for_registry,
)
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.datasets.base_dataset import BaseCoralDataset
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


class _PointStub(BaseCoralDataset):
    """Minimal point-annotation dataset (CoralNet-shaped) with no I/O."""

    SOURCE_NAME = "coralnet"

    def read_image(self, **row_kwargs: object) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)


def _make_point_stub(
    labels: list[str],
    image_ids: list[str] | None = None,
    cls: type[BaseCoralDataset] = _PointStub,
) -> BaseCoralDataset:
    """One point per label at (row=i, col=0); all on image "a" unless ``image_ids``
    given."""
    ann = pd.DataFrame(
        {
            "image_id": image_ids or ["a"] * len(labels),
            "source_id": [1] * len(labels),
            "row": list(range(len(labels))),
            "col": [0] * len(labels),
            "source_label_name": labels,
        }
    )
    images = ann[["source_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    return cls(ann, images)


def _point_targets(registry: SourceLabelRegistry, ds: BaseCoralDataset, idx: int) -> list[str]:
    """Target names of the annotated pixels (col 0, non-background) of item ``idx``."""
    _, mask = ds[idx]
    column = torch.as_tensor(mask[:, 0])
    column = column[column > 0]
    return [registry.target_id2label[int(t)] for t in registry.source_to_target[column]]


def test_point_dataset_targets_survive_vocab_canonicalization() -> None:
    """Canonical (sorted) vocab must remap baked point label ids, not just swap the
    maps.

    Train ({"10", "9"}) and val ({"11", "9"}) have different label sets, so each split's
    local ids differ from the canonical union ("10", "11", "9") and a stale index would
    permute them.
    """
    train_labels = ["9", "9", "9", "10"]
    val_labels = ["11", "11", "9"]
    train, val = _make_point_stub(train_labels), _make_point_stub(val_labels)
    datasets = {("coralnet", "train"): train, ("coralnet", "val"): val}

    _, reps = prepare_splits_for_registry(datasets)
    registry = SourceLabelRegistry(
        reps,
        source_to_target_name_maps={"coralnet": {"9": "sand", "10": "porites", "11": "rubble"}},
        fetch_remote=False,
    )
    attach_registry(registry, datasets.values())

    expected = {"9": "sand", "10": "porites", "11": "rubble"}
    for ds, labels in ((train, train_labels), (val, val_labels)):
        _, mask = ds[0]
        target_ids = registry.source_to_target[torch.as_tensor(mask[: len(labels), 0])]
        got = [registry.target_id2label[int(t)] for t in target_ids]
        assert got == [expected[name] for name in labels]


def test_vocab_setter_rejects_dataset_with_annotation_index_but_no_setter() -> None:
    """A point dataset must never silently get maps swapped without remapping its
    index."""
    from mermaidseg.dataset_reconciliation.split_wiring import _apply_source_vocabulary

    class NoSetter:
        SOURCE_NAME = "x"
        _ann_label_id = np.array([1, 2])

    with pytest.raises(TypeError):
        _apply_source_vocabulary(NoSetter(), {1: "a"}, {"a": 1}, 2)


def test_base_dataset_rejects_duplicate_image_ids() -> None:
    ann = pd.DataFrame(
        {
            "image_id": ["a", "a"],
            "source_id": [1, 2],
            "row": [0, 1],
            "col": [0, 0],
            "source_label_name": ["9", "10"],
        }
    )
    images = ann[["source_id", "image_id"]].reset_index(drop=True)
    with pytest.raises(ValueError, match="image_id"):
        _PointStub(ann, images)


def test_dense_dataset_allows_image_ids_repeated_across_sites() -> None:
    """Dense subclasses (no row/col) never build the image_id index, e.g. Benthos
    tiles."""
    ann = pd.DataFrame(
        {
            "image_id": ["0_0", "0_0"],
            "source_id": [1, 2],
            "source_label_name": ["sand", "coral"],
        }
    )
    ds = _PointStub(ann, ann[["source_id", "image_id"]].reset_index(drop=True))
    assert ds._ann_label_id is None


def test_point_dataset_mixed_case_labels_resolve_through_registry() -> None:
    """Capitalized source names, map keys and target subset must still hit the right
    target.

    Covers #148/#149 on the point path: train spells "Sand", val spells "sand", the map
    is keyed in yet another casing, and the subset is capitalized.
    """
    train_labels = ["Porites", "Porites", "Sand"]
    val_labels = ["sand", "Porites", "sand"]
    train, val = _make_point_stub(train_labels), _make_point_stub(val_labels)
    datasets = {("coralnet", "train"): train, ("coralnet", "val"): val}

    _, reps = prepare_splits_for_registry(datasets)
    registry = SourceLabelRegistry(
        reps,
        source_to_target_name_maps={"coralnet": {"PORITES": "Porites", "sand": "Sand"}},
        target_label_subset=["Porites", "Sand"],
        fetch_remote=False,
    )
    attach_registry(registry, datasets.values())

    assert set(registry.target_label2id) == {"porites", "sand"}
    for ds, labels in ((train, train_labels), (val, val_labels)):
        _, mask = ds[0]
        target_ids = registry.source_to_target[torch.as_tensor(mask[: len(labels), 0])]
        got = [registry.target_id2label[int(t)] for t in target_ids]
        assert got == [name.lower() for name in labels]


class _MermaidPointStub(_PointStub):
    SOURCE_NAME = "mermaid"


def test_point_targets_correct_across_images_and_nonzero_offset() -> None:
    """Remap must keep per-image slices intact and compose with a nonzero global
    offset."""
    mermaid = _make_point_stub(["Sand", "Porites"], cls=_MermaidPointStub)
    train_labels = ["9", "10", "9", "10", "10"]
    train_images = ["a", "a", "b", "c", "c"]
    train = _make_point_stub(train_labels, train_images)
    val = _make_point_stub(["11", "9"], ["d", "e"])
    datasets = {
        ("mermaid", "train"): mermaid,
        ("coralnet", "train"): train,
        ("coralnet", "val"): val,
    }

    _, reps = prepare_splits_for_registry(datasets)
    registry = SourceLabelRegistry(
        reps,
        source_to_target_name_maps={
            "mermaid": {"Sand": "sand", "Porites": "porites"},
            "coralnet": {"9": "sand", "10": "porites", "11": "rubble"},
        },
        fetch_remote=False,
    )
    attach_registry(registry, datasets.values())

    assert train.global_offset > 0
    # Images sorted by df_images order: a, b, c; points stay at their own row.
    _, mask_c = train[2]
    assert mask_c[0, 0] == 0  # row 0 belongs to image "a", not "c"
    assert _point_targets(registry, train, 0) == ["sand", "porites"]
    assert _point_targets(registry, train, 1) == ["sand"]
    assert _point_targets(registry, train, 2) == ["porites", "porites"]
    assert _point_targets(registry, val, 0) == ["rubble"]
    assert _point_targets(registry, val, 1) == ["sand"]
    assert _point_targets(registry, mermaid, 0) == ["sand", "porites"]


def test_attach_without_prepare_raises_clear_error_and_keeps_state() -> None:
    """A val-only label missing from the registered vocab must fail loudly, not
    permute."""
    train, val = _make_point_stub(["9", "10"]), _make_point_stub(["11", "9"])
    registry = SourceLabelRegistry(
        [train],
        source_to_target_name_maps={"coralnet": {"9": "sand", "10": "porites"}},
        fetch_remote=False,
    )
    before = (dict(val.source_id2name), val._ann_label_id.copy())

    with pytest.raises(KeyError, match="'11' is missing from the new vocabulary"):
        attach_registry(registry, [val])

    assert val.source_id2name == before[0]
    np.testing.assert_array_equal(val._ann_label_id, before[1])


def test_base_dataset_rejects_null_image_id() -> None:
    """A single null image_id passes the duplicate check but would still merge
    annotations."""
    ann = pd.DataFrame(
        {
            "image_id": ["a", None, None],
            "source_id": [1, 1, 1],
            "row": [0, 1, 2],
            "col": [0, 0, 0],
            "source_label_name": ["9", "10", "10"],
        }
    )
    images = ann[["source_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    with pytest.raises(ValueError, match="must not be null"):
        _PointStub(ann, images)


def test_annotation_index_warns_on_dropped_annotations(caplog) -> None:
    """Annotations whose image is missing from df_images must be counted, not vanish."""
    ann = pd.DataFrame(
        {
            "image_id": ["a", "a", "ghost"],
            "source_id": [1, 1, 1],
            "row": [0, 1, 2],
            "col": [0, 0, 0],
            "source_label_name": ["9", "10", "9"],
        }
    )
    images = pd.DataFrame({"source_id": [1], "image_id": ["a"]})
    with caplog.at_level("WARNING", logger="mermaidseg.datasets.base_dataset"):
        ds = _PointStub(ann, images)
    assert len(ds._ann_label_id) == 2
    assert "dropped 1 of 3 annotations" in caplog.text
    assert "1 with no matching image" in caplog.text
