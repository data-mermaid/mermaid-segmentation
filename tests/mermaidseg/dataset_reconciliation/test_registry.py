"""SourceLabelRegistry vocabulary normalisation."""

from __future__ import annotations

import pytest

from mermaidseg.dataset_reconciliation import registry as registry_module
from mermaidseg.dataset_reconciliation.registry import SourceLabelRegistry, roll_up_label


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


class _StubDataset:
    """Source-space dataset stub: only what the registry reads."""

    def __init__(self, source_name: str | None, names: list[str]) -> None:
        if source_name is not None:
            self.SOURCE_NAME = source_name
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {n: i for i, n in self.source_id2name.items()}
        self.offset: int | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def test_offsets_are_disjoint_across_datasets_and_lookups_span_all_of_them() -> None:
    mermaid = _StubDataset("mermaid", ["acropora", "sand"])  # identity mapping
    scapes = _StubDataset("coralscapes_v2", ["sand", "fish"])

    registry = SourceLabelRegistry([mermaid, scapes])

    assert registry.dataset_offsets == {"mermaid": 0, "coralscapes_v2": 2}
    assert (mermaid.offset, scapes.offset) == (0, 2)
    assert registry.global_id2source == {
        1: ("mermaid", "acropora"),
        2: ("mermaid", "sand"),
        3: ("coralscapes_v2", "sand"),
        4: ("coralscapes_v2", "fish"),
    }
    assert registry.num_global_source_classes == 4
    # Default targets are the sorted union: acropora=1, fish=2, sand=3; id 0 is background.
    assert registry.target_label2id == {"acropora": 1, "fish": 2, "sand": 3}
    assert registry.source_to_target.tolist() == [0, 1, 3, 3, 2]


def test_target_label_subset_collapses_other_targets_to_background() -> None:
    scapes = _StubDataset("coralscapes_v2", ["acropora alive", "sand"])

    registry = SourceLabelRegistry([scapes], target_label_subset=["sand"])

    assert registry.target_label2id == {"sand": 1}
    assert registry.source_to_target.tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("acropora", "hard coral"),  # rolls up one level
        ("hard coral", "hard coral"),  # already in the subset
        ("porites", "hard coral"),  # rolls up two levels
        ("sand", None),  # its chain ends at a root outside the subset
        ("not in hierarchy", None),
        (None, None),
    ],
)
def test_roll_up_label_returns_the_nearest_ancestor_in_the_subset(label, expected) -> None:
    hierarchy = {
        "acropora": "hard coral",
        "porites": "massive",
        "massive": "hard coral",
        "hard coral": None,
        "sand": "bare",
        "bare": None,
    }

    assert roll_up_label(label, hierarchy, {"hard coral"}) == expected


def _patch_hierarchy(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_module,
        "initialize_benthic_hierarchy",
        lambda: {"Acropora": "Hard coral", "Pocillopora": "Hard coral", "Hard coral": None},
    )


def test_label_roll_up_maps_unlisted_targets_onto_their_ancestor_in_the_subset(
    monkeypatch,
) -> None:
    _patch_hierarchy(monkeypatch)
    scapes = _StubDataset("coralscapes_v2", ["acropora alive", "pocillopora alive", "sand"])

    registry = SourceLabelRegistry(
        [scapes],
        target_labels=["hard coral"],
        target_label_subset=["Hard coral"],
        label_roll_up=True,
    )

    # Acropora and Pocillopora roll up to hard coral; sand has no ancestor in the subset.
    assert registry.source_to_target.tolist() == [0, 1, 1, 0]


def test_without_label_roll_up_the_same_labels_collapse_to_background(monkeypatch) -> None:
    scapes = _StubDataset("coralscapes_v2", ["acropora alive", "pocillopora alive", "sand"])

    registry = SourceLabelRegistry(
        [scapes], target_labels=["hard coral"], target_label_subset=["Hard coral"]
    )

    assert registry.source_to_target.tolist() == [0, 0, 0, 0]


def test_label_roll_up_without_a_subset_raises(monkeypatch) -> None:
    _patch_hierarchy(monkeypatch)

    with pytest.raises(ValueError, match="requires target_label_subset"):
        SourceLabelRegistry([_StubDataset("coralscapes_v2", ["sand"])], label_roll_up=True)


def test_remote_fetcher_map_is_filtered_to_the_dataset_and_drops_unmapped(monkeypatch) -> None:
    monkeypatch.setitem(
        registry_module._REMOTE_FETCHERS,
        "coralnet",
        lambda: {"Acropora": "Acropora", "Sand": None, "Fish": "Fish"},
    )
    coralnet = _StubDataset("coralnet", ["acropora", "sand"])

    registry = SourceLabelRegistry([coralnet])

    # Fish is not in the dataset; sand maps to None.
    assert registry.target_label2id == {"acropora": 1}
    assert registry.source_to_target.tolist() == [0, 1, 0]


def test_empty_dataset_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        SourceLabelRegistry([])


def test_dataset_without_source_name_raises() -> None:
    with pytest.raises(ValueError, match="does not declare SOURCE_NAME"):
        SourceLabelRegistry([_StubDataset(None, ["sand"])])


def test_duplicate_source_name_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate SOURCE_NAME 'mermaid'"):
        SourceLabelRegistry([_StubDataset("mermaid", ["a"]), _StubDataset("mermaid", ["b"])])


def test_source_name_without_a_mapping_raises() -> None:
    with pytest.raises(ValueError, match="No default source-to-target mapping"):
        SourceLabelRegistry([_StubDataset("unknown_source", ["a"])])


def test_compute_concepts_without_a_schema_raises() -> None:
    with pytest.raises(ValueError, match="requires concept_schema"):
        SourceLabelRegistry([_StubDataset("mermaid", ["a"])], compute_concepts=True)
