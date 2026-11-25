"""SourceLabelRegistry: offsets, source-to-target from the readers' label maps, and roll-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import torch

from mermaidseg.data import PointDataset
from mermaidseg.dataset_reconciliation import registry as registry_module
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.registry import (
    SourceLabelRegistry,
    source_labels_to_target_labels,
)
from segdata_spec.taxonomy import hierarchy_from_table
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, open_written


class _StubDataset:
    """Source-space dataset stub: only what the registry reads."""

    def __init__(
        self,
        source_name: str | None,
        names: list[str],
        targets: dict[str, str | None] | None = None,
    ) -> None:
        if source_name is not None:
            self.SOURCE_NAME = source_name
        self.source_id2name = dict(enumerate(names, start=1))
        self.source_name2id = {n: i for i, n in self.source_id2name.items()}
        self.label_targets = targets if targets is not None else {n: n for n in names}
        self.offset: int | None = None

    def set_global_offset(self, offset: int) -> None:
        self.offset = offset


def _reader(tmp_path: Path, labels: list[dict[str, Any]], counts: list[int]) -> PointDataset:
    """A points reader whose label ``i`` (1-based) has ``counts[i - 1]`` points."""
    points = [
        ("i1", row, label_id, label_id)
        for label_id, n in enumerate(counts, start=1)
        for row in range(n)
    ]
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1")],
        labels=labels,
        points=points,
        name="coralnet",
    )
    return PointDataset(version, split="train")


def _label(label_id: int, name: str, attribute: str | None) -> dict[str, Any]:
    return {"label_id": label_id, "name": name, "mermaid_benthic_attribute": attribute}


def test_source_to_target_follows_the_reader_label_map_and_leaves_unmapped_labels_ignored(
    tmp_path,
) -> None:
    labels = [
        _label(1, "Acro", "Acropora"),
        _label(2, "Sandy", "Sand"),
        _label(3, "Mystery", None),
        _label(4, "Blank", ""),
    ]
    reader = _reader(tmp_path, labels, [4, 3, 2, 1])

    registry = SourceLabelRegistry([reader])

    assert reader.source_id2name == {1: "Acro", 2: "Sandy", 3: "Mystery", 4: "Blank"}
    assert registry.target_label2id == {"acropora": 1, "sand": 2}
    assert registry.source_to_target.tolist() == [0, 1, 2, 0, 0]


_TAXONOMY = pa.table(
    {
        "id": ["1", "2", "3", "4"],
        "name": ["Hard coral", "Acropora", "Acropora palmata", "Orphan"],
        "parent_id": [None, "1", "2", "99"],
    }
)


@pytest.mark.parametrize(
    ("attribute", "roll_up", "expected"),
    [
        ("Acropora palmata", True, [0, 1]),  # species -> genus -> class_subset member
        ("Acropora palmata", False, [0, 0]),
        ("Orphan", True, [0, 0]),  # parent_id 99 is not in the table: roll-up stops
    ],
)
def test_label_roll_up_maps_a_target_onto_its_ancestor_in_the_class_subset(
    tmp_path, attribute, roll_up, expected
) -> None:
    reader = _reader(tmp_path, [_label(1, "label", attribute)], [1])

    registry = SourceLabelRegistry(
        [reader],
        target_labels=["hard coral"],
        target_label_subset=["Hard coral"],
        label_roll_up=roll_up,
        benthic_hierarchy=hierarchy_from_table(_TAXONOMY),
    )

    assert registry.source_to_target.tolist() == expected


def test_label_roll_up_without_a_hierarchy_raises() -> None:
    with pytest.raises(ValueError, match="requires benthic_hierarchy"):
        SourceLabelRegistry(
            [_StubDataset("mermaid", ["sand"])], target_label_subset=["sand"], label_roll_up=True
        )


def test_label_roll_up_without_a_subset_raises() -> None:
    with pytest.raises(ValueError, match="requires target_label_subset"):
        SourceLabelRegistry(
            [_StubDataset("mermaid", ["sand"])], label_roll_up=True, benthic_hierarchy={}
        )


def test_offsets_are_disjoint_across_datasets_and_lookups_span_all_of_them() -> None:
    mermaid = _StubDataset("mermaid", ["acropora", "sand"])
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
    scapes = _StubDataset(
        "coralscapes_v2", ["acropora alive", "sand"], {"acropora alive": "Acropora", "sand": "Sand"}
    )

    registry = SourceLabelRegistry([scapes], target_label_subset=["Sand"])

    assert registry.target_label2id == {"sand": 1}
    assert registry.source_to_target.tolist() == [0, 0, 1]


def test_empty_dataset_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        SourceLabelRegistry([])


def test_dataset_without_source_name_raises() -> None:
    with pytest.raises(ValueError, match="does not declare SOURCE_NAME"):
        SourceLabelRegistry([_StubDataset(None, ["sand"])])


def test_duplicate_source_name_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate SOURCE_NAME 'mermaid'"):
        SourceLabelRegistry([_StubDataset("mermaid", ["a"]), _StubDataset("mermaid", ["b"])])


def test_compute_concepts_without_a_schema_raises() -> None:
    with pytest.raises(ValueError, match="requires concept_schema"):
        SourceLabelRegistry([_StubDataset("mermaid", ["a"])], compute_concepts=True)


def test_mermaid_concept_rows_are_keyed_by_the_benthic_attribute_of_each_ba_gf_label(
    tmp_path,
) -> None:
    labels = [
        _label(1, "Acropora::Branching", "Acropora"),
        _label(2, "Acropora::", "Acropora"),
    ]
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1")],
        labels=labels,
        points=[("i1", 0, 0, 1), ("i1", 0, 1, 1), ("i1", 1, 0, 2)],
        name="mermaid",
    )
    concepts = tmp_path / "concepts.csv"
    concept_table().assign(source_dataset_source="mermaid").to_csv(concepts, index=False)
    schema = ConceptSchema.from_csv(concepts, sources={"mermaid"})

    registry = SourceLabelRegistry(
        [PointDataset(version, split="train")], compute_concepts=True, concept_schema=schema
    )

    assert registry.source_to_concepts is not None
    true_channels = [
        [name for name, v in zip(schema.channel_names, row, strict=True) if v == 2]
        for row in registry.source_to_concepts[1:].tolist()
    ]
    acropora = [
        "kingdom__animalia",
        "phylum__cnidaria",
        "class__hexacorallia",
        "order__scleractinia",
        "family__acroporidae",
        "genus__acropora",
        "branching",
    ]
    assert true_channels == [acropora, acropora]


class _AnnotatedStub(_StubDataset):
    def __init__(self, source_name: str, names: list[str], labels: list[str] | None) -> None:
        super().__init__(source_name, names, {"sand": "sand", "Sand": "Sand", "a": "a", "b": "b"})
        if labels is not None:
            self.df_annotations = pd.DataFrame({"source_label_name": labels})


def _coverage_record(caplog):
    return next(r for r in caplog.records if "source labels" in r.getMessage())


def test_coverage_is_logged_at_info_when_some_labels_map(caplog) -> None:
    ds = _AnnotatedStub("coralscapes_v2", ["sand", "zzz"], ["sand", "sand", "sand", "zzz"])

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([ds])

    record = _coverage_record(caplog)
    assert record.levelname == "INFO"
    assert "1 of 2 source labels" in record.getMessage()
    assert "75.0% of annotations" in record.getMessage()


def test_coverage_warns_when_no_label_maps(caplog) -> None:
    ds = _AnnotatedStub("coralscapes_v2", ["zz1", "zz2"], ["zz1", "zz2"])

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([ds])

    record = _coverage_record(caplog)
    assert record.levelname == "WARNING"
    assert "check source label keys against the mapping" in record.getMessage()


def test_coverage_is_judged_per_source_at_its_own_offset(caplog) -> None:
    mermaid = _AnnotatedStub("mermaid", ["a", "b"], ["a", "b"])
    scapes = _AnnotatedStub("coralscapes_v2", ["zz1", "zz2"], ["zz1", "zz2"])

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([mermaid, scapes])

    levels = {
        name: next(r.levelname for r in caplog.records if r.getMessage().startswith(f"{name}:"))
        for name in ("mermaid", "coralscapes_v2")
    }
    assert levels == {"mermaid": "INFO", "coralscapes_v2": "WARNING"}


def test_coverage_without_df_annotations_logs_label_counts_only(caplog) -> None:
    ds = _AnnotatedStub("coralscapes_v2", ["sand", "zzz"], None)

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([ds])

    record = _coverage_record(caplog)
    assert record.levelname == "INFO"
    assert "1 of 2 source labels" in record.getMessage()
    assert "of annotations" not in record.getMessage()


# ---------- source_labels_to_target_labels ----------


def test_source_labels_to_target_labels_indexes_the_lookup_for_a_non_square_batch():
    lookup = torch.tensor([0, 5, 7, 5])
    labels = torch.tensor([[[1, 0, 2], [3, 2, 0]]])  # (B=1, H=2, W=3)

    out = source_labels_to_target_labels(labels, lookup)

    assert out.tolist() == [[[5, 0, 7], [5, 7, 0]]]
    assert out.dtype == torch.long


def test_source_labels_to_target_labels_accepts_non_long_integer_labels():
    lookup = torch.tensor([0, 5, 7])
    labels = torch.tensor([[[2, 1]]], dtype=torch.uint8)

    assert source_labels_to_target_labels(labels, lookup).tolist() == [[[7, 5]]]
