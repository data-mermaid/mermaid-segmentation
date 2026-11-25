"""SourceLabelRegistry: global ids from the readers' offsets and labels.parquet, and roll-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import torch

from mermaidseg.data import DenseDataset, PointDataset
from mermaidseg.dataset_reconciliation import registry as registry_module
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.registry import (
    SourceLabelRegistry,
    source_labels_to_target_labels,
)
from segdata_spec.taxonomy import hierarchy_from_table
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import ImageSpec, open_written


def _label(label_id: int, name: str, attribute: str | None) -> dict[str, Any]:
    return {"label_id": label_id, "name": name, "mermaid_benthic_attribute": attribute}


def _named(labels: dict[int, str]) -> list[dict[str, Any]]:
    """Rows whose benthic attribute is the label name."""
    return [_label(i, name, name) for i, name in labels.items()]


def _reader(
    tmp_path: Path,
    labels: list[dict[str, Any]],
    *,
    name: str = "coralnet",
    global_offset: int = 0,
) -> PointDataset:
    """A points reader whose one image holds a single point of the first label."""
    version = open_written(
        tmp_path / name,
        kind="points",
        images=[ImageSpec("i1")],
        labels=labels,
        points=[("i1", 0, 0, labels[0]["label_id"])],
        name=name,
    )
    return PointDataset(version, split="train", global_offset=global_offset)


@pytest.fixture()
def schema(tmp_path: Path) -> ConceptSchema:
    """A concept schema whose rows belong to ``coralnet``; labels of other sources map to zeros."""
    csv = tmp_path / "schema_concepts.csv"
    concept_table().assign(source_dataset_source="coralnet").to_csv(csv, index=False)
    return ConceptSchema.from_csv(csv, sources={"coralnet"})


def test_global_ids_cover_every_labels_parquet_row_at_the_readers_offset(
    schema: ConceptSchema, tmp_path: Path
) -> None:
    # Only label 3 has a point; label 7 has none in the selection and still gets an id.
    reader = _reader(tmp_path, _named({3: "a", 7: "b"}), global_offset=5)

    registry = SourceLabelRegistry([reader], schema)

    assert registry.dataset_offsets == {"coralnet": 5}
    assert registry.global_id2source == {8: ("coralnet", "a"), 12: ("coralnet", "b")}


def test_source_to_target_follows_each_rows_benthic_attribute(
    schema: ConceptSchema, tmp_path: Path
) -> None:
    labels = [
        _label(1, "Acro", "Acropora"),
        _label(2, "Sandy", "Sand"),
        _label(3, "Mystery", None),
        _label(4, "Blank", ""),
    ]
    reader = _reader(tmp_path, labels, global_offset=10)

    registry = SourceLabelRegistry([reader], schema)

    assert registry.target_label2id == {"acropora": 1, "sand": 2}
    assert registry.source_to_target.tolist() == [0] * 11 + [1, 2, 0, 0]


def test_lookups_span_the_largest_offset_plus_label_id(
    schema: ConceptSchema, tmp_path: Path
) -> None:
    first = _reader(tmp_path, _named({58: "x", 100: "y"}), name="a")
    second = _reader(tmp_path, _named({1: "p", 9: "q"}), name="b", global_offset=100)

    registry = SourceLabelRegistry([first, second], schema)

    assert registry.num_global_source_classes == 109
    assert registry.source_to_target.shape == (110,)


def test_default_targets_are_the_sorted_union_across_sources(
    schema: ConceptSchema, tmp_path: Path
) -> None:
    mermaid = _reader(tmp_path, _named({1: "acropora", 2: "sand"}), name="mermaid")
    scapes = _reader(
        tmp_path, _named({1: "sand", 2: "fish"}), name="coralscapes_v2", global_offset=2
    )

    registry = SourceLabelRegistry([mermaid, scapes], schema)

    assert registry.target_label2id == {"acropora": 1, "fish": 2, "sand": 3}
    assert registry.source_to_target.tolist() == [0, 1, 3, 3, 2]


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
    schema: ConceptSchema, tmp_path, attribute, roll_up, expected
) -> None:
    reader = _reader(tmp_path, [_label(1, "label", attribute)])

    registry = SourceLabelRegistry(
        [reader],
        schema,
        target_labels=["hard coral"],
        target_label_subset=["Hard coral"],
        label_roll_up=roll_up,
        benthic_hierarchy=hierarchy_from_table(_TAXONOMY),
    )

    assert registry.source_to_target.tolist() == expected


def test_label_roll_up_without_a_hierarchy_raises(schema: ConceptSchema, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires benthic_hierarchy"):
        SourceLabelRegistry(
            [_reader(tmp_path, _named({1: "sand"}))],
            schema,
            target_label_subset=["sand"],
            label_roll_up=True,
        )


def test_label_roll_up_without_a_subset_raises(schema: ConceptSchema, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires target_label_subset"):
        SourceLabelRegistry(
            [_reader(tmp_path, _named({1: "sand"}))],
            schema,
            label_roll_up=True,
            benthic_hierarchy={},
        )


def test_target_label_subset_collapses_other_targets_to_background(
    schema: ConceptSchema, tmp_path: Path
) -> None:
    labels = [_label(1, "acropora alive", "Acropora"), _label(2, "sand", "Sand")]

    registry = SourceLabelRegistry(
        [_reader(tmp_path, labels)], schema, target_label_subset=["Sand"]
    )

    assert registry.target_label2id == {"sand": 1}
    assert registry.source_to_target.tolist() == [0, 0, 1]


def test_empty_dataset_list_raises(
    schema: ConceptSchema,
) -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        SourceLabelRegistry([], schema)


def test_duplicate_source_name_raises(schema: ConceptSchema, tmp_path: Path) -> None:
    first = _reader(tmp_path / "1", _named({1: "a"}), name="mermaid")
    second = _reader(tmp_path / "2", _named({1: "b"}), name="mermaid")
    with pytest.raises(ValueError, match="Duplicate source_name 'mermaid'"):
        SourceLabelRegistry([first, second], schema)


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
        [PointDataset(version, split="train", global_offset=0)],
        concept_schema=schema,
    )

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


def test_dense_concepts_follow_the_registry_lookup(tmp_path: Path) -> None:
    """Concept rows are reached through the global ids a dense reader emits."""
    labels = [_label(1, "human", "human"), _label(2, "background", "background")]
    version = open_written(
        tmp_path,
        kind="dense",
        images=[ImageSpec("i1", mask=np.array([[1, 2]], np.uint8))],
        labels=labels,
        name="coralscapes_v2",
    )
    reader = DenseDataset(version, split="train", global_offset=5)
    schema = ConceptSchema.from_csv(sources={"coralscapes_v2"})

    registry = SourceLabelRegistry([reader], concept_schema=schema)

    global_ids = torch.as_tensor(reader[0][1]).long()[None]
    assert global_ids.tolist() == [[[6, 7]]]
    concepts = source_labels_to_concepts(global_ids, registry.source_to_concepts)
    np.testing.assert_array_equal(
        concepts[0, :, 0, 0].numpy(), schema.row_for("coralscapes_v2", "human")
    )
    np.testing.assert_array_equal(
        concepts[0, :, 0, 1].numpy(), schema.row_for("coralscapes_v2", "background")
    )


def _coverage_record(caplog: pytest.LogCaptureFixture) -> Any:
    return next(r for r in caplog.records if "source labels" in r.getMessage())


def test_coverage_is_logged_at_info_when_some_labels_map(
    schema: ConceptSchema, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reader = _reader(tmp_path, [_label(1, "sand", "sand"), _label(2, "zzz", None)])

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([reader], schema)

    record = _coverage_record(caplog)
    assert record.levelname == "INFO"
    assert "1 of 2 source labels" in record.getMessage()


def test_coverage_warns_when_no_label_maps(
    schema: ConceptSchema, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reader = _reader(tmp_path, [_label(1, "zz1", None), _label(2, "zz2", None)])

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([reader], schema)

    record = _coverage_record(caplog)
    assert record.levelname == "WARNING"
    assert "check source label keys against the mapping" in record.getMessage()


def test_coverage_is_judged_per_source_at_its_own_offset(
    schema: ConceptSchema, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mermaid = _reader(tmp_path, _named({1: "a", 2: "b"}), name="mermaid")
    scapes = _reader(
        tmp_path,
        [_label(1, "zz1", None), _label(2, "zz2", None)],
        name="coralscapes_v2",
        global_offset=2,
    )

    with caplog.at_level("INFO", logger=registry_module.logger.name):
        SourceLabelRegistry([mermaid, scapes], schema)

    levels = {
        name: next(r.levelname for r in caplog.records if r.getMessage().startswith(f"{name}:"))
        for name in ("mermaid", "coralscapes_v2")
    }
    assert levels == {"mermaid": "INFO", "coralscapes_v2": "WARNING"}


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
