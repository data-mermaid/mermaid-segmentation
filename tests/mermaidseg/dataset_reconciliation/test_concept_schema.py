"""ConceptSchema encoding of the concept table."""

from __future__ import annotations

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.io import ConfigDict
from tests.mermaidseg.conftest import (
    concept_table,
    label_space_reader,
    label_space_section,
    write_test_label_space,
)
from tests.mermaidseg.data.conftest import VERSION

_BINARY = {"not_given": 0, "FALSE": 1, "TRUE": 2}


def _schema(channels: tuple[str, ...]) -> ConceptSchema:
    return ConceptSchema(
        channel_names=channels,
        concept_value2id={
            "genus": {"not_given": [0, 0], "acropora": [2, 1], "porites": [1, 2]},
            "bleached": _BINARY,
            "branching": _BINARY,
        },
        num_channels=len(channels),
        row_by_key={},
    )


def test_binary_names_skip_the_taxonomic_ranks():
    schema = _schema(("genus__acropora", "genus__porites", "bleached", "branching"))

    assert schema.binary_names() == ["bleached", "branching"]


def test_a_schema_from_the_label_space_matches_the_schema_from_its_concept_table(tmp_path):
    csv = tmp_path / "concepts.csv"
    concept_table().to_csv(csv, index=False)
    from_csv = ConceptSchema.from_csv(
        csv, sources={"coralnet"}, removed_concepts=["genus__acropora"]
    )
    training_cfg = ConfigDict(
        {
            "concept_mapping_path": str(csv),
            "concept_removal": "",
            "concept_source_mask": "",
            "class_subset": ["Sand"],
            "label_roll_up": False,
        }
    )
    reader = label_space_reader(tmp_path, "coralnet", {1: ("sand", "Sand")})
    space = write_test_label_space(
        tmp_path / "space",
        {"coralnet": reader.version},
        training_cfg,
        from_csv,
        label_space_section({"coralnet": VERSION}),
    )

    schema = ConceptSchema.from_label_space(space)

    assert schema.channel_names == from_csv.channel_names
    assert schema.concept_value2id == from_csv.concept_value2id
    assert schema.num_channels == from_csv.num_channels
    assert schema.rank_slices() == from_csv.rank_slices()
    assert schema.binary_offset == from_csv.binary_offset
    assert schema.binary_names() == from_csv.binary_names()
