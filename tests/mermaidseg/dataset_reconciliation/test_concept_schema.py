"""ConceptSchema encoding of the concept table."""

from __future__ import annotations

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema

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
