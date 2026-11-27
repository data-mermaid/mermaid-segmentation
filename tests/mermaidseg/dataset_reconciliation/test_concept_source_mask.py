"""Counting CoralNet source x binary concept points and picking the only_false pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from mermaidseg.dataset_reconciliation.artifact_meta import meta_path
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    load_concept_source_mask,
    only_false_pairs,
    point_annotations,
    source_concept_counts,
)
from tests.mermaidseg.conftest import concept_table

ENC = {"not_given": 0, False: 1, True: 2}
# Channels: one taxonomic rank (genus, 2 channels) and two binary concepts.
CHANNELS = ("genus__acropora", "genus__porites", "bleached", "branching")
LABELS = {  # label -> (genus one-hot as 1/2, bleached, branching)
    "acropora": [2, 1, 1, 2],
    "bleached acropora": [2, 1, 2, 2],
    "porites": [1, 2, 1, 1],
    "sand": [1, 1, 1, 1],
    "unknown": [0, 0, 0, 0],  # every concept not_given
}


def _schema() -> ConceptSchema:
    return ConceptSchema(
        channel_names=CHANNELS,
        concept_value2id={
            "genus": {"not_given": 0, "acropora": 1, "porites": 2},
            "bleached": ENC,
            "branching": ENC,
        },
        num_channels=len(CHANNELS),
        row_by_key={("coralnet", name): np.array(row, np.float32) for name, row in LABELS.items()},
    )


def _points(rows: list[tuple[int, str, int]]) -> pd.DataFrame:
    """(source_id, label, number of points) -> one annotation row per point."""
    return pd.DataFrame(
        [(src, label) for src, label, n in rows for _ in range(n)],
        columns=["source_id", "source_label_name"],
    )


def test_counts_and_only_false_pairs():
    ann = _points(
        [
            (1, "acropora", 3),  # bleached False, branching True
            (1, "sand", 2),
            (2, "bleached acropora", 1),
            (2, "porites", 4),
            (3, "unknown", 5),  # only not_given
            (3, "not in the table", 2),  # missing from the schema: not_given too
        ]
    )
    counts = source_concept_counts(ann, _schema()).set_index(["source_id", "concept"])
    assert list(counts.reset_index().columns) == MASK_COLUMNS
    assert counts.loc[(1, "bleached")].tolist() == [0, 5, 0]
    assert counts.loc[(1, "branching")].tolist() == [3, 2, 0]
    assert counts.loc[(2, "bleached")].tolist() == [1, 4, 0]
    assert counts.loc[(2, "branching")].tolist() == [1, 4, 0]
    assert counts.loc[(3, "bleached")].tolist() == [0, 0, 7]
    assert len(counts) == 3 * 2  # every source x binary concept, masked or not

    masked = only_false_pairs(counts.reset_index())
    # Source 1 never labels bleached; source 3 has no True/False values at all (not masked).
    assert list(zip(masked["source_id"], masked["concept"], strict=True)) == [(1, "bleached")]


def test_non_numeric_source_ids_follow_the_numeric_ones_in_the_keep_table(tmp_path):
    path = tmp_path / "mask.csv"
    pd.DataFrame(
        {"source_id": ["b", "12", "7"], "concept": ["bleached"] * 3, "n_false": [1, 1, 1]}
    ).to_csv(path, index=False)
    meta_path(path).write_text(
        yaml.safe_dump({"dataset": "coralnet", "source_ids": ["b", "12", "7"]})
    )

    mask = load_concept_source_mask(path, _schema())

    assert mask.source_ids == ("7", "12", "b")
    assert mask.row_by_source() == {"7": 1, "12": 2, "b": 3}


def _point_rows(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "row": range(len(rows)),
            "col": 0,
            "label_id": [r[1] for r in rows],
            "annotator": "default",
        }
    )


SIZED_IMAGES = pd.DataFrame(
    {"image_id": ["a1", "a2"], "group": ["s7", "s30"], "height": 8, "width": 8}
)


def test_mermaid_points_count_under_the_concept_row_of_their_benthic_attribute(tmp_path):
    labels = pd.DataFrame(
        {
            "label_id": [1, 2],
            "name": ["Acropora::Branching", "Acropora::"],
            "mermaid_benthic_attribute": ["Acropora", "Acropora"],
        }
    )
    concepts = tmp_path / "concepts.csv"
    concept_table().assign(source_dataset_source="mermaid").to_csv(concepts, index=False)
    schema = ConceptSchema.from_csv(concepts, sources={"mermaid"})

    annotations = point_annotations(
        "mermaid", _point_rows([("a1", 1), ("a1", 2), ("a2", 2)]), SIZED_IMAGES, labels, None, None
    )
    counts = source_concept_counts(annotations, schema, dataset_source="mermaid")

    # Acropora: tubular F, branching T, bleached F, at both sources; sources sort numerically.
    assert counts.to_numpy().tolist() == [
        ["7", "tubular", 0, 2, 0],
        ["7", "branching", 2, 0, 0],
        ["7", "bleached", 0, 2, 0],
        ["30", "tubular", 0, 1, 0],
        ["30", "branching", 1, 0, 0],
        ["30", "bleached", 0, 1, 0],
    ]
