"""Counting CoralNet source x binary concept points and picking the only_false pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    binary_concepts,
    meta_path,
    only_false_pairs,
    source_concept_counts,
)

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


def test_binary_concepts_skip_taxonomic_ranks():
    assert binary_concepts(_schema()) == ["bleached", "branching"]


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


def test_meta_path_sits_next_to_the_csv(tmp_path):
    assert (
        meta_path(tmp_path / "coralnet_only_false.csv")
        == tmp_path / "coralnet_only_false.meta.yaml"
    )
