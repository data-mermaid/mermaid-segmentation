"""Counting CoralNet source x binary concept points and picking the only_false pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.concept_source_mask import (
    MASK_COLUMNS,
    _source_sort_key,
    binary_concepts,
    load_concept_source_mask,
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


def test_binary_concept_without_a_channel_of_its_own_raises():
    schema = ConceptSchema(
        channel_names=("genus__acropora", "branching"),
        concept_value2id={"genus": {"not_given": 0, "acropora": 1}, "bleached": ENC},
        num_channels=2,
        row_by_key={},
    )

    with pytest.raises(ValueError, match=r"without a channel of the same name: \['bleached'\]"):
        binary_concepts(schema)


def test_loading_a_mask_without_its_sidecar_raises_file_not_found(tmp_path):
    path = tmp_path / "mask.csv"
    pd.DataFrame({"source_id": ["1"], "concept": ["bleached"]}).to_csv(path, index=False)

    with pytest.raises(FileNotFoundError, match="no sidecar"):
        load_concept_source_mask(path, _schema())


def test_loading_a_mask_missing_a_required_column_raises(tmp_path):
    path = tmp_path / "mask.csv"
    pd.DataFrame({"source_id": ["1"]}).to_csv(path, index=False)
    meta_path(path).write_text(yaml.safe_dump({"dataset": "coralnet"}))

    with pytest.raises(ValueError, match=r"missing columns \['concept'\]"):
        load_concept_source_mask(path, _schema())


def test_source_ids_sort_numerically_then_alphabetically():
    assert sorted(["12", "7", "b", "a", "100"], key=_source_sort_key) == [
        "7",
        "12",
        "100",
        "a",
        "b",
    ]


def test_non_numeric_source_ids_follow_the_numeric_ones_in_the_keep_table(tmp_path):
    path = tmp_path / "mask.csv"
    pd.DataFrame(
        {"source_id": ["b", "12", "7"], "concept": ["bleached"] * 3, "n_false": [1, 1, 1]}
    ).to_csv(path, index=False)
    meta_path(path).write_text(yaml.safe_dump({"dataset": "coralnet"}))

    mask = load_concept_source_mask(path, _schema())

    assert mask.source_ids == ("7", "12", "b")
    assert mask.row_by_source() == {"7": 1, "12": 2, "b": 3}
