"""Reading a concept-removal file and removing concepts from the concept table."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from mermaidseg.dataset_reconciliation.concept_removal import (
    REMOVAL_COLUMNS,
    apply_concept_removal,
    concept_kind,
    load_concept_removal,
    removed_concepts,
)
from mermaidseg.dataset_reconciliation.concept_source_mask import file_sha1, meta_path
from tests.mermaidseg.conftest import concept_table


def _write_removal(tmp_path, concepts: list[str], *, sha1: str | None = None):
    path = tmp_path / "removal.csv"
    pd.DataFrame({"concept": concepts, "kind": "x"}).to_csv(path, index=False)
    meta = {} if sha1 is None else {"concepts_csv_sha1": sha1}
    meta_path(path).write_text(yaml.safe_dump(meta))
    return path


# ---------- load_concept_removal ----------


def test_load_returns_the_listed_concepts_in_file_order(tmp_path):
    path = _write_removal(tmp_path, ["tubular", "genus__porites", "bleached"])

    assert load_concept_removal(path) == ["tubular", "genus__porites", "bleached"]


def test_load_without_a_sidecar_raises_file_not_found(tmp_path):
    path = tmp_path / "removal.csv"
    pd.DataFrame({"concept": ["tubular"]}).to_csv(path, index=False)

    with pytest.raises(FileNotFoundError, match="no sidecar"):
        load_concept_removal(path)


def test_load_accepts_a_matching_concept_table_sha1_and_refuses_a_changed_table(tmp_path):
    table = tmp_path / "concepts.csv"
    table.write_text("a,b\n1,2\n")
    path = _write_removal(tmp_path, ["tubular"], sha1=file_sha1(table))

    assert load_concept_removal(path, concepts_csv=table) == ["tubular"]

    table.write_text("a,b\n1,3\n")
    with pytest.raises(ValueError, match="sha1"):
        load_concept_removal(path, concepts_csv=table)


def test_load_ignores_the_sha1_when_no_concept_table_is_given(tmp_path):
    path = _write_removal(tmp_path, ["tubular"], sha1="not-a-real-digest")

    assert load_concept_removal(path) == ["tubular"]


def test_load_without_a_concept_column_raises(tmp_path):
    path = tmp_path / "removal.csv"
    pd.DataFrame({"name": ["tubular"]}).to_csv(path, index=False)
    meta_path(path).write_text(yaml.safe_dump({}))

    with pytest.raises(ValueError, match="no 'concept' column"):
        load_concept_removal(path)


# ---------- removed_concepts ----------


def test_removed_concepts_lists_only_removed_rows_with_the_removal_columns():
    coverage = pd.DataFrame(
        {
            "kind": ["genus", "binary", "genus"],
            "n_annotations": [900, 10, 20],
            "n_images": [120, 3, 4],
            "n_noncoralnet_images": [0, 0, 0],
            "n_coralnet_sources": [5, 1, 1],
            "n_other_datasets": [0, 0, 0],
            "fails_images": [False, True, True],
            "fails_annotations": [False, True, True],
            "fails_sources": [False, True, True],
            "removed": [False, True, True],
        },
        index=pd.Index(["genus__acropora", "tubular", "genus__porites"], name="concept"),
    )

    out = removed_concepts(coverage)

    assert list(out.columns) == REMOVAL_COLUMNS
    assert out["concept"].tolist() == ["tubular", "genus__porites"]
    assert out["n_annotations"].tolist() == [10, 20]


# ---------- concept_kind ----------


@pytest.mark.parametrize(
    ("channel", "kind"),
    [
        ("genus__acropora", "genus"),
        ("family__poritidae", "family"),
        ("bleached", "binary"),
        ("not_a_rank__value", "binary"),
    ],
)
def test_concept_kind_is_the_rank_for_taxa_and_binary_otherwise(channel, kind):
    assert concept_kind(channel) == kind


# ---------- apply_concept_removal ----------


def test_removing_a_taxon_sets_only_that_rank_to_not_given():
    table = concept_table()

    out = apply_concept_removal(table, ["genus__acropora"])

    by_label = out.set_index("source_label_class_name")
    assert by_label.loc["acropora", "genus"] == "not_given"
    assert by_label.loc["bleached acropora", "genus"] == "not_given"
    assert by_label.loc["acropora", "family"] == "acroporidae"  # coarser ranks stay
    assert by_label.loc["porites", "genus"] == "porites"  # other values stay
    assert by_label.loc["sand", "genus"] == "none"
    assert table.set_index("source_label_class_name").loc["acropora", "genus"] == "acropora"


def test_removing_a_binary_concept_drops_its_column():
    out = apply_concept_removal(concept_table(), ["tubular"])

    assert "tubular" not in out.columns
    assert {"bleached", "branching"} <= set(out.columns)


def test_a_repeated_concept_is_removed_once():
    out = apply_concept_removal(concept_table(), ["genus__acropora", "genus__acropora"])

    assert (out["genus"] != "acropora").all()


def test_removing_a_binary_concept_without_a_column_raises():
    with pytest.raises(ValueError, match="not a column of the concept table"):
        apply_concept_removal(concept_table().drop(columns="tubular"), ["tubular"])


def test_removing_a_taxon_value_with_no_match_raises():
    with pytest.raises(ValueError, match="no genus == 'nope'"):
        apply_concept_removal(concept_table(), ["genus__nope"])
