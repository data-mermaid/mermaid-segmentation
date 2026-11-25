"""GT mapping: concept and class lookups built from the concept table."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mermaidseg.data.point_dataset import PointDataset
from mermaidseg.evaluation import gt_mapping
from mermaidseg.evaluation.gt_mapping import (
    ConceptLookup,
    build_class_lookup,
    build_concept_lookup,
    point_scopes,
)
from segdata_spec.taxonomy import hierarchy_from_table
from tests.mermaidseg.data.conftest import ImageSpec, open_written

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CSV = _REPO_ROOT / "configs" / "class_to_concepts.csv"


def test_concept_lookup_matches_schema():
    """build_concept_lookup must reproduce ConceptSchema encoding exactly."""
    from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
    from mermaidseg.evaluation.gt_mapping import build_concept_lookup

    sources = {"coralscapes_v2", "pacific_labeled_corals", "benthos_yuval"}
    schema = ConceptSchema.from_csv(_CSV, sources=sources)
    names = list(schema.channel_names)
    import pandas as pd

    df = pd.read_csv(_CSV)
    df["source_label_class_name"] = df["source_label_class_name"].astype(str).str.lower()
    df["source_dataset_source"] = df["source_dataset_source"].astype(str).str.lower()
    checked = 0
    for src in sources:
        labels = df[df["source_dataset_source"] == src]["source_label_class_name"].unique().tolist()
        lk = build_concept_lookup(src, labels, names, _CSV)
        for lab in labels:
            ref = schema.row_for(src, lab).astype(np.int8)
            got = lk.row(lab)
            assert np.array_equal(ref, got), (src, lab)
            checked += 1
    assert checked > 100


def _reader(tmp_path: Path) -> PointDataset:
    """A points version whose labels.parquet maps two labels and leaves one unmapped."""
    version = open_written(
        tmp_path,
        kind="points",
        images=[ImageSpec("i1")],
        labels=[
            {"label_id": 1, "name": "ACRO", "mermaid_benthic_attribute": "Acropora"},
            {"label_id": 2, "name": "Sand", "mermaid_benthic_attribute": "Sand"},
            {"label_id": 3, "name": "Shadow", "mermaid_benthic_attribute": None},
        ],
        points=[("i1", 0, 0, 1), ("i1", 1, 1, 2), ("i1", 2, 2, 3)],
    )
    return PointDataset(version, split="val")


# Hard coral (1) <- Acroporidae (2) <- Acropora (3); Sand (4) is a root.
_TAXONOMY = pa.table(
    {
        "id": ["1", "2", "3", "4"],
        "name": ["Hard coral", "Acroporidae", "Acropora", "Sand"],
        "parent_id": [None, "1", "2", None],
    }
)


def test_a_reader_label_maps_to_its_labels_parquet_target(tmp_path):
    reader = _reader(tmp_path)
    id2label = {0: "ignore", 1: "acropora", 2: "sand"}

    lookup = build_class_lookup(reader.label_targets, ["acro", "sand", "shadow"], id2label)

    assert lookup.name_to_class_id == {"acro": 1, "sand": 2, "shadow": 0}


def test_with_roll_up_a_species_maps_to_its_class_subset_ancestor_in_the_taxonomy(tmp_path):
    reader = _reader(tmp_path)
    id2label = {0: "ignore", 1: "hard coral", 2: "sand"}

    lookup = build_class_lookup(
        reader.label_targets, ["acro", "sand"], id2label, hierarchy_from_table(_TAXONOMY)
    )

    assert lookup.name_to_class_id == {"acro": 1, "sand": 2}


def test_without_a_hierarchy_a_target_outside_the_vocabulary_is_ignored():
    lookup = build_class_lookup(
        {"ACRO": "Acropora", "hc": "Hard Coral", "gone": None},
        ["acro", "hc", "gone", "absent"],
        {0: "ignore", 1: "Hard Coral"},
    )
    assert lookup.name_to_class_id == {"acro": 0, "hc": 1, "gone": 0, "absent": 0}


@pytest.mark.parametrize(
    ("label", "live", "coral"),
    [
        ("porites", True, True),
        ("dead pocillopora", False, False),
        ("a. galapagensis (alive)", True, True),
        ("millepora", True, True),
        ("cred-octocoral", True, True),
        ("palythoa", True, False),
        ("macroalgae", True, False),  # live column ignored outside corals / echinoderms
    ],
)
def test_point_scopes_from_the_concept_table(label, live, coral):
    lookup = build_concept_lookup("coralnet", [label], [], _CSV)
    assert lookup.scopes(label) == (live, coral)


def test_concept_lookup_unmapped_label_gives_zeros_and_default_scopes():
    mapped = np.array([2, 1, 0], dtype=np.int8)
    lookup = ConceptLookup({"coral": mapped}, 3, {"coral": (False, True)})
    assert lookup.row("CORAL").tolist() == [2, 1, 0]
    assert lookup.unmapped == set()
    assert lookup.scopes("CORAL") == (False, True)

    assert lookup.row("Mystery").tolist() == [0, 0, 0]
    assert lookup.row("Mystery").dtype == np.int8
    assert lookup.unmapped == {"mystery"}
    assert lookup.scopes("Mystery") == (True, False)


_CONCEPT_NAMES = ["kingdom__animalia", "kingdom__plantae", "live"]


def _write_concept_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_first_row_wins_for_a_duplicated_label_and_other_sources_are_ignored(tmp_path):
    csv_path = _write_concept_csv(
        tmp_path / "c.csv",
        [
            {"source_dataset_source": "src", "source_label_class_name": "Foo",
             "kingdom": "animalia", "live": "true"},
            {"source_dataset_source": "src", "source_label_class_name": "foo",
             "kingdom": "plantae", "live": "false"},
            {"source_dataset_source": "other", "source_label_class_name": "bar",
             "kingdom": "plantae", "live": "true"},
        ],
    )  # fmt: skip
    lookup = build_concept_lookup("SRC", ["FOO", "bar"], _CONCEPT_NAMES, csv_path)
    assert lookup.row("foo").tolist() == [2, 1, 2]
    assert lookup.unmapped == set()
    assert lookup.row("bar").tolist() == [0, 0, 0]
    assert lookup.unmapped == {"bar"}


def test_table_without_a_live_column_gives_default_scopes_and_warns(tmp_path, caplog):
    coral_row = {
        "source_dataset_source": "src",
        "source_label_class_name": "acro",
        "kingdom": "animalia",
        "order": "scleractinia",
    }
    without = _write_concept_csv(tmp_path / "without.csv", [coral_row])
    with caplog.at_level(logging.WARNING, logger=gt_mapping.logger.name):
        lookup = build_concept_lookup("src", ["acro"], _CONCEPT_NAMES, without)
    assert "no 'live' column" in caplog.text
    assert lookup.name_to_scope == {}
    assert lookup.scopes("acro") == (True, False)

    with_live = _write_concept_csv(tmp_path / "with.csv", [{**coral_row, "live": "true"}])
    caplog.clear()
    lookup = build_concept_lookup("src", ["acro"], _CONCEPT_NAMES, with_live)
    assert "no 'live' column" not in caplog.text
    assert lookup.scopes("acro") == (True, True)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"phylum": "echinodermata", "live": "false"}, (False, False)),
        ({"phylum": "echinodermata", "live": "true"}, (True, False)),
        ({"phylum": "mollusca", "live": "false"}, (True, False)),
        ({"phylum": "cnidaria", "live": "false", "order": "scleractinia"}, (False, False)),
        ({"phylum": "cnidaria", "live": "true", "class": "octocorallia"}, (True, True)),
        ({"phylum": "cnidaria", "live": "true", "family": "stylasteridae"}, (True, True)),
        ({"phylum": "cnidaria", "live": "true", "order": "actiniaria"}, (True, False)),
    ],
)
def test_point_scopes_live_column_applies_to_cnidaria_and_echinoderms_only(row, expected):
    assert point_scopes(pd.Series(row)) == expected
