"""GT mapping: concept and class lookups built from the concept table."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mermaidseg.evaluation import gt_mapping
from mermaidseg.evaluation.gt_mapping import (
    ConceptLookup,
    build_class_lookup,
    build_concept_lookup,
    load_benthic_hierarchy,
    point_scopes,
)

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


def test_class_lookup_direct_and_rollup():
    from mermaidseg.evaluation.gt_mapping import build_class_lookup

    id2label = {0: "ignore", 1: "acropora", 2: "hard coral", 3: "background", 4: "sand"}
    # Pacific: 'acropora'->'acropora' (direct); 'other scleractinians'->'hard coral' (direct);
    # 'unclear'->'background'; unknown label -> 0.
    hierarchy = {"acropora": "hard coral", "hard coral": None, "background": None}
    names = ["acropora", "other scleractinians", "unclear", "sand", "totally unknown label"]
    lk = build_class_lookup("pacific_labeled_corals", names, id2label, hierarchy)
    assert lk.class_id("acropora") == 1
    assert lk.class_id("other scleractinians") == 2
    assert lk.class_id("unclear") == 3
    assert lk.class_id("sand") == 4
    assert lk.class_id("totally unknown label") == 0


def test_class_lookup_rollup_to_parent():
    from mermaidseg.evaluation.gt_mapping import build_class_lookup

    # 'acropora' not in id2label -> roll up to parent 'hard coral' which is.
    id2label = {0: "ignore", 1: "hard coral"}
    hierarchy = {"acropora": "hard coral", "hard coral": None}
    lk = build_class_lookup("pacific_labeled_corals", ["acropora"], id2label, hierarchy)
    assert lk.class_id("acropora") == 1


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


def _no_fetch(monkeypatch):
    """Make a remote hierarchy fetch fail loudly if a test reaches it."""

    def fail():
        raise AssertionError("hierarchy fetch must not be reached")

    monkeypatch.setattr(gt_mapping, "initialize_benthic_hierarchy", fail)


def test_hierarchy_file_wins_over_cache_and_lowercases(tmp_path, monkeypatch):
    _no_fetch(monkeypatch)
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"Acropora": "Hard Coral", "Hard Coral": None, "Sand": ""}))
    (tmp_path / "benthic_hierarchy.json").write_text(json.dumps({"cached": None}))
    got = load_benthic_hierarchy(explicit, output_dir=tmp_path)
    assert got == {"acropora": "hard coral", "hard coral": None, "sand": None}


def test_hierarchy_cache_hit_skips_the_fetch(tmp_path, monkeypatch):
    _no_fetch(monkeypatch)
    (tmp_path / "benthic_hierarchy.json").write_text(json.dumps({"Porites": "Hard Coral"}))
    assert load_benthic_hierarchy(output_dir=tmp_path) == {"porites": "hard coral"}


def test_hierarchy_is_none_offline_without_a_cache(tmp_path, monkeypatch, caplog):
    _no_fetch(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=gt_mapping.logger.name):
        assert load_benthic_hierarchy(output_dir=tmp_path, fetch_remote=False) is None
    assert "roll-up OFF" in caplog.text


def test_hierarchy_is_none_when_the_fetch_fails(tmp_path, monkeypatch, caplog):
    def boom():
        raise ConnectionError("no network")

    monkeypatch.setattr(gt_mapping, "initialize_benthic_hierarchy", boom)
    with caplog.at_level(logging.WARNING, logger=gt_mapping.logger.name):
        assert load_benthic_hierarchy(output_dir=tmp_path) is None
    assert "no network" in caplog.text
    assert not (tmp_path / "benthic_hierarchy.json").exists()


def test_hierarchy_fetch_is_lowercased_and_cached_for_the_next_call(tmp_path, monkeypatch):
    calls = []

    def fetch():
        calls.append(1)
        return {"Acropora": "Hard Coral", "Hard Coral": None}

    monkeypatch.setattr(gt_mapping, "initialize_benthic_hierarchy", fetch)
    out = tmp_path / "nested" / "dir"
    expected = {"acropora": "hard coral", "hard coral": None}
    assert load_benthic_hierarchy(output_dir=out) == expected
    assert json.loads((out / "benthic_hierarchy.json").read_text()) == expected
    assert load_benthic_hierarchy(output_dir=out) == expected
    assert len(calls) == 1


def test_hierarchy_fetch_without_output_dir_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(gt_mapping, "initialize_benthic_hierarchy", lambda: {"A": None})
    monkeypatch.chdir(tmp_path)
    assert load_benthic_hierarchy() == {"a": None}
    assert list(tmp_path.iterdir()) == []


def test_hierarchy_survives_an_unwritable_cache(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gt_mapping, "initialize_benthic_hierarchy", lambda: {"A": None})
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with caplog.at_level(logging.WARNING, logger=gt_mapping.logger.name):
        assert load_benthic_hierarchy(output_dir=blocker / "sub") == {"a": None}
    assert "Failed to cache" in caplog.text


def test_source_to_target_map_rejects_an_unwired_source():
    with pytest.raises(ValueError, match="'unknown_source'"):
        gt_mapping._source_to_target_map("unknown_source")


def test_source_to_target_map_dispatches_each_source_and_lowercases(monkeypatch):
    fetchers = {
        "coralnet": "fetch_coralnet_to_mermaid",
        "pacific_labeled_corals": "fetch_pacific_labeled_corals_to_mermaid",
        "benthos_yuval": "fetch_benthos_yuval_to_mermaid",
        "catlin_seaview": "fetch_catlin_seaview_to_mermaid",
        "moorea_labeled_corals": "fetch_moorea_labeled_corals_to_mermaid",
        "ucsd_mosaics": "fetch_ucsd_mosaics_to_mermaid",
    }
    for source, attr in fetchers.items():
        monkeypatch.setattr(
            gt_mapping, attr, lambda source=source: {"Label": source.upper(), "X": None}
        )
    for source in fetchers:
        # The source name itself is matched case-insensitively.
        assert gt_mapping._source_to_target_map(source.title()) == {"label": source, "x": None}


def test_class_lookup_without_hierarchy_maps_only_labels_already_in_the_vocabulary(monkeypatch):
    monkeypatch.setattr(
        gt_mapping,
        "_source_to_target_map",
        lambda _source: {"acro": "acropora", "hc": "hard coral", "gone": None, "sand": "sand"},
    )
    id2label = {0: "ignore", 1: "Hard Coral", 2: "sand"}
    # With no hierarchy 'acropora' cannot roll up to 'hard coral' and is ignored.
    lookup = build_class_lookup("any", ["ACRO", "hc", "gone", "sand", "absent"], id2label, None)
    assert lookup.name_to_class_id == {"acro": 0, "hc": 1, "gone": 0, "sand": 2, "absent": 0}


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
