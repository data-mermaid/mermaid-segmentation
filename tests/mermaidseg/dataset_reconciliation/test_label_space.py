"""Label space: resolution, lookups, the CSV round trip and the acknowledgement check."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mermaidseg.dataset_reconciliation import label_space as label_space_module
from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.dataset_reconciliation.label_space import (
    LabelSpace,
    check_label_space,
    check_label_space_tables,
    label_keys_sha1,
    read_label_space,
    read_label_space_meta,
    resolve_label_space,
    write_label_space,
)
from segdata_spec.read import VersionTables, open_tables
from segdata_spec.taxonomy import hierarchy_from_table
from tests.mermaidseg.conftest import concept_table
from tests.mermaidseg.data.conftest import VERSION, ImageSpec, PointRow, write_version

_TAXONOMY = pa.table(
    {
        "id": ["1", "3", "4", "5", "6", "7"],
        "name": ["Hard coral", "Acropora", "Acropora cytherea", "Porites", "Sand", "Orphan"],
        "parent_id": [None, "1", "3", "1", None, "99"],
    }
)
_HIERARCHY = hierarchy_from_table(_TAXONOMY)

# Channels of the conftest concept table, in channel order.
_ACROPORA_TRUE = (
    "kingdom__animalia;phylum__cnidaria;class__hexacorallia;order__scleractinia;"
    "family__acroporidae;genus__acropora;branching"
)
_ACROPORA_FALSE = "tubular;bleached"
# Value channels sort alphabetically within each rank, then tubular, branching, bleached.
_ACROPORA_CODES = [2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1]

_NO_ACKS = pd.DataFrame(columns=["dataset", "native_name", "kind", "reason"])


def _label(label_id: int, name: str, attribute: str | None) -> dict[str, Any]:
    return {"label_id": label_id, "name": name, "mermaid_benthic_attribute": attribute}


def _tables(
    tmp_path: Path,
    labels: list[dict[str, Any]],
    *,
    name: str = "coralnet",
    images: Sequence[ImageSpec] = (ImageSpec("i1"),),
    points: Sequence[PointRow] | None = None,
) -> VersionTables:
    """Tables of a points version; by default one image holds one point of the first label."""
    if points is None:
        points = [("i1", 0, 0, labels[0]["label_id"])]
    cache_dir, bucket, pin = write_version(
        tmp_path / name, kind="points", images=images, labels=labels, points=points, name=name
    )
    return open_tables(cache_dir, bucket, pin)


def _schema(
    tmp_path: Path,
    source: str = "coralnet",
    removed: Sequence[str] = (),
    table: pd.DataFrame | None = None,
) -> ConceptSchema:
    csv = tmp_path / f"{source}_concepts.csv"
    (concept_table() if table is None else table).assign(source_dataset_source=source).to_csv(
        csv, index=False
    )
    return ConceptSchema.from_csv(csv, sources={source}, removed_concepts=removed)


def _resolve(
    tmp_path: Path,
    labels: list[dict[str, Any]],
    *,
    subset: Sequence[str],
    roll_up: bool = False,
    name: str = "coralnet",
    table: pd.DataFrame | None = None,
) -> LabelSpace:
    return resolve_label_space(
        {name: _tables(tmp_path, labels, name=name)},
        _TAXONOMY,
        SimpleNamespace(class_subset=list(subset), label_roll_up=roll_up),
        _schema(tmp_path, source=name, table=table),
        train_groups={name: None},
    )


def _row(space: LabelSpace, native_name: str) -> dict[str, Any]:
    rows = space.source_labels[space.source_labels["native_name"] == native_name]
    assert len(rows) == 1
    return rows.iloc[0].to_dict()


# ---------- resolve_label_space ----------


@pytest.mark.parametrize(
    ("attribute", "roll_up", "expected"),
    [
        (None, False, (0, "unmapped", "")),
        ("Porites", False, (0, "not_in_subset", "")),
        ("Mystery", True, (0, "not_in_subset", "")),  # not in the hierarchy
        ("Orphan", True, (0, "no_ancestor_in_subset", "")),  # parent 99 is not in the taxonomy
        ("Sand", True, (2, "", "")),
    ],
    ids=["unmapped", "outside-no-roll-up", "outside-not-in-hierarchy", "no-ancestor", "kept"],
)
def test_each_label_records_why_it_drops(
    tmp_path: Path, attribute: str | None, roll_up: bool, expected: tuple[int, str, str]
) -> None:
    space = _resolve(
        tmp_path, [_label(1, "x", attribute)], subset=["Hard coral", "Sand"], roll_up=roll_up
    )

    row = _row(space, "x")
    assert (row["class_id"], row["drop_reason"], row["rolled_up_to"]) == expected


def test_roll_up_records_the_ancestor_and_keeps_the_original_target(tmp_path: Path) -> None:
    space = _resolve(
        tmp_path,
        [_label(1, "cyth", "Acropora cytherea")],
        subset=["Acropora", "Sand"],
        roll_up=True,
    )

    row = _row(space, "cyth")
    assert (row["target_name"], row["rolled_up_to"], row["class_id"]) == (
        "acropora cytherea",
        "acropora",
        1,
    )


def test_classes_match_the_taxonomy_case_insensitively(tmp_path: Path) -> None:
    space = _resolve(
        tmp_path, [_label(1, "x", "Sand")], subset=["Sand", "acropora", "Background", "sand"]
    )

    assert space.classes.to_dict("records") == [
        {"class_id": 0, "name": "ignore", "kind": "ignore", "mermaid_benthic_attribute_id": ""},
        {"class_id": 1, "name": "acropora", "kind": "mermaid", "mermaid_benthic_attribute_id": "3"},
        {"class_id": 2, "name": "background", "kind": "extra", "mermaid_benthic_attribute_id": ""},
        {"class_id": 3, "name": "sand", "kind": "mermaid", "mermaid_benthic_attribute_id": "6"},
    ]


def test_concepts_list_the_true_and_false_channel_names_in_channel_order(tmp_path: Path) -> None:
    space = _resolve(tmp_path, [_label(1, "acropora", "Acropora")], subset=["Acropora"])

    row = _row(space, "acropora")
    assert (row["concept_status"], row["concepts"], row["false_concepts"]) == (
        "ok",
        _ACROPORA_TRUE,
        _ACROPORA_FALSE,
    )


def test_a_label_without_a_concept_row_is_missing(tmp_path: Path) -> None:
    space = _resolve(tmp_path, [_label(1, "zzz", "Sand")], subset=["Sand"])

    row = _row(space, "zzz")
    assert (row["concept_status"], row["concepts"], row["false_concepts"]) == ("missing", "", "")


def test_n_counts_the_train_images_only(tmp_path: Path) -> None:
    tables = _tables(
        tmp_path,
        [_label(1, "sand", "Sand")],
        images=[ImageSpec("i1", group="g1"), ImageSpec("i2", group="g2")],
        points=[("i1", 0, 0, 1), ("i1", 0, 1, 1), ("i2", 0, 0, 1)],
    )

    space = resolve_label_space(
        {"coralnet": tables},
        _TAXONOMY,
        SimpleNamespace(class_subset=["Sand"], label_roll_up=False),
        _schema(tmp_path),
        train_groups={"coralnet": {"g1"}},
    )

    assert _row(space, "sand")["n"] == 2


def test_n_is_zero_for_a_dataset_without_a_train_split(tmp_path: Path) -> None:
    space = resolve_label_space(
        {"coralnet": _tables(tmp_path, [_label(1, "sand", "Sand")])},
        _TAXONOMY,
        SimpleNamespace(class_subset=["Sand"], label_roll_up=False),
        _schema(tmp_path),
        train_groups={},
    )

    assert _row(space, "sand")["n"] == 0


def test_concepts_table_omits_a_removed_channel(tmp_path: Path) -> None:
    space = resolve_label_space(
        {"coralnet": _tables(tmp_path, [_label(1, "sand", "Sand")])},
        _TAXONOMY,
        SimpleNamespace(class_subset=["Sand"], label_roll_up=False),
        _schema(tmp_path, removed=["tubular"]),
        train_groups={"coralnet": None},
    )

    assert list(space.concepts.itertuples(index=False, name=None)) == [
        (1, "kingdom__animalia", "kingdom"),
        (2, "kingdom__none", "kingdom"),
        (3, "phylum__cnidaria", "phylum"),
        (4, "phylum__none", "phylum"),
        (5, "class__hexacorallia", "class"),
        (6, "class__none", "class"),
        (7, "order__none", "order"),
        (8, "order__scleractinia", "order"),
        (9, "family__acroporidae", "family"),
        (10, "family__none", "family"),
        (11, "family__poritidae", "family"),
        (12, "genus__acropora", "genus"),
        (13, "genus__none", "genus"),
        (14, "genus__porites", "genus"),
        (15, "branching", ""),
        (16, "bleached", ""),
    ]


def _coverage_level(caplog: pytest.LogCaptureFixture, dataset: str) -> tuple[str, str]:
    record = next(r for r in caplog.records if r.getMessage().startswith(f"{dataset}:"))
    return record.levelname, record.getMessage()


def test_coverage_is_logged_per_source_at_info_when_some_labels_map_and_warns_when_none_do(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tables = {
        "coralnet": _tables(tmp_path, [_label(1, "sand", "Sand"), _label(2, "zzz", None)]),
        "mermaid": _tables(tmp_path, [_label(1, "zz1", None)], name="mermaid"),
    }

    with caplog.at_level("INFO", logger=label_space_module.logger.name):
        resolve_label_space(
            tables,
            _TAXONOMY,
            SimpleNamespace(class_subset=["Sand"], label_roll_up=False),
            _schema(tmp_path),
            train_groups={},
        )

    assert _coverage_level(caplog, "coralnet") == (
        "INFO",
        "coralnet: 1 of 2 source labels map to a target class",
    )
    level, message = _coverage_level(caplog, "mermaid")
    assert level == "WARNING"
    assert "check source label keys against the mapping" in message


# ---------- LabelSpace.lookups ----------


def test_source_to_target_follows_each_rows_benthic_attribute(tmp_path: Path) -> None:
    labels = [
        _label(1, "Acro", "Acropora"),
        _label(2, "Sandy", "Sand"),
        _label(3, "Mystery", None),
        _label(4, "Blank", ""),
    ]
    space = _resolve(tmp_path, labels, subset=["Acropora", "Sand"])

    source_to_target, _ = space.lookups({"coralnet": 10})

    assert source_to_target.tolist() == [0] * 11 + [1, 2, 0, 0]


def test_a_gap_id_inside_a_sources_range_maps_to_background(tmp_path: Path) -> None:
    space = _resolve(tmp_path, [_label(1, "a", "a"), _label(3, "b", "b")], subset=["a", "b"])

    source_to_target, _ = space.lookups({"coralnet": 10})

    assert source_to_target.tolist() == [0] * 11 + [1, 0, 2]


def test_mermaid_concept_rows_are_keyed_by_the_benthic_attribute_of_each_ba_gf_label(
    tmp_path: Path,
) -> None:
    labels = [
        _label(1, "Acropora::Branching", "Acropora"),
        _label(2, "Acropora::", "Acropora"),
    ]
    space = _resolve(tmp_path, labels, subset=["Acropora"], name="mermaid")

    _, source_to_concepts = space.lookups({"mermaid": 0})

    assert source_to_concepts.tolist() == [[0] * 17, _ACROPORA_CODES, _ACROPORA_CODES]


def test_a_not_given_binary_concept_is_zero_beside_a_false_one_in_the_same_row(
    tmp_path: Path,
) -> None:
    table = concept_table()
    table.loc[table["source_label_class_name"] == "acropora", "bleached"] = "not_given"
    schema = _schema(tmp_path, table=table)
    space = resolve_label_space(
        {"coralnet": _tables(tmp_path, [_label(1, "acropora", "Acropora")])},
        _TAXONOMY,
        SimpleNamespace(class_subset=["Acropora"], label_roll_up=False),
        schema,
        train_groups={"coralnet": None},
    )

    _, source_to_concepts = space.lookups({"coralnet": 0})

    # Channels 14..16 are tubular (FALSE), branching (TRUE), bleached (not_given).
    assert source_to_concepts[1, 14:].tolist() == [1, 2, 0]


def test_a_rank_value_removed_to_not_given_leaves_that_ranks_channels_zero(
    tmp_path: Path,
) -> None:
    space = resolve_label_space(
        {"coralnet": _tables(tmp_path, [_label(1, "acropora", "Acropora")])},
        _TAXONOMY,
        SimpleNamespace(class_subset=["Acropora"], label_roll_up=False),
        _schema(tmp_path, removed=["genus__acropora"]),
        train_groups={"coralnet": None},
    )

    _, source_to_concepts = space.lookups({"coralnet": 0})

    # Genus channels are 11 (genus__none) and 12 (genus__porites); the other ranks keep their codes.
    assert source_to_concepts[1].tolist() == [2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 0, 0, 1, 2, 1]


# ---------- write_label_space / read_label_space ----------


def _round_trip_space(tmp_path: Path) -> LabelSpace:
    labels = [
        _label(1, "acropora", "Acropora cytherea"),
        _label(2, "zzz", None),
        _label(3, "sand", "Sand"),
    ]
    space = _resolve(tmp_path, labels, subset=["Acropora", "Sand", "Background"], roll_up=True)
    return LabelSpace(
        classes=space.classes,
        concepts=space.concepts,
        source_labels=space.source_labels,
        meta={
            "name": "tiny",
            "concepts_csv_sha1": "abc",
            "datasets": {"coralnet": {"version": VERSION}},
        },
    )


def test_label_space_round_trips_through_its_directory(tmp_path: Path) -> None:
    space = _round_trip_space(tmp_path)

    write_label_space(tmp_path / "space", space)
    read = read_label_space(tmp_path / "space")

    pd.testing.assert_frame_equal(read.classes, space.classes)
    pd.testing.assert_frame_equal(read.concepts, space.concepts)
    pd.testing.assert_frame_equal(read.source_labels, space.source_labels)
    assert read.meta == space.meta
    assert _row(read, "zzz")["target_name"] == ""


_ACK_BYTES = b"dataset,native_name,kind,reason\r\nx,y,z,w\r\n"


def test_rewriting_a_directory_replaces_every_table_and_keeps_acknowledged_csv(
    tmp_path: Path,
) -> None:
    space = _round_trip_space(tmp_path)
    directory = tmp_path / "space"
    directory.mkdir()
    for name in ("classes.csv", "concepts.csv", "source_labels.csv", "label_space.meta.yaml"):
        (directory / name).write_text("stale")
    (directory / "acknowledged.csv").write_bytes(_ACK_BYTES)

    write_label_space(directory, space)

    assert sorted(p.name for p in directory.iterdir()) == [
        "acknowledged.csv",
        "classes.csv",
        "concepts.csv",
        "label_space.meta.yaml",
        "source_labels.csv",
    ]
    assert (directory / "acknowledged.csv").read_bytes() == _ACK_BYTES
    read = read_label_space(directory)
    pd.testing.assert_frame_equal(read.classes, space.classes)
    pd.testing.assert_frame_equal(read.source_labels, space.source_labels)


def _swap_class_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(class_id=[1, 0, 2, 3])


def _gap_in_class_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(class_id=[0, 1, 2, 4])


def _reordered_channel_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(channel_id=list(reversed(df["channel_id"])))


def _second_version(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(version=["20990101-abcdef0", *df["version"][1:]])


def _duplicate_row(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _unknown_class(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(class_id=[99, *df["class_id"][1:]])


def _unknown_true_concept(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(concepts=["genus__zzz", *df["concepts"][1:]])


def _unknown_false_concept(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(false_concepts=["zzz", *df["false_concepts"][1:]])


def test_check_label_space_tables_accepts_a_valid_space(tmp_path: Path) -> None:
    check_label_space_tables(_round_trip_space(tmp_path))


@pytest.mark.parametrize(
    ("table", "edit", "match"),
    [
        ("classes", _swap_class_ids, r"classes\.csv class_id"),
        ("classes", _gap_in_class_ids, r"classes\.csv class_id"),
        ("concepts", _reordered_channel_ids, r"concepts\.csv channel_id"),
        ("source_labels", _second_version, r"coralnet.*" + VERSION + r".*20990101-abcdef0"),
        ("source_labels", _duplicate_row, r"source_labels\.csv repeats"),
        ("source_labels", _unknown_class, r"source_labels\.csv names class ids"),
        ("source_labels", _unknown_true_concept, r"source_labels\.csv names concepts"),
        ("source_labels", _unknown_false_concept, r"source_labels\.csv names concepts"),
    ],
    ids=[
        "class-swapped",
        "class-gap",
        "channels-reordered",
        "two-versions",
        "duplicate-label",
        "unknown-class",
        "unknown-concept",
        "unknown-false-concept",
    ],
)
def test_check_label_space_tables_refuses_an_inconsistent_table_naming_the_file(
    tmp_path: Path, table: str, edit: Callable[[pd.DataFrame], pd.DataFrame], match: str
) -> None:
    space = _round_trip_space(tmp_path)
    broken = dataclasses.replace(space, **{table: edit(getattr(space, table))})

    with pytest.raises(ValueError, match=match):
        check_label_space_tables(broken)


def test_read_label_space_meta_returns_the_meta(tmp_path: Path) -> None:
    directory = tmp_path / "space"
    directory.mkdir()
    (directory / "label_space.meta.yaml").write_text("concepts_csv_sha1: abc\nname: tiny\n")

    assert read_label_space_meta(directory) == {"concepts_csv_sha1": "abc", "name": "tiny"}


# ---------- check_label_space ----------


def _acks(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(*row, "reviewed") for row in rows], columns=["dataset", "native_name", "kind", "reason"]
    )


def test_an_unacknowledged_missing_concept_row_raises(tmp_path: Path) -> None:
    space = _resolve(tmp_path, [_label(1, "zzz", "Sand")], subset=["Sand"])

    with pytest.raises(ValueError, match=r"coralnet.*zzz.*missing_concepts"):
        check_label_space(space, _NO_ACKS, _HIERARCHY)


def test_an_acknowledged_missing_concept_row_passes(tmp_path: Path) -> None:
    space = _resolve(tmp_path, [_label(1, "zzz", "Sand")], subset=["Sand"])

    check_label_space(space, _acks(("coralnet", "zzz", "missing_concepts")), _HIERARCHY)


@pytest.mark.parametrize(
    ("kind", "reason"),
    [("typo", "reviewed"), ("missing_concepts", "")],
    ids=["unknown-kind", "empty-reason"],
)
def test_a_malformed_acknowledgement_raises(tmp_path: Path, kind: str, reason: str) -> None:
    space = _resolve(tmp_path, [_label(1, "zzz", "Sand")], subset=["Sand"])
    acks = pd.DataFrame(
        [("coralnet", "zzz", kind, reason)], columns=["dataset", "native_name", "kind", "reason"]
    )

    with pytest.raises(ValueError, match="acknowledge"):
        check_label_space(space, acks, _HIERARCHY)


def _ranked_space(
    tmp_path: Path, target: str, ranks: dict[str, str], subset: Sequence[str] | None = None
) -> LabelSpace:
    """A label "acropora" mapped to ``target`` whose concept row sets the given rank values.

    ``subset`` defaults to ``[target]``, so the label is kept.
    """
    table = concept_table()
    for rank, value in ranks.items():
        table.loc[table["source_label_class_name"] == "acropora", rank] = value
    return _resolve(
        tmp_path,
        [_label(1, "acropora", target)],
        subset=[target] if subset is None else subset,
        table=table,
    )


_OTHER_ROOT = {"soft coral": None, "xeniidae": "soft coral", "other": None, "dead coral": "other"}
_SAME_ROOT = {"hard coral": None, "agariciidae": "hard coral", "pachyseris": "hard coral"}
_DESCENDANT = {"sponge": None, "porifera": "sponge"}
_ORDER_OTHER_ROOT = {
    "hard coral": None,
    "scleractinia": "hard coral",
    "other": None,
    "dead coral": "other",
}


@pytest.mark.parametrize(
    ("ranks", "detail", "subset"),
    [
        ({"family": "xeniidae"}, "family 'xeniidae'", None),
        ({"genus": "Xeniidae"}, "genus 'xeniidae'", None),
        # The target is outside the subset, so the label drops; its concept row is still checked.
        ({"family": "xeniidae"}, "family 'xeniidae'", ["Sand"]),
    ],
    ids=["family", "genus-case-insensitive", "dropped-label"],
)
def test_a_value_under_another_root_than_the_target_raises(
    tmp_path: Path, ranks: dict[str, str], detail: str, subset: list[str] | None
) -> None:
    space = _ranked_space(tmp_path, "Dead coral", ranks, subset)

    with pytest.raises(ValueError, match=rf"coralnet.*acropora.*rank_disagreement.*{detail}"):
        check_label_space(space, _NO_ACKS, _OTHER_ROOT)


@pytest.mark.parametrize(
    ("target", "ranks", "hierarchy"),
    [
        # acropora, acroporidae ... name no MERMAID attribute.
        ("Acropora", {}, _HIERARCHY),
        # genus acropora is the target's parent.
        ("Acropora cytherea", {}, _HIERARCHY),
        # agariciidae is off the target's lineage but under the same root.
        ("Pachyseris", {"family": "agariciidae"}, _SAME_ROOT),
        # order is not a checked rank, whatever root scleractinia is under.
        ("Dead coral", {"order": "scleractinia"}, _ORDER_OTHER_ROOT),
        ("Sponge", {"family": "porifera"}, _DESCENDANT),
        # a target outside the hierarchy has no root to disagree with.
        ("Mystery", {"family": "xeniidae"}, _OTHER_ROOT),
    ],
    ids=[
        "target-and-unlisted-ranks",
        "ancestor",
        "same-root",
        "unchecked-rank",
        "descendant",
        "target-not-in-hierarchy",
    ],
)
def test_ranks_on_the_targets_line_or_root_or_unchecked_pass(
    tmp_path: Path, target: str, ranks: dict[str, str], hierarchy: dict[str, str | None]
) -> None:
    space = _ranked_space(tmp_path, target, ranks)

    check_label_space(space, _NO_ACKS, hierarchy)


def test_drops_to_background_do_not_raise(tmp_path: Path) -> None:
    labels = [
        _label(1, "acropora", None),
        _label(2, "porites", "Porites"),
        _label(3, "zzz", None),  # dropped, and no concept row
    ]
    space = _resolve(tmp_path, labels, subset=["Sand"])

    check_label_space(space, _NO_ACKS, _HIERARCHY)


def test_every_unacknowledged_case_is_reported_in_one_error(tmp_path: Path) -> None:
    table = concept_table()
    table.loc[table["source_label_class_name"] == "acropora", "family"] = "xeniidae"
    labels = [_label(1, "zzz", "Sand"), _label(2, "acropora", "Dead coral")]
    space = _resolve(tmp_path, labels, subset=["Dead coral", "Sand"], table=table)

    with pytest.raises(ValueError) as error:
        check_label_space(space, _NO_ACKS, _OTHER_ROOT)

    assert "missing_concepts" in str(error.value)
    assert "rank_disagreement" in str(error.value)


@pytest.mark.parametrize(
    ("native_name", "attribute"),
    [
        ("ghost", None),  # no such label
        ("zzz", None),  # unmapped: drops, so it is no case
        ("zzz", "Porites"),  # not in the subset: drops, so it is no case
    ],
    ids=["unknown-label", "dropped-unmapped", "dropped-not-in-subset"],
)
def test_a_stale_acknowledgement_raises(
    tmp_path: Path, native_name: str, attribute: str | None
) -> None:
    labels = [_label(1, "sand", "Sand"), _label(2, "zzz", attribute)]
    space = _resolve(tmp_path, labels, subset=["Sand"])

    with pytest.raises(ValueError, match=rf"stale.*{native_name}"):
        check_label_space(space, _acks(("coralnet", native_name, "missing_concepts")), _HIERARCHY)


_KEY_CONFIG = {
    "class_subset": ["Acropora", "Sand"],
    "label_roll_up": True,
    "concept_mapping_path": "configs/class_to_concepts.csv",
    "concept_removal": "",
    "concept_source_mask": "",
}


@pytest.mark.parametrize(
    ("other", "same"),
    [
        ({**_KEY_CONFIG, "class_subset": ["Acropora", "Porites"]}, False),
        ({**_KEY_CONFIG, "label_roll_up": False}, False),
        ({**_KEY_CONFIG, "concept_mapping_path": "configs/other.csv"}, False),
        ({**_KEY_CONFIG, "concept_removal": "configs/r.csv"}, False),
        ({**_KEY_CONFIG, "concept_source_mask": "configs/m.csv"}, False),
        ({**_KEY_CONFIG, "epochs": 99}, True),
    ],
    ids=[
        "class_subset",
        "label_roll_up",
        "concept_mapping_path",
        "concept_removal",
        "concept_source_mask",
        "unrelated_key",
    ],
)
def test_label_keys_sha1_follows_the_label_keys_only(other: dict[str, Any], same: bool) -> None:
    assert (label_keys_sha1(_KEY_CONFIG) == label_keys_sha1(other)) is same
