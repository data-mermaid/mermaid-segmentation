"""Phylum-level relabelling of quantile-flagged CoralNet sources. No S3 access."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mermaidseg.datasets.coralnet.coralnet_dataset import _CORALNET_ID2NAME_KEY, CoralNetDataset
from mermaidseg.datasets.local_cache import LocalS3Cache
from scripts.build_coralnet_phylum_manifest import (
    ALREADY_COARSE,
    CNIDARIA_LABELS,
    COARSE_LABELS,
    COLLAPSED,
    NO_COARSE_LABEL,
    NON_TAXONOMIC,
    TAXON_RANKS,
    apply_train_dedup_filters,
    collapse_manifest,
    label_changes,
    load_concept_rows,
    name_to_id,
    points_without_concepts,
    replacement_table,
)
from scripts.coralnet_source_report import DEFAULT_CONCEPTS_CSV

ID2NAME = {
    "1": "Acropora",
    "2": "Agelas",
    "3": "Sand",
    "4": "Coral",
    "5": "Sponge",
    "6": "Not In CSV",
    "7": "coral",
}
REPLACEMENTS = {"acropora": "coral", "agelas": "sponge"}


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    LocalS3Cache.reset_singleton()
    yield
    LocalS3Cache.reset_singleton()


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return replacement_table(load_concept_rows(DEFAULT_CONCEPTS_CSV))


def _manifest(ids: list) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "source_id": [10, 10, 10, 20, 20, 20][:n],
            "image_id": [100, 100, 101, 200, 200, 201][:n],
            "row": list(range(n)),
            "col": list(range(n)),
            "coralnet_id": ids,
            "image_s3_key": [f"img/{i}.jpg" for i in [100, 100, 101, 200, 200, 201][:n]],
        }
    )


def _concepts(rows: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Rows shaped like ``load_concept_rows``: every rank and state column, lowercased."""
    columns = [*TAXON_RANKS, "live", "bleached"]
    frame = pd.DataFrame(
        [{c: values.get(c, "") for c in columns} for values in rows.values()], columns=columns
    )
    frame.index = pd.Index(list(rows), name="label")
    return frame


def _target_rows() -> dict[str, dict[str, str]]:
    rows = {label: {"phylum": "cnidaria"} for label in CNIDARIA_LABELS.values()}
    for (rank, taxon), label in COARSE_LABELS.items():
        rows[label] = {rank: taxon}
    return rows


@pytest.mark.parametrize(
    ("label", "reason", "replacement"),
    [
        ("acropora", COLLAPSED, "coral"),
        ("millepora", COLLAPSED, "coral"),
        ("soft coral", COLLAPSED, "coral"),
        ("dead coral", COLLAPSED, "dead"),
        ("bleached acropora", COLLAPSED, "condition_fully bleached"),
        ("agelas", COLLAPSED, "sponge"),
        ("aiolochroia crassa", COLLAPSED, "sponge"),
        ("crustose coralline algae", COLLAPSED, "red algae"),
        ("dictyota", COLLAPSED, "brown algae, unidentified"),
        ("halimeda", COLLAPSED, "green algae, unidentified"),
        ("thalassia", COLLAPSED, "seagrass"),
        ("coral", ALREADY_COARSE, None),
        ("sponge", ALREADY_COARSE, None),
        ("seagrass", ALREADY_COARSE, None),
        ("sand", NON_TAXONOMIC, None),
        ("turf algae", NON_TAXONOMIC, None),
        ("chondrilla spp.", NO_COARSE_LABEL, None),
    ],
)
def test_real_csv_replacements(table, label, reason, replacement):
    assert table.loc[label, "reason"] == reason
    assert table.loc[label, "replacement"] == replacement or (
        replacement is None and pd.isna(table.loc[label, "replacement"])
    )


def test_real_csv_targets_are_never_collapsed(table):
    targets = set(table["replacement"].dropna())
    assert targets <= set(table.index)
    assert not (table.loc[sorted(targets), "reason"] == COLLAPSED).any()


def test_cnidaria_condition_precedence():
    rows = _target_rows() | {
        "dead bleached acro": {
            "phylum": "cnidaria",
            "genus": "acropora",
            "live": "false",
            "bleached": "true",
        },
        "bleached acro": {"phylum": "cnidaria", "genus": "acropora", "bleached": "true"},
        "acro": {"phylum": "cnidaria", "genus": "acropora", "live": "true"},
    }
    table = replacement_table(_concepts(rows))
    assert table.loc["dead bleached acro", "replacement"] == "dead"
    assert table.loc["bleached acro", "replacement"] == "condition_fully bleached"
    assert table.loc["acro", "replacement"] == "coral"


def test_missing_target_row_raises():
    rows = _target_rows()
    del rows["sponge"]
    with pytest.raises(ValueError, match="no CoralNet row"):
        replacement_table(_concepts(rows))


def test_target_that_would_collapse_raises():
    rows = _target_rows()
    rows["sponge"] = {"phylum": "porifera", "genus": "agelas"}
    with pytest.raises(ValueError, match="would themselves be collapsed"):
        replacement_table(_concepts(rows))


def test_detailed_label_without_coarse_label_raises():
    rows = _target_rows() | {"mystery worm": {"phylum": "nematoda", "genus": "x"}}
    with pytest.raises(ValueError, match="no coarse label"):
        replacement_table(_concepts(rows))


def test_name_to_id_lowercases_and_keeps_smallest_id():
    assert name_to_id(ID2NAME)["coral"] == 4
    assert name_to_id(ID2NAME)["acropora"] == 1


@pytest.mark.parametrize(
    "ids",
    [
        [1, 2, 3, 1, 2, 6],
        ["1", "2", "3", "1", "2", "6"],
        pd.Categorical([1, 2, 3, 1, 2, 6]),
    ],
    ids=["int", "str", "categorical"],
)
def test_collapse_only_rewrites_flagged_sources(ids):
    manifest = _manifest(ids)
    before = manifest.copy(deep=True)

    out, changed = collapse_manifest(manifest, ID2NAME, REPLACEMENTS, flagged_sources=[20])

    pd.testing.assert_frame_equal(manifest, before)
    expected = [1, 2, 3, 4, 5, 6]
    assert [int(x) for x in out["coralnet_id"]] == expected
    assert [type(x) for x in out["coralnet_id"]] == [type(x) for x in before["coralnet_id"]]
    pd.testing.assert_series_equal(
        out["original_coralnet_id"], before["coralnet_id"], check_names=False
    )
    pd.testing.assert_frame_equal(
        out.drop(columns=["coralnet_id", "original_coralnet_id"]),
        before.drop(columns=["coralnet_id"]),
    )
    assert changed.to_dict("records") == [
        {"source_id": 20, "label": "acropora", "replacement": "coral"},
        {"source_id": 20, "label": "agelas", "replacement": "sponge"},
    ]


def test_collapse_rejects_replacement_without_id():
    with pytest.raises(ValueError, match="not in coralnet_id2name"):
        collapse_manifest(_manifest([1]), ID2NAME, {"acropora": "rock"}, flagged_sources=[10])


def test_label_changes_and_points_without_concepts(table):
    manifest = _manifest([1, 2, 3, 1, 2, 6])
    out, changed = collapse_manifest(manifest, ID2NAME, REPLACEMENTS, flagged_sources=[10, 20])
    changes = label_changes(changed, table)
    assert changes[["label", "replacement", "n_points", "n_sources"]].to_dict("records") == [
        {"label": "acropora", "replacement": "coral", "n_points": 2, "n_sources": 2},
        {"label": "agelas", "replacement": "sponge", "n_points": 2, "n_sources": 2},
    ]
    assert set(changes["phylum"]) == {"cnidaria", "porifera"}
    assert points_without_concepts(manifest, ID2NAME, table.index, [20]) == 1
    assert points_without_concepts(manifest, ID2NAME, table.index, [10]) == 0


def test_train_dedup_filters_replace_points_then_drop_blacklisted_images(tmp_path):
    bucket = "test-bucket"
    cache = LocalS3Cache.configure(tmp_path / "cache", offline=True)
    id2name_path = cache.local_path(bucket, _CORALNET_ID2NAME_KEY)
    id2name_path.parent.mkdir(parents=True)
    id2name_path.write_text(json.dumps(ID2NAME))

    manifest = _manifest([1, 2, 3, 1, 2, 6])
    overrides = tmp_path / "overrides.parquet"
    pd.DataFrame(
        {
            "image_id": [100, 100],
            "row": [0, 4],
            "col": [0, 4],
            "coralnet_id": [3, 1],
        }
    ).to_parquet(overrides, index=False)
    blacklist = tmp_path / "blacklist.parquet"
    pd.DataFrame({"image_id": [101]}).to_parquet(blacklist, index=False)
    config = tmp_path / "data.yaml"
    config.write_text(
        "data:\n"
        "  coralnet:\n"
        "    train:\n"
        f"      annotation_overrides: {overrides}\n"
        f"      image_blacklist: {blacklist}\n"
    )

    filtered = apply_train_dedup_filters(manifest, config, bucket)

    image_100 = filtered[filtered["image_id"] == 100].sort_values("row")
    assert image_100["coralnet_id"].tolist() == [3, 1]
    assert image_100["image_s3_key"].tolist() == ["img/100.jpg", "img/100.jpg"]
    assert 101 not in set(filtered["image_id"])
    assert filtered.loc[filtered["image_id"] == 200, "coralnet_id"].tolist() == [1, 2]


def test_coralnet_loader_reads_local_manifest_offline(tmp_path):
    out, _ = collapse_manifest(
        _manifest([1, 2, 3, 1, 2, 6]), ID2NAME, REPLACEMENTS, flagged_sources=[20]
    )
    path = tmp_path / "coralnet_training_manifest_phylum.parquet"
    out.to_parquet(path, index=False)

    bucket = "test-bucket"
    cache = LocalS3Cache.configure(tmp_path / "cache", offline=True)
    id2name_path = cache.local_path(bucket, _CORALNET_ID2NAME_KEY)
    id2name_path.parent.mkdir(parents=True)
    id2name_path.write_text(json.dumps(ID2NAME))

    dataset = CoralNetDataset.__new__(CoralNetDataset)
    dataset.manifest_path = str(path)
    dataset.source_bucket = bucket
    annotations, images = dataset.load_annotations()

    assert annotations["source_label_name"].tolist() == [
        "acropora",
        "agelas",
        "sand",
        "coral",
        "sponge",
        "not in csv",
    ]
    assert len(images) == 4
