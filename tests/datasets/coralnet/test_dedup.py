"""CoralNet deduplication rules (``mermaidseg.datasets.coralnet.dedup``) on tiny synthetic inputs."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from mermaidseg.datasets.coralnet.dedup import dedup, leakage, splits
from mermaidseg.datasets.coralnet.dedup.common import (
    DEFAULT_BUCKET,
    DEFAULT_MANIFEST,
    ID2NAME_KEY,
    add_split_labels,
    compare_labels,
    coralnet_splits,
    load_concepts,
    read_source_list,
)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
NG = "not_given"
# CoralNet label id -> (name, concept fields: six taxonomic ranks + bleached).
LABELS = {
    1: ("hard coral", ["animalia", "cnidaria", "hexacorallia", NG, NG, NG, NG]),
    2: ("acropora", ["animalia", "cnidaria", "hexacorallia", "scleractinia", "acroporidae", "acropora", NG]),
    3: ("acropora sp.", ["animalia", "cnidaria", "hexacorallia", "scleractinia", "acroporidae", "acropora", NG]),
    4: ("sand", ["none"] * 6 + [NG]),
    5: ("bleached", [NG] * 6 + ["true"]),
    6: ("unlisted label", None),
    7: ("other unlisted label", None),
}


@pytest.fixture
def concepts_csv(tmp_path):
    rows = [
        {"source_label_class_name": name, "source_dataset_source": "coralnet", "target_label_class_name": "",
         "target_dataset_source": "", "species": NG, **dict(zip([*RANKS, "bleached"], fields, strict=True))}
        for name, fields in LABELS.values() if fields is not None
    ]
    path = tmp_path / "class_to_concepts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def feat(concepts_csv):
    concepts = load_concepts(concepts_csv)
    return {lid: (concepts.loc[name] if name in concepts.index else None) for lid, (name, _) in LABELS.items()}


def test_compare_labels(feat):
    assert compare_labels(feat[2], feat[1]) == "a_finer"
    assert compare_labels(feat[1], feat[2]) == "b_finer"
    assert compare_labels(feat[2], feat[3]) == "equivalent"
    assert compare_labels(feat[1], feat[5]) == "compatible"
    assert compare_labels(feat[2], feat[4]) == "contradictory"
    assert compare_labels(feat[2], feat[6]) == "unmapped"


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ((2,), (2, "same_label", False)),
        ((1, 2), (2, "finer", False)),  # the finest consistent label wins, whichever copy has it
        ((2, 1), (2, "finer", False)),
        ((3, 2), (3, "equivalent", False)),  # synonyms: the best-ranked copy's label
        ((5, 1), (1, "compatible", False)),  # no superset: most taxonomic ranks specified
        ((4, 2), (4, "contradictory", False)),  # clash: the best-ranked copy's label
        ((6, 2), (2, "unmapped", True)),  # labels missing from the CSV are ignored
        ((6, 7), (6, "unknown", True)),
        ((1, 6, 2), (2, "finer", True)),
    ],
)
def test_resolver(feat, labels, expected):
    assert dedup.make_resolver(feat)(labels) == expected


def test_config_helpers(tmp_path):
    lst = tmp_path / "sources.txt"
    lst.write_text("12  # low quality\n\n# comment\n7\n")
    assert read_source_list(lst) == {7, 12}
    assert read_source_list(None) == set()

    cfg = tmp_path / "data_config.yaml"
    cfg.write_text("data:\n  coralnet:\n    train:\n      whitelist_sources: [1, 2]\n"
                   "    val:\n      whitelist_sources: [3]\n")
    assert coralnet_splits(cfg) == {"train": {1, 2}, "val": {3}}

    pairs = pd.DataFrame({"source_a": [1, 1, 4], "source_b": [2, 3, 5], "n": [1, 1, 1]})
    out = add_split_labels(pairs, coralnet_splits(cfg))
    assert out["cross_split"].tolist() == [False, True, False]
    assert out["split_b"].tolist() == ["train", "val", "unused"]


def test_dedup_end_to_end(tmp_path, concepts_csv, monkeypatch):
    """Image 10 (source 1) and 20 (source 2) are copies; source 1 is better ranked, so 10 is kept.

    At (5, 5) the copy's finer label replaces the kept one, at (6, 6) the labels contradict (kept
    copy's label stays, point goes to review), and (7, 7) is added from the copy. Image 30 is unique.
    """
    bucket = tmp_path / "cache" / DEFAULT_BUCKET
    (bucket / DEFAULT_MANIFEST).parent.mkdir(parents=True)
    (bucket / ID2NAME_KEY).parent.mkdir(parents=True)
    (bucket / ID2NAME_KEY).write_text(json.dumps({str(k): name for k, (name, _) in LABELS.items()}))
    pd.DataFrame({
        "source_id": pd.array([1, 1, 2, 2, 2, 3], dtype="int32"),
        "image_id": ["10", "10", "20", "20", "20", "30"],
        "row": pd.array([5, 6, 5, 6, 7, 1], dtype="int16"),
        "col": pd.array([5, 6, 5, 6, 7, 1], dtype="int16"),
        "coralnet_id": pd.array([1, 4, 2, 2, 4, 4], dtype="int32"),
    }).to_parquet(bucket / DEFAULT_MANIFEST, index=False)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    pd.DataFrame({"source_id": [1, 2, 3], "image_id": ["10", "20", "30"],
                  "hash": ["h", "h", "u"]}).to_parquet(outputs / "image_hashes_exact_all.parquet")
    pd.DataFrame({"source_id": [1, 2], "cluster": [0, 0], "rank": [1, 2]}).to_csv(
        outputs / "annotation_source_ranking.csv", index=False)
    cfg = tmp_path / "data_config.yaml"
    cfg.write_text("data:\n  coralnet:\n    train:\n      whitelist_sources: [1, 2, 3]\n")
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("")

    out_dir, config_dir = tmp_path / "dedup", tmp_path / "configs"
    monkeypatch.setattr(sys, "argv", [
        "dedup", "--outputs", str(outputs), "--out-dir", str(out_dir), "--config-dir", str(config_dir),
        "--cache-root", str(tmp_path / "cache"), "--concepts-csv", str(concepts_csv),
        "--split-config", str(cfg), "--exclude-sources", str(exclude),
    ])
    dedup.main()

    blacklist = pd.read_parquet(config_dir / "image_blacklist.parquet")
    assert blacklist[["source_id", "image_id", "kept_image_id"]].to_numpy().tolist() == [[2, "20", "10"]]
    overrides = pd.read_parquet(config_dir / "annotation_overrides.parquet").sort_values("row")
    assert overrides[["image_id", "row", "col", "coralnet_id"]].to_numpy().tolist() == [
        ["10", 5, 5, 2], ["10", 6, 6, 4], ["10", 7, 7, 4]]
    prov = pd.read_parquet(out_dir / "point_provenance.parquet").sort_values("row")
    assert prov["change"].tolist() == ["replaced", "unchanged", "added"]
    assert prov["category"].tolist() == ["finer", "contradictory", "same_label"]
    assert prov["origin_copy_rank"].tolist() == [1, 0, 1]
    review = pd.read_csv(out_dir / "review_points.csv")
    assert review[["row", "category", "chosen_label"]].to_numpy().tolist() == [[6, "contradictory", "sand"]]
    assert pd.read_parquet(out_dir / "images_unique.parquet")["image_id"].tolist() == ["30"]
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_points_before"] == 6 and summary["n_points_after"] == 4


def test_replace_coralnet_block():
    text = ("data:\n  mermaid:\n    train: {}\n  coralnet:\n    train:\n      whitelist_sources: [1]\n"
            "  catlin:\n    train: {}\n")
    out = splits.replace_coralnet_block(text, {"train": [1, 2], "val": [3]},
                                        {"image_blacklist": "configs/coralnet/image_blacklist.parquet"})
    assert out.startswith("data:\n  mermaid:\n    train: {}\n  coralnet:\n")
    assert out.endswith("  catlin:\n    train: {}\n")
    assert "      whitelist_sources: [1, 2]\n      image_blacklist: configs/coralnet/image_blacklist.parquet\n" in out
    assert "    val:\n      whitelist_sources: [3]\n" in out


def test_project_edges():
    meta = pd.DataFrame({
        "source_id": [10, 11, 20, 30, 40],
        "admins": ["ann", "ann;bob", "ann", "carl", "ann"],
        "latitude": [0.0, 40.0, 0.1, 0.0, -30.0],
        "longitude": [0.0, 40.0, 0.1, 0.0, -30.0],
    })
    sources = [10, 11, 20, 30, 40]
    edges = {(a, b): why for a, b, why in splits.project_edges(meta, sources, max_id_gap=1, max_km=50)}
    # The id gap counts positions in the sorted source list: 10-11 and 11-20 are neighbours.
    # 10-20: two positions apart but ~16 km apart. 40: same admin, but far in the list and on the
    # map. 30: a different admin.
    assert edges == {(10, 11): "id", (11, 20): "id", (10, 20): "location"}


def test_leakage(tmp_path, monkeypatch):
    pd.DataFrame({"source_id": [1, 1, 2, 3], "hash": ["a", "b", "a", "c"]}).to_parquet(
        tmp_path / "image_hashes_exact_all.parquet")
    pd.DataFrame({"source_a": [1], "source_b": [2], "n_overlap_images": [1]}).to_csv(
        tmp_path / "source_overlap_exact_all.csv", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("coralnet:\n  train:\n    whitelist_sources: [1]\n  val:\n    whitelist_sources: [2]\n"
                   "  test:\n    whitelist_sources: [3]\n")
    monkeypatch.setattr(sys, "argv", ["leakage", "--config", str(cfg), "--outputs", str(tmp_path)])
    leakage.main()

    per_source = pd.read_csv(tmp_path / "split_leakage_cfg.csv").set_index(["eval_split", "other_split"])
    assert per_source.loc[("val", "train"), "n_dup"] == 1
    assert per_source.loc[("test", "train"), "n_dup"] == 0
    pairs = pd.read_csv(tmp_path / "split_leakage_pairs_cfg.csv")
    assert pairs[["source_a", "source_b", "split_a", "split_b"]].to_numpy().tolist() == [[1, 2, "train", "val"]]
