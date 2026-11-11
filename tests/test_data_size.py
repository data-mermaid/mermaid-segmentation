"""Data-size experiments: drops, allowlists, configs, and launcher wiring."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from datasets import Dataset

from mermaidseg.dataset_reconciliation.concept_schema import ConceptSchema
from mermaidseg.datasets.base_dataset import BaseCoralDataset
from mermaidseg.datasets.coralscapes_v2 import CoralscapesV2Dataset
from mermaidseg.datasets.data_size import (
    DATASET_ORDER,
    EXPERIMENTS,
    FRACTIONS,
    HF_DATASETS,
    METRIC,
    SEED,
    SizeUnit,
    all_profile_names,
    apply_hf_image_keep,
    assemble_config,
    assert_profile_config,
    closest_prefix,
    config_for_plan,
    drop_count,
    filter_point_frames,
    fractions_for,
    genus_counts,
    parse_profile_name,
    per_dataset_counts,
    permute_uids,
    plan_drop,
    profile_name,
    rank_units,
    run_name,
)
from mermaidseg.datasets.factory import configured_source_names
from mermaidseg.datasets.ucsd_mosaics import UCSDMosaicsDataset
from scripts.build_data_size_profiles import _profile_record

_REPO = Path(__file__).resolve().parents[1]
_CONCEPTS = _REPO / "configs" / "class_to_concepts.csv"
_BASE_CONFIG = _REPO / "configs" / "data_config.yaml"



def _concepts(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _inventory(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.insert(0, "image_uid", np.arange(len(frame), dtype=np.int64))
    return frame


def _stats(inventory: pd.DataFrame, genera: dict[str, int]) -> pd.DataFrame:
    rows = []
    for unit_id, group in inventory.groupby("unit_id", sort=False):
        rows.append(
            {
                "unit_id": unit_id,
                "dataset": group["dataset"].iloc[0],
                "n_images": len(group),
                "n_genera": genera[str(unit_id)],
                "n_coral_genera": genera[str(unit_id)],
                "n_labels": 1,
                "n_unmapped_labels": 0,
            }
        )
    return pd.DataFrame(rows)


def _mini_base() -> dict:
    return {
        "data": {
            "local_cache_dir": "/tmp/cache",
            "coralnet": {
                "train": {"whitelist_sources": [1, 2, 3], "image_blacklist": "bl.parquet"},
                "val": {"whitelist_sources": [9]},
                "test": {"whitelist_sources": [8]},
            },
            "mermaid": {"train": {"annotation_date_cutoff": "2026-08-12"}, "val": "None"},
            "coralscapes_v2": {
                "train": {"split": "train"},
                "val": {"split": "validation"},
                "test": {"split": "test"},
            },
            "catlin_seaview": {"train": {"whitelist_regions": ["ATL"]}, "val": "None"},
            "moorea_labeled_corals": {"train": {"whitelist_years": ["2008"]}, "val": "None"},
            "ucsd_mosaics": {"train": {"whitelist_sites": ["fr3"]}, "val": "None"},
            "pacific_labeled_corals": {
                "train": "None",
                "val": {"whitelist_subsets": ["reference"]},
            },
            "benthos_yuval": {"train": "None", "val": {"whitelist_sites": ["RS24"]}},
            "coralscapes": {"train": "None", "val": "None"},
        }
    }


class _CoralNet(BaseCoralDataset):
    SOURCE_NAME = "coralnet"


def _point_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 2, 2],
            "image_id": ["a", "a", "b", "x", "x"],
            "row": [0, 1, 0, 0, 1],
            "col": [0, 0, 0, 0, 0],
            "source_label_name": ["acropora", "porites", "acropora", "sand", "sand"],
        }
    )
    images = pd.DataFrame(
        {"source_id": [1, 1, 2], "image_id": ["a", "b", "x"], "image_key": ["ka", "kb", "kx"]}
    )
    return annotations, images


# ---------- names and genus counts ----------


def test_profile_names_round_trip_and_cover_the_grid():
    names = all_profile_names()
    assert names == [
        "exp1_drop10",
        "exp1_drop20",
        "exp1_drop50",
        "exp2_drop10",
        "exp2_drop20",
        "exp2_drop50",
        "exp3_drop10",
        "exp3_drop20",
        "exp3_drop50",
        "exp4_drop10",
        "exp4_drop20",
        "exp4_drop50",
    ]
    assert len(set(names)) == 12
    for experiment in EXPERIMENTS:
        for fraction in FRACTIONS:
            name = profile_name(experiment, fraction)
            assert parse_profile_name(name) == (experiment, fraction)
            assert run_name(name) == f"cbm_dsize_{name}"
    with pytest.raises(ValueError):
        profile_name("exp1", 0.15)
    with pytest.raises(ValueError):
        profile_name("exp2", 0.0)
    with pytest.raises(ValueError):
        parse_profile_name("sparse")
    with pytest.raises(ValueError):
        parse_profile_name("exp2_drop0")
    with pytest.raises(ValueError):
        run_name("cbm_dq_sparse")
    assert fractions_for("exp1") == (0.0, 0.1, 0.2, 0.5)
    assert fractions_for("exp3") == FRACTIONS
    assert profile_name("exp1", 0.0) == "exp1_drop0"
    assert parse_profile_name("exp1_drop0") == ("exp1", 0.0)
    assert run_name("exp1_drop0") == "cbm_dsize_exp1_drop0"
    assert "exp1_drop0" not in names


def test_genus_counts_use_distinct_genus_entries_for_that_dataset():
    concepts = _concepts(
        {
            "source_label_class_name": "acropora",
            "source_dataset_source": "coralnet",
            "genus": "Acropora",
            "class": "Hexacorallia",
        },
        {
            "source_label_class_name": "table acropora",
            "source_dataset_source": "coralnet",
            "genus": "acropora",
            "class": "hexacorallia",
        },
        {
            "source_label_class_name": "gorgonia",
            "source_dataset_source": "coralnet",
            "genus": "gorgonia",
            "class": "octocorallia",
        },
        {
            "source_label_class_name": "hard coral",
            "source_dataset_source": "coralnet",
            "genus": "not_given",
            "class": "hexacorallia",
        },
        {
            "source_label_class_name": "sponge",
            "source_dataset_source": "coralnet",
            "genus": "agelas",
            "class": "demospongiae",
        },
        {
            "source_label_class_name": "mystery",
            "source_dataset_source": "mermaid",
            "genus": "mysterygenus",
            "class": "hexacorallia",
        },
        {
            "source_label_class_name": "unused",
            "source_dataset_source": "coralnet",
            "genus": "unusedgenus",
            "class": "hexacorallia",
        },
        {
            "source_label_class_name": "acropora",
            "source_dataset_source": "coralnet",
            "genus": "othergenus",
            "class": "hexacorallia",
        },
    )
    counts = genus_counts(
        ["Acropora", "table acropora", "gorgonia", "hard coral", "sponge", "nope", "Acropora"],
        concepts,
        "coralnet",
    )
    assert counts.n_genera == 3  # acropora, gorgonia, agelas
    assert counts.n_coral_genera == 2  # acropora once, plus gorgonia; agelas is not a coral class
    assert counts.n_labels == 6
    assert counts.n_unmapped_labels == 1
    with pytest.raises(ValueError, match="no rows"):
        genus_counts(["acropora"], concepts, "not_a_dataset")


def test_every_training_dataset_has_its_own_concept_rows():
    concepts = pd.read_csv(_CONCEPTS, dtype=str, keep_default_na=False)
    for dataset in DATASET_ORDER:
        counts = genus_counts([], concepts, dataset)
        assert counts.n_genera == 0
        assert counts.n_coral_genera == 0


def test_drop_count_is_the_closest_integer_and_nested():
    assert drop_count(11, 0.1) == 1
    assert drop_count(11, 0.2) == 2
    assert drop_count(11, 0.5) == 6
    assert drop_count(1, 0.5) == 1
    for n in range(0, 500):
        counts = [drop_count(n, fraction) for fraction in FRACTIONS]
        assert counts == sorted(counts)
        assert counts[-1] <= n


# ---------- which images and sources are removed ----------


def _ranked_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three CoralNet sources: 1, 1 and 100 images, with 3, 2 and 1 coral genera."""
    rows = [
        {"dataset": "coralnet", "unit_id": "coralnet:0000003", "source_id": 3, "image_id": "a"},
        {"dataset": "coralnet", "unit_id": "coralnet:0000002", "source_id": 2, "image_id": "b"},
    ]
    rows.extend(
        {
            "dataset": "coralnet",
            "unit_id": "coralnet:0000001",
            "source_id": 1,
            "image_id": f"c{i}",
        }
        for i in range(100)
    )
    inventory = _inventory(rows)
    stats = _stats(
        inventory,
        {"coralnet:0000003": 3, "coralnet:0000002": 2, "coralnet:0000001": 1},
    )
    return inventory, stats


def test_ranking_uses_coral_genera_and_breaks_ties_by_source_id():
    units = [
        SizeUnit("coralnet:0000002", "coralnet", 10, n_coral_genera=5, n_genera=5),
        SizeUnit("coralnet:0000001", "coralnet", 10, n_coral_genera=5, n_genera=1),
        SizeUnit("dataset:catlin_seaview", "catlin_seaview", 1000, n_coral_genera=0, n_genera=40),
    ]
    ranked = rank_units(units, METRIC)
    assert [unit.unit_id for unit in ranked] == [
        "coralnet:0000001",
        "coralnet:0000002",
        "dataset:catlin_seaview",
    ]


def test_closest_prefix_prefers_the_shorter_prefix_on_a_tie():
    ranked = rank_units(
        [
            SizeUnit("coralnet:0000002", "coralnet", 40, n_coral_genera=5),
            SizeUnit("coralnet:0000001", "coralnet", 60, n_coral_genera=1),
        ]
    )
    # 0 and 40 images are equally far from 20% of 100.
    assert closest_prefix(ranked, 0.2, weight="n_images") == []


def test_a_source_larger_than_the_target_is_not_split():
    ranked = rank_units(
        [
            SizeUnit("coralnet:0000009", "coralnet", 40, n_coral_genera=10),
            SizeUnit("coralnet:0000008", "coralnet", 5, n_coral_genera=1),
            SizeUnit("coralnet:0000007", "coralnet", 5, n_coral_genera=0),
        ]
    )
    assert closest_prefix(ranked, 0.1, weight="n_images") == []
    assert closest_prefix(ranked, 0.2, weight="n_images") == []
    dropped = closest_prefix(ranked, 0.5, weight="n_images")
    assert [unit.unit_id for unit in dropped] == ["coralnet:0000009"]


def test_source_prefixes_are_nested_for_every_positive_weight_list():
    weights = [3, 1, 8, 2, 5, 13, 1, 1, 4]
    units = [
        SizeUnit(f"coralnet:{index:07d}", "coralnet", weight, n_coral_genera=100 - index)
        for index, weight in enumerate(weights)
    ]
    ranked = rank_units(units)
    dropped = [closest_prefix(ranked, fraction, weight="n_images") for fraction in FRACTIONS]
    ids = [{unit.unit_id for unit in prefix} for prefix in dropped]
    assert ids[0] <= ids[1] <= ids[2]
    unit_prefixes = [closest_prefix(ranked, fraction, weight="units") for fraction in FRACTIONS]
    unit_ids = [{unit.unit_id for unit in prefix} for prefix in unit_prefixes]
    assert unit_ids[0] <= unit_ids[1] <= unit_ids[2]


def test_exp2_50_percent_removes_the_highest_coral_genera_sources():
    """The 50% CoralNet cut removes a prefix of the highest-genera sources.

    Five sources, 100 images. Sorted by coral genera, the first two sources are
    30 + 20 images, exactly half. Those two are removed whole; the three sources
    with fewer coral genera stay, including a larger low-genera source.
    """
    specs = [
        ("coralnet:0000010", 10, 30, 10),
        ("coralnet:0000008", 8, 20, 8),
        ("coralnet:0000006", 6, 15, 4),
        ("coralnet:0000003", 3, 25, 2),
        ("coralnet:0000001", 1, 10, 1),
    ]
    rows = []
    genera = {}
    for unit_id, source_id, n_images, n_genera in specs:
        genera[unit_id] = n_genera
        rows.extend(
            {
                "dataset": "coralnet",
                "unit_id": unit_id,
                "source_id": source_id,
                "image_id": f"{unit_id}-{i}",
            }
            for i in range(n_images)
        )
    inventory = _inventory(rows)
    stats = _stats(inventory, genera)
    plan = plan_drop("exp2", 0.5, inventory, stats)

    assert plan.n_images_total == 100
    assert plan.dropped_units == ("coralnet:0000010", "coralnet:0000008")
    assert plan.n_images_dropped == 50
    dropped = set(plan.dropped_units)
    assert min(genera[unit] for unit in dropped) > max(
        genera[unit] for unit in genera if unit not in dropped
    )
    # Every prefix of the same ranking is farther from 50 images, or equally far and longer.
    ranked = sorted(specs, key=lambda spec: (-spec[3], spec[0]))
    running = 0
    prefix_counts = [0]
    for spec in ranked:
        running += spec[2]
        prefix_counts.append(running)
    target = 50
    distances = [abs(count - target) for count in prefix_counts]
    assert prefix_counts[len(plan.dropped_units)] == plan.n_images_dropped
    assert distances[len(plan.dropped_units)] == min(distances)

    # Half of these six units would also take the 20-image source. The image fraction stops
    # at Catlin plus the 30-image source.
    rows.extend(
        {
            "dataset": "catlin_seaview",
            "unit_id": "dataset:catlin_seaview",
            "source_id": pd.NA,
            "image_id": f"catlin-{i}",
        }
        for i in range(40)
    )
    genera["dataset:catlin_seaview"] = 12
    inventory = _inventory(rows)
    stats = _stats(inventory, genera)
    meta = plan_drop("exp4", 0.5, inventory, stats)
    assert meta.n_images_total == 140
    assert meta.dropped_units == ("dataset:catlin_seaview", "coralnet:0000010")
    assert meta.n_images_dropped == 70
    assert meta.n_units_dropped != drop_count(meta.n_units_total, 0.5)


def test_exp4_image_fraction_can_drop_a_dataset_exp2_ignores():
    inventory, stats = _ranked_population()
    extra = _inventory(
        [
            {
                "dataset": "catlin_seaview",
                "unit_id": "dataset:catlin_seaview",
                "source_id": pd.NA,
                "image_id": f"catlin-{i}",
            }
            for i in range(80)
        ]
    )
    extra["image_uid"] = inventory["image_uid"].max() + 1 + extra.index
    inventory = pd.concat([inventory, extra], ignore_index=True)
    stats = pd.concat(
        [stats, _stats(extra, {"dataset:catlin_seaview": 10})],
        ignore_index=True,
    )
    coralnet = plan_drop("exp2", 0.5, inventory, stats)
    meta = plan_drop("exp4", 0.5, inventory, stats)
    assert coralnet.n_images_total == 102
    assert coralnet.dropped_units == ("coralnet:0000003", "coralnet:0000002")
    assert coralnet.n_images_dropped == 2
    assert "dataset:catlin_seaview" not in coralnet.dropped_units
    # 182 images. The closest whole-unit prefix to 91 is Catlin plus the two
    # small CoralNet sources (82). Half of the four units would stop one source earlier.
    assert meta.n_images_total == 182
    assert meta.dropped_units == (
        "dataset:catlin_seaview",
        "coralnet:0000003",
        "coralnet:0000002",
    )
    assert meta.n_images_dropped == 82
    assert meta.n_units_dropped != drop_count(meta.n_units_total, 0.5)
    for plan in (coralnet, meta):
        dropped_units = set(plan.dropped_units)
        dropped_uids = set(plan.dropped_uids)
        if plan.experiment == "exp4":
            population = inventory
        else:
            population = inventory[inventory.dataset == "coralnet"]
        for row in population.itertuples(index=False):
            if row.unit_id in dropped_units:
                assert row.image_uid in dropped_uids
            else:
                assert row.image_uid not in dropped_uids


def test_iid_drop_is_one_permutation_and_can_split_a_source():
    rows = [
        {"dataset": "coralnet", "unit_id": "coralnet:0000001", "source_id": 1, "image_id": i}
        for i in range(9)
    ]
    rows.append(
        {"dataset": "coralnet", "unit_id": "coralnet:0000002", "source_id": 2, "image_id": 9}
    )
    inventory = _inventory(rows)
    stats = _stats(inventory, {"coralnet:0000001": 1, "coralnet:0000002": 1})
    perm = permute_uids(inventory["image_uid"].tolist(), SEED)
    plans = [plan_drop("exp1", fraction, inventory, stats, perm) for fraction in FRACTIONS]
    for plan, fraction in zip(plans, FRACTIONS, strict=True):
        k = drop_count(len(perm), fraction)
        assert plan.dropped_uids == tuple(int(uid) for uid in perm[:k])
    assert set(plans[0].dropped_uids) <= set(plans[1].dropped_uids) <= set(plans[2].dropped_uids)
    big = set(inventory.loc[inventory["unit_id"] == "coralnet:0000001", "image_uid"])
    dropped = set(plans[-1].dropped_uids)
    assert big & dropped and big - dropped
    # A per-source 50% quota would always drop the one-image source. The shared shuffle need not.
    assert drop_count(1, 0.5) == 1
    small = int(inventory.loc[inventory["unit_id"] == "coralnet:0000002", "image_uid"].iloc[0])
    assert any(small not in set(permute_uids(range(10), seed)[:5]) for seed in range(30))


def test_exp1_counts_other_datasets_as_untrained():
    inventory, stats = _ranked_population()
    extra = _inventory(
        [{"dataset": "mermaid", "unit_id": "dataset:mermaid", "image_id": "m", "source_id": pd.NA}]
    )
    extra["image_uid"] = inventory["image_uid"].max() + 1
    inventory = pd.concat([inventory, extra], ignore_index=True)
    stats = pd.concat(
        [stats, _stats(extra, {"dataset:mermaid": 0})],
        ignore_index=True,
    )
    perm = permute_uids(inventory.loc[inventory["dataset"] == "coralnet", "image_uid"], SEED)
    plan = plan_drop("exp1", 0.1, inventory, stats, perm)
    counts = per_dataset_counts(plan, inventory)
    assert counts["mermaid"] == {"n_total": 1, "n_kept": 0}
    assert counts["coralnet"]["n_kept"] == counts["coralnet"]["n_total"] - plan.n_images_dropped


def test_exp1_drop0_keeps_every_coralnet_training_image(tmp_path: Path):
    inventory, stats = _ranked_population()
    extra = _inventory(
        [{"dataset": "mermaid", "unit_id": "dataset:mermaid", "image_id": "m", "source_id": pd.NA}]
    )
    extra["image_uid"] = inventory["image_uid"].max() + 1
    inventory = pd.concat([inventory, extra], ignore_index=True)
    stats = pd.concat([stats, _stats(extra, {"dataset:mermaid": 0})], ignore_index=True)
    perm = permute_uids(inventory.loc[inventory["dataset"] == "coralnet", "image_uid"], SEED)
    plan = plan_drop("exp1", 0.0, inventory, stats, perm)
    assert plan.n_images_dropped == 0
    assert plan.dropped_uids == ()
    assert plan.n_images_kept == int((inventory["dataset"] == "coralnet").sum())
    smaller = plan_drop("exp1", 0.1, inventory, stats, perm)
    assert set(plan.dropped_uids) <= set(smaller.dropped_uids)
    base = _mini_base()
    config, frames = config_for_plan(base, plan, inventory, {}, keep_dir=tmp_path / "keep")
    assert frames == {}
    assert config["data"]["coralnet"]["train"] == base["data"]["coralnet"]["train"]
    assert "image_keep" not in config["data"]["coralnet"]["train"]
    assert config["data"]["mermaid"]["train"] == "None"
    counts = per_dataset_counts(plan, inventory)
    assert counts["coralnet"]["n_kept"] == counts["coralnet"]["n_total"]
    assert counts["mermaid"]["n_kept"] == 0


# ---------- configs do not touch val, test, or dense masks ----------


def test_exp1_keeps_the_coralnet_whitelist_and_turns_other_training_off():
    base = yaml.safe_load(_BASE_CONFIG.read_text())
    config = assemble_config(base, "exp1", image_keep={"coralnet": "/tmp/coralnet.parquet"})
    train = config["data"]["coralnet"]["train"]
    assert train["whitelist_sources"] == base["data"]["coralnet"]["train"]["whitelist_sources"]
    assert train["image_keep"] == "/tmp/coralnet.parquet"
    assert train["image_blacklist"] == base["data"]["coralnet"]["train"]["image_blacklist"]
    for dataset in (
        "mermaid",
        "coralscapes_v2",
        "catlin_seaview",
        "moorea_labeled_corals",
        "ucsd_mosaics",
    ):
        assert config["data"][dataset]["train"] == "None"
    assert config["data"]["coralscapes_v2"]["val"] == base["data"]["coralscapes_v2"]["val"]
    assert config["data"]["coralscapes_v2"]["test"] == base["data"]["coralscapes_v2"]["test"]
    assert config["data"]["pacific_labeled_corals"] == base["data"]["pacific_labeled_corals"]
    assert config["data"]["benthos_yuval"] == base["data"]["benthos_yuval"]
    assert (
        base["data"]["coralnet"]["val"]
        == yaml.safe_load(_BASE_CONFIG.read_text())["data"]["coralnet"]["val"]
    )


def test_exp4_drops_whole_datasets_and_leaves_dense_training_unchanged():
    base = yaml.safe_load(_BASE_CONFIG.read_text())
    sources = list(base["data"]["coralnet"]["train"]["whitelist_sources"])
    kept = sources[1:]
    config = assemble_config(
        base, "exp4", kept_coralnet_sources=kept, dropped_datasets=["catlin_seaview"]
    )
    assert config["data"]["catlin_seaview"]["train"] == "None"
    assert config["data"]["catlin_seaview"]["val"] == "None"
    assert config["data"]["coralscapes_v2"]["train"] == base["data"]["coralscapes_v2"]["train"]
    assert config["data"]["ucsd_mosaics"]["train"] == base["data"]["ucsd_mosaics"]["train"]
    assert "image_keep" not in config["data"]["coralnet"]["train"]
    assert "hf_revision" not in config["data"]["coralnet"]["train"]
    assert config["data"]["coralnet"]["train"]["whitelist_sources"] == kept
    held_out = set(base["data"]["coralnet"]["val"]["whitelist_sources"]) | set(
        base["data"]["coralnet"]["test"]["whitelist_sources"]
    )
    assert set(kept).isdisjoint(held_out)
    assert sources[0] not in kept


def test_exp3_pins_dense_rows_without_replacing_the_repo():
    base = yaml.safe_load(_BASE_CONFIG.read_text())
    keeps = {dataset: f"/tmp/{dataset}.parquet" for dataset in DATASET_ORDER}
    config = assemble_config(
        base,
        "exp3",
        image_keep=keeps,
        hf_pins={
            "coralscapes_v2": {"revision": "abc", "n_rows": 10},
            "ucsd_mosaics": {"revision": "def", "n_rows": 20},
        },
    )
    coralscapes = config["data"]["coralscapes_v2"]["train"]
    assert coralscapes["split"] == "train"
    assert coralscapes["hf_revision"] == "abc"
    assert coralscapes["image_keep_n_rows"] == 10
    assert "hf_repo_id" not in coralscapes
    assert "sparse" not in yaml.safe_dump(coralscapes)
    assert config["data"]["coralscapes_v2"]["val"]["split"] == "validation"
    assert config["data"]["ucsd_mosaics"]["train"]["hf_revision"] == "def"
    assert config["data"]["mermaid"]["train"]["image_keep"].endswith("mermaid.parquet")
    assert "hf_revision" not in config["data"]["mermaid"]["train"]
    with pytest.raises(ValueError, match="row count"):
        assemble_config(base, "exp3", image_keep=keeps)
    with pytest.raises(ValueError, match="revision"):
        assemble_config(
            base,
            "exp3",
            image_keep=keeps,
            hf_pins={"coralscapes_v2": {"n_rows": 10}, "ucsd_mosaics": {"n_rows": 20}},
        )


def test_a_val_edit_or_a_held_out_source_is_rejected():
    base = _mini_base()
    good = assemble_config(base, "exp2", kept_coralnet_sources=[1, 2], dropped_datasets=[])
    leaked = yaml.safe_load(yaml.safe_dump(good))
    leaked["data"]["coralnet"]["train"]["whitelist_sources"] = [1, 9]
    with pytest.raises(ValueError, match="subset"):
        assert_profile_config(base, leaked, "exp2")
    changed_val = yaml.safe_load(yaml.safe_dump(good))
    changed_val["data"]["coralnet"]["val"]["whitelist_sources"] = [1]
    with pytest.raises(ValueError, match="val"):
        assert_profile_config(base, changed_val, "exp2")
    sparse = yaml.safe_load(yaml.safe_dump(good))
    sparse["data"]["coralscapes_v2"]["train"] = {"split": "train", "hf_repo_id": "/tmp/sparse"}
    with pytest.raises(ValueError, match="sparse|hf_repo_id"):
        assert_profile_config(base, sparse, "exp4")


def test_config_for_plan_writes_allowlists_only_for_iid_drops(tmp_path: Path):
    inventory, stats = _ranked_population()
    perm = permute_uids(inventory["image_uid"].tolist(), SEED)
    plan = plan_drop("exp1", 0.5, inventory, stats, perm)
    keys = {"coralnet": ("source_id", "image_id")}
    config, frames = config_for_plan(
        _mini_base(), plan, inventory, keys, keep_dir=tmp_path / "keep"
    )
    assert set(frames) == {"coralnet"}
    assert set(frames["coralnet"]["image_id"]) == set(
        inventory.loc[~inventory["image_uid"].isin(plan.dropped_uids), "image_id"].astype(str)
    )
    assert config["data"]["coralnet"]["train"]["image_keep"] == str(
        tmp_path / "keep" / "coralnet.parquet"
    )
    assert config["data"]["mermaid"]["train"] == "None"
    source_plan = plan_drop("exp2", 0.5, inventory, stats)
    source_config, source_frames = config_for_plan(
        _mini_base(), source_plan, inventory, keys, keep_dir=tmp_path / "unused"
    )
    assert source_frames == {}
    assert "image_keep" not in source_config["data"]["coralnet"]["train"]
    assert source_config["data"]["coralnet"]["train"]["whitelist_sources"] == [1]
    record = _profile_record(
        source_plan, per_dataset_counts(source_plan, inventory), base_config=_BASE_CONFIG
    )
    assert record["metric"] == METRIC
    assert record["seed"] == SEED
    assert record["fraction"] == 0.5
    assert record["n_images_dropped"] == source_plan.n_images_dropped


# ---------- allowlists ----------


def test_image_keep_drops_only_the_listed_images_and_rejects_a_foreign_key(tmp_path: Path):
    annotations, images = _point_frames()
    untouched = _CoralNet(annotations.copy(), images.copy())
    assert len(untouched) == 3
    assert inspect.signature(BaseCoralDataset.__init__).parameters["image_keep"].default is None

    keep = pd.DataFrame(
        {"dataset": ["coralnet", "coralnet"], "source_id": [1, 2], "image_id": ["a", "x"]}
    )
    path = tmp_path / "keep.parquet"
    keep.to_parquet(path, index=False)
    mixed = pd.DataFrame(
        {"dataset": ["coralnet", "coralnet"], "source_id": ["1", 2.0], "image_id": ["a", "x"]}
    )
    mixed_ann, mixed_images = filter_point_frames(
        annotations.copy(), images.copy(), mixed, "coralnet"
    )
    assert len(mixed_images) == 2
    assert set(mixed_ann["image_id"]) == {"a", "x"}
    kept = _CoralNet(annotations.copy(), images.copy(), image_keep=path)
    assert len(kept) == 2
    got = {
        (int(source), str(image))
        for source, image in zip(
            kept.df_images["source_id"], kept.df_images["image_id"], strict=True
        )
    }
    assert got == {(1, "a"), (2, "x")}
    assert set(kept.df_annotations["image_id"]) == {"a", "x"}

    missing = pd.DataFrame({"dataset": ["coralnet"], "source_id": [1], "image_id": ["missing"]})
    missing_path = tmp_path / "missing.parquet"
    missing.to_parquet(missing_path, index=False)
    with pytest.raises(ValueError, match="not in this split"):
        _CoralNet(annotations.copy(), images.copy(), image_keep=missing_path)
    wrong = pd.DataFrame({"dataset": ["mermaid"], "source_id": [1], "image_id": ["a"]})
    wrong_path = tmp_path / "wrong.parquet"
    wrong.to_parquet(wrong_path, index=False)
    with pytest.raises(ValueError, match="mermaid"):
        _CoralNet(annotations.copy(), images.copy(), image_keep=wrong_path)


def test_same_image_id_in_two_sources_is_not_one_image(tmp_path: Path):
    annotations = pd.DataFrame(
        {
            "source_id": [1, 2],
            "image_id": ["x", "x"],
            "row": [0, 0],
            "col": [0, 0],
            "source_label_name": ["acropora", "sand"],
        }
    )
    images = pd.DataFrame({"source_id": [1, 2], "image_id": ["x", "x"]})
    keep = pd.DataFrame({"dataset": ["coralnet"], "source_id": [1], "image_id": ["x"]})
    path = tmp_path / "keep.parquet"
    keep.to_parquet(path, index=False)
    kept = _CoralNet(annotations, images, image_keep=path)
    assert len(kept) == 1
    assert int(kept.df_images["source_id"].iloc[0]) == 1
    assert list(kept.df_annotations["source_label_name"]) == ["acropora"]


def test_hf_allowlist_selects_rows_and_rejects_drift(tmp_path: Path):
    dataset = Dataset.from_dict(
        {"site": ["a", "a", "b"], "filename": ["1", "2", "3"], "label": [0, 1, 2]}
    )
    keep = pd.DataFrame(
        {"dataset": ["ucsd_mosaics", "ucsd_mosaics"], "site": ["a", "b"], "filename": ["1", "3"]}
    )
    path = tmp_path / "keep.parquet"
    keep.to_parquet(path, index=False)
    selected = apply_hf_image_keep(dataset, path, "ucsd_mosaics", 3)
    assert selected["filename"] == ["1", "3"]
    assert selected["label"] == [0, 2]

    with pytest.raises(ValueError, match="built for 4"):
        apply_hf_image_keep(dataset, path, "ucsd_mosaics", 4)
    absent = pd.DataFrame({"dataset": ["ucsd_mosaics"], "site": ["a"], "filename": ["missing"]})
    absent_path = tmp_path / "absent.parquet"
    absent.to_parquet(absent_path, index=False)
    with pytest.raises(ValueError, match="not in this split"):
        apply_hf_image_keep(dataset, absent_path, "ucsd_mosaics", 3)

    duplicate = Dataset.from_dict({"site": ["a", "a"], "filename": ["1", "1"]})
    one = pd.DataFrame({"dataset": ["ucsd_mosaics"], "site": ["a"], "filename": ["1"]})
    one_path = tmp_path / "one.parquet"
    one.to_parquet(one_path, index=False)
    with pytest.raises(ValueError, match="not unique"):
        apply_hf_image_keep(duplicate, one_path, "ucsd_mosaics", 2)

    indexed = pd.DataFrame({"dataset": ["coralscapes_v2", "coralscapes_v2"], "row_index": [2, 0]})
    index_path = tmp_path / "index.parquet"
    indexed.to_parquet(index_path, index=False)
    by_index = apply_hf_image_keep(dataset, index_path, "coralscapes_v2", 3)
    assert by_index["filename"] == ["1", "3"]


def test_hub_image_keep_without_a_revision_fails_before_download():
    with pytest.raises(ValueError, match="hf_revision"):
        CoralscapesV2Dataset(split="train", image_keep="keep.parquet", image_keep_n_rows=1)
    with pytest.raises(ValueError, match="hf_revision"):
        UCSDMosaicsDataset(image_keep="keep.parquet", image_keep_n_rows=1)
    assert inspect.signature(CoralscapesV2Dataset.__init__).parameters["image_keep"].default is None
    assert inspect.signature(UCSDMosaicsDataset.__init__).parameters["hf_revision"].default is None
    assert HF_DATASETS == ("coralscapes_v2", "ucsd_mosaics")


def test_base_concept_layout_matches_demo_and_ignores_skipped_trains():
    data = yaml.safe_load(_BASE_CONFIG.read_text())["data"]
    frozen = configured_source_names(data)
    schema = ConceptSchema.from_csv(_REPO / "configs" / "class_to_concepts.csv", sources=frozen)
    demo = json.loads((_REPO / "demo" / "concept_id2name.json").read_text())
    assert {str(channel_id): name for channel_id, name in schema.channel_id2name().items()} == demo

    # Experiments 1 and 2 turn off every non-CoralNet train split. Train-only
    # datasets then leave the registry, and a schema built from what remains is
    # shorter. The freeze keeps reading the unmodified base config.
    shrunk = yaml.safe_load(_BASE_CONFIG.read_text())["data"]
    train_only = ("mermaid", "catlin_seaview", "moorea_labeled_corals", "ucsd_mosaics")
    for name in train_only:
        shrunk[name]["train"] = "None"
    reduced_sources = configured_source_names(shrunk)
    assert frozen - reduced_sources == set(train_only)
    assert reduced_sources == {
        "benthos_yuval",
        "coralnet",
        "coralscapes_v2",
        "pacific_labeled_corals",
    }
    reduced = ConceptSchema.from_csv(
        _REPO / "configs" / "class_to_concepts.csv", sources=reduced_sources
    )
    assert reduced.num_channels < schema.num_channels
    assert configured_source_names(data) == frozen
