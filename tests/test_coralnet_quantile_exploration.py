"""Quantile flags over CoralNet training sources. No cache or S3 access."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from scripts.coralnet_quantile_exploration import (
    IMAGES_FILE,
    METRIC_KEYS,
    OVERLAP,
    UNION,
    flag_sources,
    flagged_images,
    load_train_sources,
    main,
    quantile_thresholds,
    stats_from_csv,
    summarize,
)


def _stats(source_ids: list[int] | None = None, **columns: list[float]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    data = {key: [1] * n for key in METRIC_KEYS} | columns
    ids = source_ids or list(range(1, n + 1))
    return pd.DataFrame(data, index=pd.Index(ids, name="source_id"))


def _quantiles(**active: float) -> dict[str, float]:
    return {key: active.get(key, 0.0) for key in METRIC_KEYS}


def _run(stats: pd.DataFrame, **active: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantiles = _quantiles(**active)
    thresholds = quantile_thresholds(stats, quantiles)
    flags = flag_sources(stats, thresholds)
    return flags, summarize(stats, flags, quantiles, thresholds).set_index("criterion")


def _write_config(path, train, val=(), test=()):
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "coralnet": {
                        "train": {"whitelist_sources": list(train)},
                        "val": {"whitelist_sources": list(val)},
                        "test": {"whitelist_sources": list(test)},
                    },
                    "mermaid": {"train": {"annotation_date_cutoff": "2026-08-12"}},
                }
            }
        )
    )
    return path


def test_train_sources_come_only_from_the_coralnet_train_whitelist(tmp_path):
    config = _write_config(tmp_path / "cfg.yaml", train=[57, 23, 57], val=[1645], test=[3401])
    assert load_train_sources(config) == [23, 57]


@pytest.mark.parametrize(
    "train",
    [None, {"whitelist_sources": []}, {"whitelist_sources": [1], "class_subset": ["a"]}],
)
def test_train_configs_the_dataset_would_read_differently_are_rejected(tmp_path, train):
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"data": {"coralnet": {"train": train}}}))
    with pytest.raises(ValueError):
        load_train_sources(config)


def test_zero_quantile_turns_a_metric_off():
    flags, summary = _run(_stats(n_genera=[1, 2, 3], n_coral_genera=[3, 2, 1]), n_genera=0.4)
    assert list(flags.columns) == ["flag_n_genera", "flagged"]
    assert list(summary.index) == ["n_genera", UNION]


def test_all_zero_quantiles_flag_nothing():
    flags, summary = _run(_stats(n_genera=[1, 2, 3]))
    assert not flags["flagged"].any()
    assert summary.loc[UNION, "n_images"] == 0


def test_top_tail_without_ties():
    stats = _stats(n_coral_genera=list(range(10)))
    flags, summary = _run(stats, n_coral_genera=0.1)
    assert list(flags.index[flags["flagged"]]) == [10]
    assert summary.loc["n_coral_genera", "min_value"] == 9
    flags, _ = _run(stats, n_coral_genera=0.2)
    assert list(flags.index[flags["flagged"]]) == [9, 10]


def test_every_source_tied_at_the_threshold_is_flagged():
    flags, summary = _run(_stats(n_coral_genera=[0, 1, 2, 3, 4, 5, 6, 9, 9, 9]), n_coral_genera=0.1)
    assert list(flags.index[flags["flagged"]]) == [8, 9, 10]
    assert summary.loc["n_coral_genera", "pct_sources"] == pytest.approx(30.0)


def test_quantile_one_flags_every_source():
    flags, summary = _run(_stats(n_labels=[5, 1, 3]), n_labels=1.0)
    assert flags["flagged"].all()
    assert summary.loc[UNION, "pct_images"] == pytest.approx(100.0)


def test_union_counts_a_source_in_two_tails_once():
    stats = _stats(
        n_images=[10, 20, 30, 40],
        n_pixels=[100, 200, 300, 400],
        n_coral_genera=[0, 0, 5, 9],
        avg_pixels_per_image=[1.0, 2.0, 3.0, 50.0],
    )
    flags, summary = _run(stats, n_coral_genera=0.5, avg_pixels_per_image=0.25)
    assert list(flags.index[flags["flag_n_coral_genera"]]) == [3, 4]
    assert list(flags.index[flags["flag_avg_pixels_per_image"]]) == [4]

    assert summary.loc[OVERLAP, ["n_sources", "n_images", "n_pixels"]].tolist() == [1, 40, 400]
    union = summary.loc[UNION]
    assert [union.n_sources, union.n_images, union.n_pixels] == [2, 70, 700]
    assert union.pct_sources == pytest.approx(50.0)
    assert union.pct_images == pytest.approx(70.0)
    assert union.pct_pixels == pytest.approx(70.0)


def _stats_csv(path):
    stats = _stats(
        [23, 57, 69, 1645, 9999],
        n_images=[10, 20, 30, 1000, 500],
        n_pixels=[100, 200, 300, 10_000, 5000],
        n_coral_genera=[1, 2, 3, 80, 90],
    )
    stats.insert(0, "split", ["train", "train", "train", "val", "none"])
    stats.to_csv(path)
    return path


def test_stats_csv_keeps_only_training_sources(tmp_path):
    stats = stats_from_csv(_stats_csv(tmp_path / "source_stats.csv"), [23, 57, 69])
    assert list(stats.index) == [23, 57, 69]
    assert list(stats.columns) == list(METRIC_KEYS)

    flags, summary = _run(stats, n_coral_genera=0.34)
    assert list(flags.index[flags["flagged"]]) == [69]
    assert summary.loc[UNION, "pct_images"] == pytest.approx(50.0)


def test_stats_csv_missing_a_training_source_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="missing 1 of 3 training sources"):
        stats_from_csv(_stats_csv(tmp_path / "source_stats.csv"), [23, 57, 4242])


def test_flagged_images_lists_each_image_of_a_flagged_source_once():
    annotations = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 2, 2, 3],
            "image_id": ["a", "a", "b", "c", "c", "d"],
            "row": [0, 1, 2, 3, 4, 5],
            "image_key": pd.Categorical(["ka", "ka", None, "kc", "kc", "kd"]),
        }
    )
    flags, _ = _run(_stats(n_genera=[5, 1, 9]), n_genera=0.5)
    images = flagged_images(annotations, flags)
    pairs = images[["source_id", "image_id"]].to_numpy().tolist()
    assert pairs == [[1, "a"], [1, "b"], [3, "d"]]
    assert images["flagged"].all()
    assert list(images.columns) == [
        "source_id",
        "image_id",
        "image_key",
        "flag_n_genera",
        "flagged",
    ]


def test_cli_with_stats_csv_prints_and_writes_flags(tmp_path, capsys):
    config = _write_config(tmp_path / "cfg.yaml", train=[23, 57, 69], val=[1645])
    stats_csv = _stats_csv(tmp_path / "source_stats.csv")
    out = tmp_path / "out"
    out.mkdir()
    (out / IMAGES_FILE).write_text("stale")

    argv = ["--quantiles", "0", "0", "0", "0", "0.34", "0", "--data-config", str(config)]
    main([*argv, "--stats-csv", str(stats_csv), "--output-dir", str(out)])
    printed = capsys.readouterr().out
    assert "3 sources, 60 images, 600 annotated pixels" in printed
    assert UNION in printed

    flags = pd.read_csv(out / "source_flags.csv", index_col="source_id")
    assert flags.loc[flags["flagged"]].index.tolist() == [69]
    assert not (out / IMAGES_FILE).exists()
    run = json.loads((out / "run.json").read_text())
    assert run["summary"][-1]["criterion"] == UNION
    assert run["summary"][-1]["n_images"] == 30

    main([*argv, "--stats-csv", str(out / "source_flags.csv")])
    assert capsys.readouterr().out == printed
