"""Evaluation entrypoint: one loading path, through an exported bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch
import yaml

import scripts.evaluate as evaluate_script
from mermaidseg.evaluation.coralscapes_probe import load_legacy_probes
from mermaidseg.inference.bundle import load_bundle
from mermaidseg.io import REPO_ROOT
from tests.mermaidseg.conftest import deployable_lineage, export_ckpt, repo_with_origin
from tests.mermaidseg.evaluation.conftest import EvalCache, write_eval_cache
from tests.mermaidseg.inference import _tiny


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("deployable")
    repo, pushed = repo_with_origin(base)
    checkpoint, _ = _tiny.write_run(base / "run", checkpoint_lineage=deployable_lineage(pushed))
    return export_ckpt(checkpoint, tmp_path_factory.mktemp("out") / "bundle", repo_dir=repo)


@pytest.fixture
def cache(tmp_path: Path) -> EvalCache:
    return write_eval_cache(tmp_path / "data")


@pytest.fixture
def data_config(tmp_path: Path, cache: EvalCache) -> Path:
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.safe_dump({"data": cache.data}))
    return path


def _run_main_capturing_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bundle_dir: Path,
    cache: EvalCache,
    data_config: Path,
    *extra: str,
) -> Any:
    from mermaidseg.evaluation import coralscapes_probe

    seen: dict[str, Any] = {}

    def _probe(*, predictor, **_):
        seen["predictor"] = predictor
        return {}

    monkeypatch.setattr(coralscapes_probe, "evaluate_coralscapes39_probe", _probe)
    evaluate_script.main(
        [
            "--bundle",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(cache.cache_dir),
            "--data-config",
            str(data_config),
            "--evals",
            "coralscapes39",
            "--device",
            "cpu",
            *extra,
        ]
    )
    return seen["predictor"]


@pytest.mark.parametrize(("extra", "expected"), [([], None), (["--amp"], torch.bfloat16)])
def test_amp_dtype_is_resolved_from_the_bundle_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bundle_dir: Path,
    cache: EvalCache,
    data_config: Path,
    extra,
    expected,
):
    predictor = _run_main_capturing_predictor(
        monkeypatch, tmp_path, bundle_dir, cache, data_config, *extra
    )
    assert predictor.amp_dtype is expected


def test_concept_names_are_the_bundles_in_channel_order(bundle_dir: Path):
    bundle = load_bundle(bundle_dir, "cpu")
    assert evaluate_script.concept_names_of(bundle) == _tiny.CONCEPT_NAMES


def test_concept_names_follow_channel_id_order_not_insertion_order():
    bundle = SimpleNamespace(concept_id2name={10: "last", 2: "first", 3: "middle"})
    assert evaluate_script.concept_names_of(bundle) == ["first", "middle", "last"]


def test_main_passes_the_bundles_concept_mapping_path_resolved_against_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache: EvalCache, data_config: Path
):
    from mermaidseg.evaluation import coralscapes_probe

    mapping = "configs/not_the_default_mapping.csv"
    config = _tiny.resolved_config()
    config["training"] = {**config["training"], "concept_mapping_path": mapping}
    repo, pushed = repo_with_origin(tmp_path)
    checkpoint, _ = _tiny.write_run(
        tmp_path / "run",
        config=config,
        checkpoint_lineage=deployable_lineage(pushed, resolved_config=config),
    )
    bundle = export_ckpt(checkpoint, tmp_path / "bundle", repo_dir=repo)
    seen: dict[str, Any] = {}

    def _probe(*, taxonomy_csv, **_):
        seen["taxonomy_csv"] = taxonomy_csv
        return {}

    monkeypatch.setattr(coralscapes_probe, "evaluate_coralscapes_probe", _probe)
    monkeypatch.setattr(evaluate_script, "run_removed_concepts", lambda *_: [])
    # The tiny bundle has no reported binary concept channels; the eval stage is under test.
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", [])
    evaluate_script.main(
        [
            "--bundle",
            str(bundle),
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(cache.cache_dir),
            "--data-config",
            str(data_config),
            "--evals",
            "coralscapes",
            "--device",
            "cpu",
        ]
    )
    assert seen["taxonomy_csv"] == REPO_ROOT / mapping


@pytest.mark.parametrize(
    ("split", "groups"), [("train", ["s1", "s2"]), ("val", ["s3"]), ("test", ["s4"])]
)
def test_coralnet_groups_are_the_split_files_groups_of_the_split(cache, split, groups):
    assert evaluate_script.coralnet_groups(cache.section, split) == groups


def test_coralnet_groups_need_a_split_file(cache):
    data = {**cache.data, "datasets": {**cache.data["datasets"]}}
    data["datasets"]["coralnet"] = {
        k: v
        for k, v in data["datasets"]["coralnet"].items()
        if k not in ("split_file", "train", "test")
    } | {"val": {"groups": ["s3"]}}
    section = evaluate_script.DataSection.model_validate(data)
    with pytest.raises(ValueError, match="no split_file"):
        evaluate_script.coralnet_groups(section, "val")


def test_no_training_overlap_when_the_coralnet_entry_has_no_train_block(cache):
    data = {**cache.data, "datasets": {**cache.data["datasets"]}}
    data["datasets"]["coralnet"] = {
        k: v for k, v in data["datasets"]["coralnet"].items() if k != "train"
    }
    section = evaluate_script.DataSection.model_validate(data)
    assert evaluate_script.trained_on_scored_groups(section, ["s1", "s2"]) == []


def _with_coralscapes_train(cache: EvalCache, train: dict[str, Any] | None) -> dict[str, Any]:
    """The cache's data config with the coralscapes_v2 train block set, or removed for None."""
    entry = {k: v for k, v in cache.data["datasets"]["coralscapes_v2"].items() if k != "train"}
    if train is not None:
        entry["train"] = train
    return {**cache.data, "datasets": {**cache.data["datasets"], "coralscapes_v2": entry}}


def test_coralscapes_trained_on_scored_when_the_train_block_selects_every_group(cache):
    section = evaluate_script.DataSection.model_validate(
        _with_coralscapes_train(cache, {"groups": "all"})
    )
    assert evaluate_script.coralscapes_trained_on_scored(section, "val") is True


def test_coralscapes_not_trained_on_scored_when_the_entry_has_no_train_block(cache):
    section = evaluate_script.DataSection.model_validate(_with_coralscapes_train(cache, None))
    assert evaluate_script.coralscapes_trained_on_scored(section, "val") is False


def test_train_split_allows_only_coralnet_and_loss():
    check = evaluate_script.check_split_evals
    check("train", ["coralnet", "val_loss"])
    check("val", list(evaluate_script.ALL_EVALS))
    for other in ("pacific", "coralscapes", "coralscapes39"):
        with pytest.raises(ValueError, match="--split train supports only"):
            check("train", ["coralnet", other])


def test_train_loss_data_config_is_one_coralnet_val_block_over_the_groups():
    data = {
        "taxonomy": {"version": "t1", "manifest_sha256": "tt"},
        "default": {"train": {"transform": {"RandomResizedCrop": {"size": [256, 256]}}}},
        "datasets": {
            "coralnet": {
                "version": "v1",
                "manifest_sha256": "aa",
                "split_file": "configs/splits/coralnet.csv",
                "resize": {"max_edge": 2048},
                "train": {"use": ["train", "val"]},
                "test": {"use": ["test"]},
            },
            "mermaid": {"version": "v2", "manifest_sha256": "bb", "train": {"groups": "all"}},
        },
    }

    out = evaluate_script.coralnet_loss_data_config(data, ["s1", "s2"])

    assert out == {
        "data": {
            "datasets": {
                "coralnet": {
                    "version": "v1",
                    "manifest_sha256": "aa",
                    "resize": {"max_edge": 2048},
                    "val": {"groups": ["s1", "s2"]},
                }
            },
            "taxonomy": {"version": "t1", "manifest_sha256": "tt"},
        }
    }


_POOLED = {
    "class_accuracy": 0.5,
    "taxonomic": {
        "genus": {
            "acc_all": 0.8,
            "acc_living": 0.7,
            "macro_f1_all": 0.6,
            "macro_f1_living": 0.5,
            "f1_none": 0.4,
            "acc_coral": 0.3,
            "macro_f1_coral": 0.2,
            "n_coral": 9,
        }
    },
    "binary": {"macro_accuracy": 0.9, "macro_f1": 0.85},
}
_BLEACHED = {
    "accuracy": 0.1,
    "precision": 0.2,
    "recall": 0.3,
    "f1": 0.4,
    "n_true": 5,
    "n_false": 6,
    "n_not_given": 7,
}
_HIERARCHICAL = {
    "consistency/overall": 0.91,
    "severity/mean_normalized": 0.12,
    "ece/mean": 0.03,
}


def test_concept_lines_report_each_reported_binary_concept_then_the_hierarchy():
    section = {"bleached": _BLEACHED, "hierarchical": _HIERARCHICAL}

    assert evaluate_script._concept_lines(section) == [
        "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
        "(n_true=5 n_false=6 n_not_given=7)",
        "- hierarchical: consistency 0.9100, severity (mean normalized) 0.1200, ECE (mean) 0.0300",
    ]


def test_concept_lines_append_the_unannotated_count_when_present():
    section = {"anthropogenic": {**_BLEACHED, "n_unannotated": 11}}

    assert evaluate_script._concept_lines(section) == [
        "- anthropogenic acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
        "(n_true=5 n_false=6 n_not_given=7 n_unannotated=11)"
    ]


def test_concept_lines_skip_concepts_and_hierarchy_that_are_not_dicts():
    section = {"bleached": None, "anthropogenic": "n/a", "hierarchical": None}

    assert evaluate_script._concept_lines(section) == []


def test_point_lines_format_the_pooled_metrics_under_the_title():
    pooled = {**_POOLED, "bleached": _BLEACHED}

    assert evaluate_script._point_lines("Pooled title", pooled) == [
        "## Pooled title",
        "- class accuracy: 0.5000",
        "- taxonomic acc_all/acc_living: genus=0.8000/0.7000",
        "- taxonomic macro F1 all/living, none-class F1: genus=0.6000/0.5000, 0.4000",
        "- live coral taxonomic acc/macro F1: genus=0.3000/0.2000 (n=9)",
        "- binary macro acc/F1: 0.9000/0.8500",
        "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
        "(n_true=5 n_false=6 n_not_given=7)",
        "",
    ]


def test_point_lines_show_n_a_for_a_missing_binary_block_and_taxonomic_scores():
    pooled = {"class_accuracy": 0.5, "taxonomic": {"genus": {"acc_all": 0.8, "acc_living": 0.7}}}

    lines = evaluate_script._point_lines("T", pooled)

    assert "- binary macro acc/F1: n/a/n/a" in lines
    assert "- taxonomic macro F1 all/living, none-class F1: genus=n/a/n/a, n/a" in lines


_LOSS_SECTION = {
    "overall": {
        "loss": 1.0,
        "classification": 0.5,
        "multi_hot/bleached": 0.1,
        "genus": 0.2,
        "family": 0.3,
        "genus/accuracy": 0.9,
    },
    "per_dataset": {"coralnet": {"loss": 0.4, "multi_hot/bleached": 0.25, "other": 5.0}},
}


def test_val_loss_lines_list_headline_keys_then_ranks_then_each_dataset():
    assert evaluate_script._val_loss_lines(_LOSS_SECTION) == [
        "## Validation loss (overall)",
        "- loss: 1.0000",
        "- classification: 0.5000",
        "- multi_hot/bleached: 0.1000",
        "- ranks: genus=0.2000, family=0.3000",
        "- coralnet: loss=0.4000, multi_hot/bleached=0.2500",
        "",
    ]


def test_val_loss_lines_use_the_given_title_and_tolerate_an_empty_section():
    assert evaluate_script._val_loss_lines({}, "Custom") == ["## Custom (overall)", ""]


def test_write_summary_writes_the_summary_as_json(tmp_path):
    summary = {"bundle": "b", "split": "val", "coralnet_val": {"error": "X: y"}}

    evaluate_script._write_summary(tmp_path, summary)

    assert json.loads((tmp_path / "summary.json").read_text()) == summary


@pytest.mark.parametrize(
    ("key", "section", "expected_lines"),
    [
        (
            "coralnet_val",
            {"error": "RuntimeError: boom"},
            ["## coralnet_val: ERROR - RuntimeError: boom"],
        ),
        (
            "coralnet_train",
            {"pooled": _POOLED},
            ["## CoralNet (pooled over train sources)", "- class accuracy: 0.5000"],
        ),
        (
            "pacific",
            {"pooled": _POOLED},
            ["## Pacific Labeled Corals (pooled over regions)", "- class accuracy: 0.5000"],
        ),
        ("val_loss", _LOSS_SECTION, ["## Validation loss (overall)", "- loss: 1.0000"]),
        (
            "train_loss",
            _LOSS_SECTION,
            ["## CoralNet train-source loss (overall)", "- loss: 1.0000"],
        ),
        (
            "coralscapes39_test",
            {
                "by_k": {
                    "50": {
                        "test": {"accuracy": 0.7, "miou": 0.4},
                        "probe": {"train_accuracy": 0.95, "num_points": 1950},
                    }
                }
            },
            [
                "## CoralscapesV2 39-class linear probe (test)",
                "- k=50: test accuracy 0.7000, mIoU 0.4000 (probe train acc 0.9500, 1950 pixels)",
            ],
        ),
        (
            "coralscapes_val",
            {
                "feature_kinds": {
                    "concepts": {
                        "val": {"accuracy": 0.6, "miou": 0.3},
                        "probe": {"train_accuracy": 0.9},
                    }
                },
                "bleached": _BLEACHED,
            },
            [
                "## CoralscapesV2 95-class linear probe (val)",
                "- concepts: val accuracy 0.6000, mIoU 0.3000 (probe train acc 0.9000)",
                "- bleached acc/precision/recall/F1: 0.1000/0.2000/0.3000/0.4000 "
                "(n_true=5 n_false=6 n_not_given=7)",
            ],
        ),
    ],
)
def test_write_summary_renders_each_section_kind_under_its_own_heading(
    tmp_path, key, section, expected_lines
):
    evaluate_script._write_summary(tmp_path, {key: section})

    lines = (tmp_path / "summary.md").read_text().splitlines()
    assert lines[:2] == ["# CBM evaluation summary", ""]
    for expected in expected_lines:
        assert expected in lines


def test_write_summary_leaves_scalar_entries_out_of_the_markdown(tmp_path):
    summary = {"bundle": "outputs/b", "input_size": [64, 64], "num_removed_concepts": 3}

    evaluate_script._write_summary(tmp_path, summary)

    assert (tmp_path / "summary.md").read_text() == "# CBM evaluation summary\n"


def test_failed_sections_lists_only_the_sections_with_an_error():
    summary = {
        "bundle": "b",
        "coralnet_val": {"error": "boom"},
        "pacific": {"pooled": {}},
        "val_loss": {"error": "bang"},
    }

    assert evaluate_script.failed_sections(summary) == ["coralnet_val", "val_loss"]


# --- main: wiring of the eval stages, with the package evaluators replaced by recorders ---------

_POOLED_RESULT = {"pooled": {"class_accuracy": 0.5, "taxonomic": {}}}


class _Calls:
    """Keyword arguments each stubbed evaluator received, by evaluator name."""

    def __init__(self) -> None:
        self.by_name: dict[str, list[dict[str, Any]]] = {}

    def record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.by_name.setdefault(name, []).append(kwargs)

    def one(self, name: str) -> dict[str, Any]:
        assert len(self.by_name.get(name, [])) == 1, name
        return self.by_name[name][0]

    def never(self, name: str) -> bool:
        return name not in self.by_name


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Replace every package evaluator; the readers are built over the tiny cache."""
    from mermaidseg.evaluation import (
        coralnet_eval,
        coralscapes_probe,
        pacific_eval,
        val_loss,
    )

    recorded = _Calls()

    def _stub(name: str, result: Any):
        def _fn(**kwargs: Any) -> Any:
            recorded.record(name, kwargs)
            return result

        return _fn

    monkeypatch.setattr(coralnet_eval, "evaluate_coralnet", _stub("coralnet", _POOLED_RESULT))
    monkeypatch.setattr(pacific_eval, "evaluate_pacific", _stub("pacific", _POOLED_RESULT))
    monkeypatch.setattr(
        coralscapes_probe,
        "evaluate_coralscapes_probe",
        _stub("coralscapes", {"val": {"feature_kinds": {}}, "train": {"feature_kinds": {}}}),
    )
    monkeypatch.setattr(
        coralscapes_probe,
        "evaluate_coralscapes39_probe",
        _stub("coralscapes39", {"val": {"by_k": {}}}),
    )
    monkeypatch.setattr(
        val_loss, "evaluate_val_loss", _stub("val_loss", {"overall": {"loss": 1.0}})
    )
    write_sample = coralnet_eval.write_coralnet_sample

    def _write_sample(**kwargs: Any) -> Any:
        recorded.record("sample", kwargs)
        return write_sample(**kwargs)

    monkeypatch.setattr(coralnet_eval, "write_coralnet_sample", _write_sample)
    monkeypatch.setattr(evaluate_script, "load_benthic_hierarchy", lambda *_: {"hier": 1})
    monkeypatch.setattr(evaluate_script, "run_removed_concepts", lambda *_: ["removed__a"])
    # The tiny bundle has no reported binary concept channels; the eval stages are under test.
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", [])
    return recorded


def _main(
    bundle_dir: Path, tmp_path: Path, cache: EvalCache, data_config: Path, *extra: str
) -> dict[str, Any]:
    """Run ``main`` on the tiny bundle and return the parsed ``summary.json``."""
    evaluate_script.main(
        [
            "--bundle",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(cache.cache_dir),
            "--data-config",
            str(data_config),
            "--device",
            "cpu",
            *extra,
        ]
    )
    return json.loads((tmp_path / "out" / "summary.json").read_text())


def _groups(reader: Any) -> list[str]:
    return sorted(reader.images["group"].unique())


def test_main_runs_each_requested_eval_once_even_when_listed_twice(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _main(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        "--evals",
        "coralscapes39",
        "pacific",
        "coralscapes39",
    )

    assert summary["evals"] == ["coralscapes39", "pacific"]
    calls.one("coralscapes39")
    calls.one("pacific")
    assert calls.never("coralnet")


def test_main_skips_the_hierarchy_unless_coralnet_or_pacific_run(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    monkeypatch.setattr(
        evaluate_script,
        "load_benthic_hierarchy",
        lambda *_: pytest.fail("hierarchy is only for coralnet and pacific"),
    )

    _main(bundle_dir, tmp_path, cache, data_config, "--evals", "coralscapes39", "val_loss")


@pytest.mark.parametrize(
    ("split", "groups"), [("val", ["s3"]), ("test", ["s4"]), ("train", ["s1", "s2"])]
)
def test_main_scores_coralnet_on_exactly_the_split_files_groups_of_the_split(
    calls, bundle_dir, tmp_path, cache, data_config, split, groups
):
    summary = _main(
        bundle_dir, tmp_path, cache, data_config, "--evals", "coralnet", "--split", split
    )

    kwargs = calls.one("coralnet")
    assert _groups(kwargs["reader"]) == groups
    assert kwargs["split"] == split
    assert kwargs["removed_concepts"] == ["removed__a"]
    assert kwargs["hierarchy"] == {"hier": 1}
    assert kwargs["concept_names"] == _tiny.CONCEPT_NAMES
    assert summary[f"coralnet_{split}"] == _POOLED_RESULT
    assert summary["num_removed_concepts"] == 1


def test_main_warns_and_records_when_the_scored_coralnet_split_is_trained_on(
    calls, bundle_dir, tmp_path, cache, data_config, caplog
):
    with caplog.at_level("WARNING", logger="evaluate"):
        summary = _main(
            bundle_dir, tmp_path, cache, data_config, "--evals", "coralnet", "--split", "val"
        )

    assert summary["coralnet_split_in_training"] is True
    assert any("train" in r.getMessage() and "s3" in r.getMessage() for r in caplog.records)


def test_main_records_no_training_overlap_for_a_split_the_config_does_not_train_on(
    calls, bundle_dir, tmp_path, cache, data_config, caplog
):
    with caplog.at_level("WARNING", logger="evaluate"):
        summary = _main(
            bundle_dir, tmp_path, cache, data_config, "--evals", "coralnet", "--split", "test"
        )

    assert summary["coralnet_split_in_training"] is False
    assert not [
        r for r in caplog.records if r.levelname == "WARNING" and "trains on" in r.getMessage()
    ]


def test_main_scores_pacific_on_the_val_block_with_its_annotators(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _main(bundle_dir, tmp_path, cache, data_config, "--evals", "pacific")

    reader = calls.one("pacific")["reader"]
    assert _groups(reader) == ["heron_reef", "line_islands"]
    assert sorted(reader.points["annotator"]) == ["archived", "host"]


def test_main_scores_the_coralscapes_probes_on_the_pinned_version(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _main(bundle_dir, tmp_path, cache, data_config, "--evals", "coralscapes", "coralscapes39")

    pin = cache.data["datasets"]["coralscapes_v2"]["version"]
    for name in ("coralscapes", "coralscapes39"):
        manifest = calls.one(name)["version"].manifest
        assert (manifest.dataset, manifest.version) == ("coralscapes_v2", pin)


def test_main_images_per_source_hands_the_seeded_sample_to_the_coralnet_reader(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _main(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        "--evals",
        "coralnet",
        "--split",
        "train",
        "--images-per-source",
        "2",
        "--sample-seed",
        "7",
    )

    sample = calls.one("sample")
    expected_path = tmp_path / "out" / "coralnet_train_sample.parquet"
    assert (sample["path"], sample["images_per_source"], sample["seed"]) == (expected_path, 2, 7)
    assert _groups(sample["reader"]) == ["s1", "s2"]
    sampled = pd.read_parquet(expected_path)["image_id"].tolist()
    assert len(sampled) == 3 and "c2a" in sampled
    reader = calls.one("coralnet")["reader"]
    assert sorted(reader.images["image_id"]) == sorted(sampled)
    assert summary["images_per_source"] == 2
    assert summary["sample_seed"] == 7
    assert summary["sample"] == str(expected_path)


def test_main_draws_no_sample_without_images_per_source(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _main(bundle_dir, tmp_path, cache, data_config, "--evals", "coralnet")

    assert calls.never("sample")
    assert "sample" not in summary


def test_main_draws_no_sample_when_no_eval_uses_it(calls, bundle_dir, tmp_path, cache, data_config):
    _main(
        bundle_dir, tmp_path, cache, data_config, "--evals", "pacific", "--images-per-source", "5"
    )

    assert calls.never("sample")


def test_main_val_loss_on_val_uses_the_data_config_cache_and_the_max_images_flag(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _main(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        "--evals",
        "val_loss",
        "--val-loss-max-images",
        "17",
    )

    kwargs = calls.one("val_loss")
    assert kwargs["data_config"] == data_config
    assert kwargs["cache_dir"] == cache.cache_dir
    assert kwargs["image_ids"] is None
    assert kwargs["name"] == "val_loss"
    assert kwargs["apply_source_mask"] is False
    assert kwargs["max_images_per_dataset"] == 17
    assert "val_loss" in summary
    assert "train_loss" not in summary


def test_main_val_loss_on_train_scores_the_coralnet_train_groups_under_the_source_mask(
    calls, bundle_dir, tmp_path, cache, data_config
):
    summary = _main(
        bundle_dir, tmp_path, cache, data_config, "--evals", "val_loss", "--split", "train"
    )

    out = tmp_path / "out"
    kwargs = calls.one("val_loss")
    assert kwargs["data_config"] == out / "data_config_coralnet_train.yaml"
    assert kwargs["name"] == "train_loss"
    assert kwargs["apply_source_mask"] is True
    assert kwargs["max_images_per_dataset"] == 1000
    assert kwargs["image_ids"] is None
    assert "train_loss" in summary
    written = yaml.safe_load((out / "data_config_coralnet_train.yaml").read_text())
    coralnet = cache.data["datasets"]["coralnet"]
    assert written == {
        "data": {
            "datasets": {
                "coralnet": {
                    "version": coralnet["version"],
                    "manifest_sha256": coralnet["manifest_sha256"],
                    "val": {"groups": ["s1", "s2"]},
                }
            }
        }
    }


def test_main_train_loss_on_a_sample_is_uncapped_and_restricted_to_the_sample(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _main(
        bundle_dir,
        tmp_path,
        cache,
        data_config,
        "--evals",
        "val_loss",
        "--split",
        "train",
        "--images-per-source",
        "5",
    )

    kwargs = calls.one("val_loss")
    assert kwargs["max_images_per_dataset"] is None
    sampled = pd.read_parquet(tmp_path / "out" / "coralnet_train_sample.parquet")
    assert kwargs["image_ids"] == {("coralnet", "val"): sampled["image_id"].tolist()}


def test_main_records_a_failed_eval_and_runs_the_remaining_ones(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    from mermaidseg.evaluation import coralnet_eval

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(coralnet_eval, "evaluate_coralnet", _boom)

    summary = _main(bundle_dir, tmp_path, cache, data_config, "--evals", "coralnet", "pacific")

    assert summary["coralnet_val"] == {"error": "RuntimeError: boom"}
    assert summary["pacific"] == _POOLED_RESULT
    markdown = (tmp_path / "out" / "summary.md").read_text()
    assert "## coralnet_val: ERROR - RuntimeError: boom" in markdown


def test_main_exits_zero_after_a_failed_eval_without_strict(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    from mermaidseg.evaluation import val_loss

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(val_loss, "evaluate_val_loss", _boom)

    summary = _main(bundle_dir, tmp_path, cache, data_config, "--evals", "val_loss")

    assert summary["val_loss"] == {"error": "RuntimeError: boom"}


def test_main_strict_exits_1_after_writing_the_summary_of_a_failed_eval(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config
):
    from mermaidseg.evaluation import val_loss

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(val_loss, "evaluate_val_loss", _boom)

    with pytest.raises(SystemExit) as exit_info:
        _main(bundle_dir, tmp_path, cache, data_config, "--evals", "val_loss", "--strict")

    assert exit_info.value.code == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["val_loss"] == {"error": "RuntimeError: boom"}


def test_main_strict_does_not_exit_when_every_eval_succeeds(
    calls, bundle_dir, tmp_path, cache, data_config
):
    _main(bundle_dir, tmp_path, cache, data_config, "--evals", "val_loss", "--strict")


def test_main_rejects_a_concept_name_count_that_differs_from_the_models_channels(
    calls, bundle_dir, tmp_path, cache, data_config
):
    bundle = tmp_path / "bundle_extra_concept"
    shutil.copytree(bundle_dir, bundle)
    model_json = bundle / evaluate_script.MODEL_JSON
    metadata = json.loads(model_json.read_text())
    metadata["concept_id2name"]["999"] = "phantom"
    model_json.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="concept names but the model has"):
        _main(bundle, tmp_path, cache, data_config, "--evals", "val_loss")
    assert calls.never("val_loss")


@pytest.mark.parametrize("name", ["coralnet", "pacific", "coralscapes"])
def test_main_requires_the_reported_binary_concepts_for_the_concept_evals(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config, name
):
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", ["bleached", "anthropogenic"])

    with pytest.raises(RuntimeError, match=r"\['anthropogenic'\]"):
        _main(bundle_dir, tmp_path, cache, data_config, "--evals", name)
    assert calls.never(name)


@pytest.mark.parametrize("name", ["coralscapes39", "val_loss"])
def test_main_does_not_require_the_reported_binary_concepts_for_the_other_evals(
    monkeypatch, calls, bundle_dir, tmp_path, cache, data_config, name
):
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", ["bleached", "anthropogenic"])

    _main(bundle_dir, tmp_path, cache, data_config, "--evals", name)

    calls.one(name)


def _parse_defaults() -> Any:
    return evaluate_script.build_parser().parse_args(
        ["--bundle", "b", "--output-dir", "out", "--cache-dir", "c"]
    )


def test_default_probe_points_file_loads():
    path = Path(_parse_defaults().probe_points)

    probe = json.loads(path.read_text())

    assert probe["num_classes"] == 95
    assert probe["images"]


def test_default_coralscapes39_probe_file_loads():
    path = _parse_defaults().coralscapes39_probe

    probes = load_legacy_probes([path])

    assert 50 in probes


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "coralscapes_train_config.yaml"
    path.write_text(yaml.safe_dump({"data": data}))
    return path


def test_main_warns_and_records_when_the_scored_coralscapes_split_is_trained_on(
    calls, bundle_dir, tmp_path, cache, caplog
):
    config = _write_config(
        tmp_path, _with_coralscapes_train(cache, {"groups": ["train", "validation"]})
    )
    with caplog.at_level("WARNING", logger="evaluate"):
        summary = _main(
            bundle_dir, tmp_path, cache, config, "--evals", "coralscapes", "--split", "val"
        )

    assert summary["coralscapes_split_in_training"] is True
    assert any(
        "trains on" in r.getMessage() and "validation" in r.getMessage() for r in caplog.records
    )


def test_main_records_no_coralscapes_overlap_for_a_split_the_config_does_not_train_on(
    calls, bundle_dir, tmp_path, cache, caplog
):
    config = _write_config(
        tmp_path, _with_coralscapes_train(cache, {"groups": ["train", "validation"]})
    )
    with caplog.at_level("WARNING", logger="evaluate"):
        summary = _main(
            bundle_dir, tmp_path, cache, config, "--evals", "coralscapes39", "--split", "test"
        )

    assert summary["coralscapes_split_in_training"] is False
    assert not [
        r for r in caplog.records if r.levelname == "WARNING" and "trains on" in r.getMessage()
    ]


def test_main_records_a_coralscapes_error_when_the_config_has_no_coralscapes_entry(
    calls, bundle_dir, tmp_path, cache
):
    datasets = {k: v for k, v in cache.data["datasets"].items() if k != "coralscapes_v2"}
    config = _write_config(tmp_path, {**cache.data, "datasets": datasets})

    summary = _main(
        bundle_dir,
        tmp_path,
        cache,
        config,
        "--evals",
        "coralnet",
        "coralscapes",
        "--split",
        "val",
    )

    assert "error" not in summary["coralnet_val"]
    assert "error" in summary["coralscapes_val"]
    assert "coralscapes_split_in_training" not in summary
