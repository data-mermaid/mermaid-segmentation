"""Evaluation entrypoint: one loading path, through an exported bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

import scripts.evaluate as evaluate_script
from mermaidseg.evaluation.coralscapes_probe import load_legacy_probes
from mermaidseg.export import export_bundle
from mermaidseg.inference.bundle import load_bundle
from mermaidseg.io import REPO_ROOT
from tests.mermaidseg.inference import _tiny

DATA = {
    "coralnet": {
        "train": {"whitelist_sources": [1, 2]},
        "val": {"whitelist_sources": [3], "image_blacklist": "/bl.parquet"},
        "test": {
            "whitelist_sources": [4, 5],
            "image_blacklist": "/bl.parquet",
            "annotation_overrides": "/ov.parquet",
        },
    },
    "benthos_yuval": {"train": None, "val": {"whitelist_sites": ["RS24"]}},
}


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint, _ = _tiny.write_run(tmp_path_factory.mktemp("run"))
    return export_bundle(checkpoint, tmp_path_factory.mktemp("out") / "bundle")


def _run_main_capturing_predictor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path, *extra: str
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bundle_dir: Path, extra, expected
):
    predictor = _run_main_capturing_predictor(monkeypatch, tmp_path, bundle_dir, *extra)
    assert predictor.amp_dtype is expected


def test_concept_names_are_the_bundles_in_channel_order(bundle_dir: Path):
    bundle = load_bundle(bundle_dir, "cpu")
    assert evaluate_script.concept_names_of(bundle) == _tiny.CONCEPT_NAMES


def test_concept_names_follow_channel_id_order_not_insertion_order():
    bundle = SimpleNamespace(concept_id2name={10: "last", 2: "first", 3: "middle"})
    assert evaluate_script.concept_names_of(bundle) == ["first", "middle", "last"]


def test_main_passes_the_bundles_concept_mapping_path_resolved_against_the_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from mermaidseg.evaluation import coralscapes_probe

    mapping = "configs/not_the_default_mapping.csv"
    config = _tiny.resolved_config()
    config["training"] = {**config["training"], "concept_mapping_path": mapping}
    lineage = {"git_sha": "abc123", "git_dirty": False, "resolved_config": config}
    checkpoint, _ = _tiny.write_run(tmp_path / "run", config=config, checkpoint_lineage=lineage)
    bundle = export_bundle(checkpoint, tmp_path / "bundle")
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
            "--evals",
            "coralscapes",
            "--device",
            "cpu",
        ]
    )
    assert seen["taxonomy_csv"] == REPO_ROOT / mapping


def test_split_field():
    f = evaluate_script._split_field
    assert f(DATA, "coralnet", "val", "whitelist_sources") == [3]
    assert f(DATA, "coralnet", "test", "whitelist_sources") == [4, 5]
    assert f(DATA, "coralnet", "test", "annotation_overrides") == "/ov.parquet"
    assert f(DATA, "coralnet", "val", "annotation_overrides") is None
    assert f(DATA, "benthos_yuval", "val", "whitelist_sites") == ["RS24"]
    assert f(DATA, "benthos_yuval", "train", "whitelist_sites") is None
    assert f(DATA, "missing", "val", "whitelist_sources") is None


def test_coralnet_split_without_sources_fails_early():
    data = {"coralnet": {"val": {"whitelist_sources": [3]}}}
    with pytest.raises(ValueError, match=r"no coralnet\.test\.whitelist_sources"):
        evaluate_script._coralnet_sources(data, "test")


def test_coralnet_manifest_returns_the_configured_key():
    data = {"coralnet": {"val": {"manifest_path": "etl/runA/m.parquet"}}}
    assert evaluate_script._coralnet_manifest(data, "val") == "etl/runA/m.parquet"


@pytest.mark.parametrize(
    "data",
    [
        {"coralnet": {"val": {"whitelist_sources": [3]}}},
        {"coralnet": {"val": None}},
    ],
)
def test_coralnet_split_without_manifest_fails_early(data):
    with pytest.raises(ValueError, match=r"coralnet\.val\.manifest_path"):
        evaluate_script._coralnet_manifest(data, "val")


def test_train_split_reads_the_train_sources():
    assert "train" in evaluate_script.SPLITS
    assert evaluate_script._coralnet_sources(DATA, "train") == [1, 2]


def test_train_split_allows_only_coralnet_and_loss():
    check = evaluate_script.check_split_evals
    check("train", ["coralnet", "val_loss"])
    check("val", list(evaluate_script.ALL_EVALS))
    for other in ("pacific", "coralscapes", "coralscapes39"):
        with pytest.raises(ValueError, match="--split train supports only"):
            check("train", ["coralnet", other])


def test_train_loss_data_config_is_coralnet_train_without_a_transform(tmp_path):
    data = {
        "local_cache_dir": "/cache",
        "default": {
            "train": {"transform": {"RandomResizedCrop": {"size": [256, 256]}}},
            "val": {},
        },
        "coralnet": {
            "train": {
                "whitelist_sources": [1, 2],
                "image_blacklist": "/bl.parquet",
                "transform": {"Resize": {"height": 8}},
            },
            "val": None,
        },
        "mermaid": {"train": {"annotation_date_cutoff": "2026-08-12"}},
    }
    keep = tmp_path / "keep.parquet"
    out = evaluate_script.coralnet_loss_data_config(data, "train", keep)
    assert out == {
        "data": {
            "local_cache_dir": "/cache",
            "coralnet": {
                "val": {
                    "whitelist_sources": [1, 2],
                    "image_blacklist": "/bl.parquet",
                    "image_keep": str(keep.resolve()),
                }
            },
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


def test_loss_data_config_without_a_sample_has_no_image_keep():
    data = {"coralnet": {"train": {"whitelist_sources": [1], "transform": {"Resize": {}}}}}

    out = evaluate_script.coralnet_loss_data_config(data, "train", None)

    assert out == {"data": {"coralnet": {"val": {"whitelist_sources": [1]}}}}


def test_loss_data_config_rejects_a_split_that_is_not_configured():
    data = {"coralnet": {"train": {"whitelist_sources": [1]}, "val": None}}

    with pytest.raises(ValueError, match=r"no coralnet\.val"):
        evaluate_script.coralnet_loss_data_config(data, "val", None)


# --- main: wiring of the eval stages, with the package evaluators replaced by recorders ---------

_DATA_YAML = {
    "data": {
        "local_cache_write_through": True,
        "coralnet": {
            "train": {
                "whitelist_sources": [1, 2],
                "manifest_path": "etl/runA/m.parquet",
                "image_blacklist": "/bl.parquet",
                "annotation_overrides": "/ov.parquet",
            },
            "val": {"whitelist_sources": [3], "manifest_path": "etl/runA/m.parquet"},
        },
        "pacific_labeled_corals": {"val": {"whitelist_sites": ["RS24"]}},
    }
}
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
    """Replace every package evaluator and the heavyweight setup main performs."""
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

    def _write_sample(**kwargs: Any) -> Any:
        recorded.record("sample", kwargs)
        return kwargs["path"]

    monkeypatch.setattr(coralnet_eval, "write_coralnet_sample", _write_sample)
    monkeypatch.setattr(evaluate_script, "load_benthic_hierarchy", lambda **_: {"hier": 1})
    monkeypatch.setattr(evaluate_script, "run_removed_concepts", lambda *_: ["removed__a"])
    # The tiny bundle has no reported binary concept channels; the eval stages are under test.
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", [])
    monkeypatch.setattr(
        evaluate_script, "setup_local_cache", lambda data: recorded.record("cache", dict(data))
    )
    return recorded


@pytest.fixture
def data_config(tmp_path: Path) -> Path:
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.safe_dump(_DATA_YAML))
    return path


def _main(bundle_dir: Path, tmp_path: Path, data_config: Path, *extra: str) -> dict[str, Any]:
    """Run ``main`` on the tiny bundle and return the parsed ``summary.json``."""
    evaluate_script.main(
        [
            "--bundle",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--data-config",
            str(data_config),
            "--device",
            "cpu",
            *extra,
        ]
    )
    return json.loads((tmp_path / "out" / "summary.json").read_text())


def test_main_runs_each_requested_eval_once_even_when_listed_twice(
    calls, bundle_dir, tmp_path, data_config
):
    summary = _main(
        bundle_dir, tmp_path, data_config, "--evals", "coralscapes39", "pacific", "coralscapes39"
    )

    assert summary["evals"] == ["coralscapes39", "pacific"]
    calls.one("coralscapes39")
    calls.one("pacific")
    assert calls.never("coralnet")


def test_main_skips_the_hierarchy_unless_coralnet_or_pacific_run(
    monkeypatch, calls, bundle_dir, tmp_path, data_config
):
    monkeypatch.setattr(
        evaluate_script,
        "load_benthic_hierarchy",
        lambda **_: pytest.fail("hierarchy is only for coralnet and pacific"),
    )

    _main(bundle_dir, tmp_path, data_config, "--evals", "coralscapes39", "val_loss")


def test_main_scores_coralnet_on_the_split_sources_with_the_removed_concepts(
    calls, bundle_dir, tmp_path, data_config
):
    summary = _main(bundle_dir, tmp_path, data_config, "--evals", "coralnet", "--split", "train")

    kwargs = calls.one("coralnet")
    assert kwargs["whitelist_sources"] == [1, 2]
    assert kwargs["image_blacklist"] == "/bl.parquet"
    assert kwargs["annotation_overrides"] == "/ov.parquet"
    assert kwargs["split"] == "train"
    assert kwargs["removed_concepts"] == ["removed__a"]
    assert kwargs["hierarchy"] == {"hier": 1}
    assert kwargs["image_keep"] is None
    assert kwargs["concept_names"] == _tiny.CONCEPT_NAMES
    assert summary["coralnet_train"] == _POOLED_RESULT
    assert summary["num_removed_concepts"] == 1


def test_main_scores_pacific_on_the_configured_validation_sites(
    calls, bundle_dir, tmp_path, data_config
):
    _main(bundle_dir, tmp_path, data_config, "--evals", "pacific")

    kwargs = calls.one("pacific")
    assert kwargs["whitelist_sites"] == ["RS24"]
    assert kwargs["whitelist_subsets"] is None


def test_main_cache_dir_flag_overrides_the_data_configs_cache_root(
    calls, bundle_dir, tmp_path, data_config
):
    _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss", "--cache-dir", "/the/cache")

    assert calls.one("cache")["local_cache_dir"] == "/the/cache"
    assert calls.one("val_loss")["cache_dir"] == "/the/cache"


def test_main_leaves_the_cache_root_of_the_data_config_without_a_cache_dir_flag(
    calls, bundle_dir, tmp_path, data_config
):
    _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss")

    assert "local_cache_dir" not in calls.one("cache")
    assert calls.one("val_loss")["cache_dir"] is None


def test_main_images_per_source_samples_once_and_shares_it_with_the_coralnet_eval(
    calls, bundle_dir, tmp_path, data_config
):
    summary = _main(
        bundle_dir,
        tmp_path,
        data_config,
        "--evals",
        "coralnet",
        "--split",
        "train",
        "--images-per-source",
        "5",
        "--sample-seed",
        "7",
    )

    sample = calls.one("sample")
    expected_path = tmp_path / "out" / "coralnet_train_sample.parquet"
    assert sample == {
        "path": expected_path,
        "manifest_path": "etl/runA/m.parquet",
        "whitelist_sources": [1, 2],
        "image_blacklist": "/bl.parquet",
        "annotation_overrides": "/ov.parquet",
        "images_per_source": 5,
        "seed": 7,
    }
    assert calls.one("coralnet")["image_keep"] == expected_path
    assert summary["images_per_source"] == 5
    assert summary["sample_seed"] == 7
    assert summary["sample"] == str(expected_path)


def test_main_draws_no_sample_without_images_per_source(calls, bundle_dir, tmp_path, data_config):
    summary = _main(bundle_dir, tmp_path, data_config, "--evals", "coralnet")

    assert calls.never("sample")
    assert "sample" not in summary


def test_main_draws_no_sample_when_no_eval_uses_it(calls, bundle_dir, tmp_path, data_config):
    _main(bundle_dir, tmp_path, data_config, "--evals", "pacific", "--images-per-source", "5")

    assert calls.never("sample")


def test_main_val_loss_on_val_uses_the_data_config_and_the_max_images_flag(
    calls, bundle_dir, tmp_path, data_config
):
    summary = _main(
        bundle_dir, tmp_path, data_config, "--evals", "val_loss", "--val-loss-max-images", "17"
    )

    kwargs = calls.one("val_loss")
    assert kwargs["data_config"] == data_config
    assert kwargs["name"] == "val_loss"
    assert kwargs["apply_source_mask"] is False
    assert kwargs["max_images_per_dataset"] == 17
    assert "val_loss" in summary
    assert "train_loss" not in summary


def test_main_val_loss_on_train_scores_the_coralnet_train_sources_under_the_source_mask(
    calls, bundle_dir, tmp_path, data_config
):
    summary = _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss", "--split", "train")

    out = tmp_path / "out"
    kwargs = calls.one("val_loss")
    assert kwargs["data_config"] == out / "data_config_coralnet_train.yaml"
    assert kwargs["name"] == "train_loss"
    assert kwargs["apply_source_mask"] is True
    assert kwargs["max_images_per_dataset"] == 1000
    assert "train_loss" in summary
    written = yaml.safe_load((out / "data_config_coralnet_train.yaml").read_text())
    assert written == {
        "data": {
            "local_cache_write_through": True,
            "coralnet": {"val": _DATA_YAML["data"]["coralnet"]["train"]},
        }
    }


def test_main_train_loss_on_a_sample_is_uncapped_and_restricted_to_the_sample(
    calls, bundle_dir, tmp_path, data_config
):
    _main(
        bundle_dir,
        tmp_path,
        data_config,
        "--evals",
        "val_loss",
        "--split",
        "train",
        "--images-per-source",
        "5",
    )

    out = tmp_path / "out"
    assert calls.one("val_loss")["max_images_per_dataset"] is None
    written = yaml.safe_load((out / "data_config_coralnet_train.yaml").read_text())
    keep = written["data"]["coralnet"]["val"]["image_keep"]
    assert keep == str((out / "coralnet_train_sample.parquet").resolve())


def test_main_records_a_failed_eval_and_runs_the_remaining_ones(
    monkeypatch, calls, bundle_dir, tmp_path, data_config
):
    from mermaidseg.evaluation import coralnet_eval

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(coralnet_eval, "evaluate_coralnet", _boom)

    summary = _main(bundle_dir, tmp_path, data_config, "--evals", "coralnet", "pacific")

    assert summary["coralnet_val"] == {"error": "RuntimeError: boom"}
    assert summary["pacific"] == _POOLED_RESULT
    markdown = (tmp_path / "out" / "summary.md").read_text()
    assert "## coralnet_val: ERROR - RuntimeError: boom" in markdown


def test_main_exits_zero_after_a_failed_eval_without_strict(
    monkeypatch, calls, bundle_dir, tmp_path, data_config
):
    from mermaidseg.evaluation import val_loss

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(val_loss, "evaluate_val_loss", _boom)

    summary = _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss")

    assert summary["val_loss"] == {"error": "RuntimeError: boom"}


def test_main_strict_exits_1_after_writing_the_summary_of_a_failed_eval(
    monkeypatch, calls, bundle_dir, tmp_path, data_config
):
    from mermaidseg.evaluation import val_loss

    def _boom(**_):
        raise RuntimeError("boom")

    monkeypatch.setattr(val_loss, "evaluate_val_loss", _boom)

    with pytest.raises(SystemExit) as exit_info:
        _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss", "--strict")

    assert exit_info.value.code == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["val_loss"] == {"error": "RuntimeError: boom"}


def test_main_strict_does_not_exit_when_every_eval_succeeds(
    calls, bundle_dir, tmp_path, data_config
):
    _main(bundle_dir, tmp_path, data_config, "--evals", "val_loss", "--strict")


def test_main_rejects_a_concept_name_count_that_differs_from_the_models_channels(
    calls, bundle_dir, tmp_path, data_config
):
    bundle = tmp_path / "bundle_extra_concept"
    shutil.copytree(bundle_dir, bundle)
    model_json = bundle / evaluate_script.MODEL_JSON
    metadata = json.loads(model_json.read_text())
    metadata["concept_id2name"]["999"] = "phantom"
    model_json.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="concept names but the model has"):
        _main(bundle, tmp_path, data_config, "--evals", "val_loss")
    assert calls.never("val_loss")


@pytest.mark.parametrize("name", ["coralnet", "pacific", "coralscapes"])
def test_main_requires_the_reported_binary_concepts_for_the_concept_evals(
    monkeypatch, calls, bundle_dir, tmp_path, data_config, name
):
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", ["bleached", "anthropogenic"])

    with pytest.raises(RuntimeError, match=r"\['anthropogenic'\]"):
        _main(bundle_dir, tmp_path, data_config, "--evals", name)
    assert calls.never(name)


@pytest.mark.parametrize("name", ["coralscapes39", "val_loss"])
def test_main_does_not_require_the_reported_binary_concepts_for_the_other_evals(
    monkeypatch, calls, bundle_dir, tmp_path, data_config, name
):
    monkeypatch.setattr(evaluate_script, "REPORTED_BINARY_CONCEPTS", ["bleached", "anthropogenic"])

    _main(bundle_dir, tmp_path, data_config, "--evals", name)

    calls.one(name)


def _parse_defaults() -> Any:
    return evaluate_script.build_parser().parse_args(["--bundle", "b", "--output-dir", "out"])


def test_default_probe_points_file_loads():
    path = Path(_parse_defaults().probe_points)

    probe = json.loads(path.read_text())

    assert probe["num_classes"] == 95
    assert probe["images"]


def test_default_coralscapes39_probe_file_loads():
    path = _parse_defaults().coralscapes39_probe

    probes = load_legacy_probes([path])

    assert 50 in probes
