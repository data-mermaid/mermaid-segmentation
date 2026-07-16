"""Tests for offline experiment-config validation (``mermaidseg.experiment``)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mermaidseg.experiment import (
    Experiment,
    ExperimentSpec,
    offline_target_universe,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

# Real committed configs used as valid fixtures for the tmp-run-YAML tests.
DATA = CONFIGS / "data_config_coralnet_mermaid.yaml"
MODEL = CONFIGS / "model_config.yaml"
TRAINING = CONFIGS / "training_config_dinov3_linear.yaml"
LOGGER = CONFIGS / "logger_config.yaml"

_ALL_REGISTRY_DATASETS = (
    "pacific_labeled_corals",
    "moorea_labeled_corals",
    "catlin_seaview",
    "mermaid",
    "coralnet",
    "coralscapes",
    "coralscapes_v2",
    "benthos_yuval",
)


def _write(tmp_path: Path, doc: dict, name: str = "run.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _config_block(**overrides_and_paths) -> dict:
    """A `config:` block pointing at the real committed configs, with optional
    path/override subs."""
    overrides = overrides_and_paths.pop("overrides", {})
    block = {
        "config_data": overrides_and_paths.pop("config_data", str(DATA)),
        "config_model": overrides_and_paths.pop("config_model", str(MODEL)),
        "config_training": overrides_and_paths.pop("config_training", str(TRAINING)),
        "config_logger": overrides_and_paths.pop("config_logger", str(LOGGER)),
    }
    if overrides:
        block["overrides"] = overrides
    assert not overrides_and_paths, f"unused kwargs: {overrides_and_paths}"
    return block


# --------------------------------------------------------------------------------------
# Happy path — the committed run YAMLs validate (run from the repo root for relative paths).
# --------------------------------------------------------------------------------------


def test_issue12_run_yaml_validates(monkeypatch):
    monkeypatch.chdir(REPO)
    report = Experiment.validate(REPO / "sagemaker/runs/issue_12_dinov3_baseline.yaml")
    assert report.ok, report.errors
    assert report.summary["metric_of_interest"] == "miou"
    assert report.summary["datasets_enabled"].get("coralnet") == ["train", "val"]
    assert report.summary["datasets_enabled"].get("mermaid") == ["train"]


def test_example_run_yaml_validates(monkeypatch):
    monkeypatch.chdir(REPO)
    report = Experiment.validate(REPO / "sagemaker/configs/example/run.yaml")
    assert report.ok, report.errors


# --------------------------------------------------------------------------------------
# Schema errors (caught before any config file is opened).
# --------------------------------------------------------------------------------------


def test_unknown_metric_is_error(tmp_path):
    run = _write(tmp_path, {"config": _config_block(overrides={"metric-of-interest": "nope"})})
    report = Experiment.validate(run)
    assert not report.ok
    assert any("metric_of_interest" in err for err in report.errors)


def test_typo_override_key_is_error(tmp_path):
    # `epoch` instead of `epochs` — extra="forbid" turns it into an error, not a silent no-op.
    run = _write(tmp_path, {"config": _config_block(overrides={"epoch": 1})})
    report = Experiment.validate(run)
    assert not report.ok
    assert any("epoch" in err for err in report.errors)


def test_missing_config_block_is_error(tmp_path):
    run = _write(tmp_path, {"job": {"name_prefix": "x"}})
    report = Experiment.validate(run)
    assert not report.ok
    assert any("config:" in err for err in report.errors)


# --------------------------------------------------------------------------------------
# Static semantic errors (need the merged config).
# --------------------------------------------------------------------------------------


def test_missing_config_file_is_error(tmp_path):
    run = _write(
        tmp_path,
        {"config": _config_block(config_data=str(tmp_path / "does_not_exist.yaml"))},
    )
    report = Experiment.validate(run)
    assert not report.ok
    assert any("data config file not found" in err for err in report.errors)


def test_unknown_dataset_is_error(tmp_path):
    data_cfg = _write(
        tmp_path,
        {"data": {name: {} for name in (*_ALL_REGISTRY_DATASETS, "bogus_ds")}},
        name="data.yaml",
    )
    run = _write(tmp_path, {"config": _config_block(config_data=str(data_cfg))})
    report = Experiment.validate(run)
    assert not report.ok
    assert any("unknown dataset 'bogus_ds'" in err for err in report.errors)


def test_missing_dataset_is_error(tmp_path):
    # Declare only two of the eight registry datasets — _run_training would KeyError on the rest.
    data_cfg = _write(tmp_path, {"data": {"coralnet": {}, "mermaid": {}}}, name="data.yaml")
    run = _write(tmp_path, {"config": _config_block(config_data=str(data_cfg))})
    report = Experiment.validate(run)
    assert not report.ok
    assert any("missing dataset 'benthos_yuval'" in err for err in report.errors)


def test_bad_class_subset_name_is_warning_not_error(tmp_path):
    training_cfg = _write(
        tmp_path,
        {
            "training": {
                "training_mode": "standard",
                "padding": 3,
                "class_subset": ["Acropora", "ZZZ_NOT_A_REAL_LABEL"],
            }
        },
        name="training.yaml",
    )
    run = _write(tmp_path, {"config": _config_block(config_training=str(training_cfg))})
    report = Experiment.validate(run)
    assert report.ok, report.errors  # an unknown class name is advisory, not fatal
    assert any("ZZZ_NOT_A_REAL_LABEL" in warn for warn in report.warnings)


# --------------------------------------------------------------------------------------
# Units.
# --------------------------------------------------------------------------------------


def test_spec_parses_hyphenated_override_aliases():
    spec = ExperimentSpec.model_validate(
        {
            "config_data": "d",
            "config_model": "m",
            "config_training": "t",
            "overrides": {
                "metric-of-interest": "miou",
                "early-stopping": True,
                "early-stopping-patience": 20,
                "num-workers": 8,
            },
        }
    )
    assert spec.overrides.metric_of_interest == "miou"
    assert spec.overrides.early_stopping is True
    assert spec.overrides.early_stopping_patience == 20
    assert spec.overrides.num_workers == 8
    assert spec.mapping_source == "remote"  # additive default


def test_offline_universe_is_nonempty_and_has_snapshot_names():
    universe = offline_target_universe()
    assert "acropora" in universe  # from the committed CoralNet snapshot
    assert all(name == name.lower() for name in universe)
