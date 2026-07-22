"""Tests for the experiment module (``mermaidseg.experiment``): offline validation, the
config-loading equivalence with the legacy CLI merge, and offline dry-run assembly."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from mermaidseg.experiment import (
    Experiment,
    ExperimentSpec,
    offline_target_universe,
)
from mermaidseg.io import setup_config, update_config_with_args

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
    assert report.summary["datasets_enabled"].get("mermaid") == ["train", "val"]


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


# --------------------------------------------------------------------------------------
# Loader: config-merge equivalence, YAML loading, and offline dry-run assembly.
# --------------------------------------------------------------------------------------


def test_from_args_config_matches_legacy_merge(monkeypatch):
    """Experiment.from_args must produce the same merged config as the legacy CLI path
    (setup_config + update_config_with_args) — the behavior-preservation guarantee for
    PR 2."""
    monkeypatch.chdir(REPO)
    args = argparse.Namespace(
        config_data="configs/data_config_coralnet_mermaid.yaml",
        config_model="configs/model_config.yaml",
        config_training="configs/training_config_dinov3_linear.yaml",
        config_logger="configs/logger_config.yaml",
        run_name="rn",
        experiment_name="baselines",
        epochs=7,
        batch_size=16,
        lr=0.002,
        seed=123,
        num_workers=4,
        metric_of_interest="miou_weighted",
        early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.01,
        per_class_metrics=None,
        dry_run=False,
        auto_shutdown=False,
        log_dir="logs",
        failure_report_path=None,
        model=None,
        model_checkpoint=None,
        iterations_per_train_epoch=None,
        iterations_per_val_epoch=None,
        log_epochs=None,
    )
    exp = Experiment.from_args(args)
    legacy = update_config_with_args(
        setup_config(
            {
                "data": args.config_data,
                "training": args.config_training,
                "model": args.config_model,
                "logger": args.config_logger,
            }
        ),
        args,
    )
    # Every override-affected field matches the legacy merge.
    assert exp.config.run_name == legacy.run_name == "rn"
    assert exp.config.training.epochs == legacy.training.epochs == 7
    assert exp.config.training.batch_size == legacy.training.batch_size == 16
    assert exp.config.training.optimizer.lr == legacy.training.optimizer.lr == 0.002
    # Run-params (not in cfg today) are surfaced on the unified override object.
    assert exp.overrides.seed == 123
    assert exp.overrides.metric_of_interest == "miou_weighted"
    assert exp.overrides.early_stopping is True


def test_from_run_yaml_loads_spec_and_applies_overrides(monkeypatch):
    monkeypatch.chdir(REPO)
    exp = Experiment.from_run_yaml("sagemaker/runs/issue_12_dinov3_baseline.yaml")
    assert exp.spec.overrides.metric_of_interest == "miou"
    assert exp.spec.overrides.early_stopping is True
    assert exp.config.training.epochs == 200  # override merged into the config
    assert exp.config.training.training_mode == "standard"


class _SyntheticDataset:
    """Minimal dataset matching the SourceLabelRegistry / DataLoader interface (no
    S3)."""

    def __init__(self, name: str, num_samples: int = 8, **_ignored):
        self.SOURCE_NAME = name
        self._num_samples = num_samples
        self.source_id2name = {0: "background", 1: "coral"}
        self.source_name2id = {"background": 0, "coral": 1}
        self.num_source_classes = 2
        self._global_offset = 0

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, _idx: int):
        return torch.zeros(3, 512, 512), torch.zeros(512, 512, dtype=torch.long)

    def set_global_offset(self, offset: int) -> None:
        self._global_offset = offset

    def num_load_failures(self) -> int:
        return 0


def test_dry_run_assembles_offline(tmp_path, monkeypatch):
    """dry_run() assembles datasets -> registry -> (mocked) model -> evaluator and pulls
    one batch, offline: only MERMAID is active (identity mapping, no live API),
    label_roll_up off."""
    monkeypatch.chdir(REPO)
    data_doc = {"data": {name: {"train": "None", "val": "None"} for name in _ALL_REGISTRY_DATASETS}}
    data_doc["data"]["mermaid"] = {
        "train": {},
        "val": {},
    }  # both splits active (ConcatDataset needs val)
    data_cfg = _write(tmp_path, data_doc, name="data.yaml")
    training_cfg = _write(
        tmp_path,
        {
            "training": {
                "training_mode": "standard",
                "padding": 3,
                "batch_size": 4,
                "label_roll_up": False,
                "class_subset": ["coral"],  # MERMAID's identity map emits "coral"
            }
        },
        name="training.yaml",
    )
    spec = ExperimentSpec(
        config_data=str(data_cfg),
        config_model=str(MODEL),
        config_training=str(training_cfg),
        config_logger=str(LOGGER),
        overrides={"run_name": "dry-run-test"},  # meta_model() reads cfg.run_name
    )
    override = {"mermaid": lambda **kw: _SyntheticDataset("mermaid")}
    with (
        patch.dict("mermaidseg.experiment.DATASET_REGISTRY", override),
        patch("mermaidseg.experiment.MetaModel"),
    ):
        experiment = Experiment.from_spec(spec, device=torch.device("cpu"))
        summary = experiment.dry_run()

    assert summary["datasets"] == ["mermaid"]
    assert summary["train_batches"] == 2  # 8 synthetic samples / batch 4, drop_last
    assert summary["train_batch_image_shape"] == (4, 3, 512, 512)
