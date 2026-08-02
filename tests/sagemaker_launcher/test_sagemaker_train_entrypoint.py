"""Tests for the in-container training entrypoint
(scripts/sagemaker_train_entrypoint.py).

Regression guard: ``_find_run_yaml`` used to sniff for a run YAML by checking whether
the literal substring ``"job:"`` appeared in the file's first 200 characters. Any run
YAML with a comment header longer than that (e.g. the taxonomical-loss-ablation configs)
pushed the ``job:`` key past the sniff window, so the container failed at start with "No
run YAML with a job: block" even though the file was a perfectly valid run YAML. The fix
parses each YAML and checks for a top-level ``job`` key instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sagemaker_train_entrypoint as entrypoint  # type: ignore  # noqa: E402

RUN_YAML_BODY = """
job:
  name_prefix: test-run
  image: mermaid-segmentation-jobs:training-latest
  entrypoint: scripts/sagemaker_train_entrypoint.py
  instance_type: ml.g5.2xlarge
  volume_gb: 200
  max_runtime_hours: 48

config:
  config_data: configs/data_config.yaml
  config_model: configs/model_config.yaml
  config_training: configs/training_config.yaml
  config_logger: configs/logger_config.yaml
"""

LONG_COMMENT_HEADER = "\n".join(f"# {'x' * 76}" for _ in range(6))  # > 200 chars


def test_finds_run_yaml_behind_a_long_comment_header(tmp_path, monkeypatch):
    """A comment header longer than 200 chars must not hide the job: block (the bug)."""
    (tmp_path / "run.yaml").write_text(LONG_COMMENT_HEADER + "\n" + RUN_YAML_BODY)
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    found = entrypoint._find_run_yaml()

    assert found == tmp_path / "run.yaml"


def test_ignores_non_run_yaml_in_the_same_directory(tmp_path, monkeypatch):
    """A sibling YAML without a job: key (e.g. a split config) is not mistaken for the
    run YAML."""
    (tmp_path / "run.yaml").write_text(RUN_YAML_BODY)
    (tmp_path / "data_config.yaml").write_text("data:\n  default:\n    train: {}\n")
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    found = entrypoint._find_run_yaml()

    assert found == tmp_path / "run.yaml"


def test_raises_when_no_run_yaml_present(tmp_path, monkeypatch):
    (tmp_path / "data_config.yaml").write_text("data:\n  default: {}\n")
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    with pytest.raises(SystemExit, match="No run YAML with a job: block"):
        entrypoint._find_run_yaml()


def test_raises_on_multiple_run_yamls(tmp_path, monkeypatch):
    (tmp_path / "run.yaml").write_text(RUN_YAML_BODY)
    (tmp_path / "run2.yaml").write_text(RUN_YAML_BODY)
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)

    with pytest.raises(SystemExit, match="Multiple candidate run YAMLs"):
        entrypoint._find_run_yaml()
