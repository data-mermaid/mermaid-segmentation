"""Training entrypoint parser."""

from __future__ import annotations

import pytest

import scripts.train as train_script
from mermaidseg.io import resolve_repo_path
from scripts.train import build_parser

REQUIRED = ["--num-workers", "8"]


CONFIG_FLAGS = (
    "training_config",
    "model_config",
    "data_config",
    "concept_schema_config",
    "logger_config",
)


def test_config_defaults_point_at_existing_repo_files():
    args = build_parser().parse_args(REQUIRED)
    for flag in CONFIG_FLAGS:
        assert resolve_repo_path(getattr(args, flag)).is_file(), flag


def test_main_rejects_zero_nccl_timeout(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", *REQUIRED, "--nccl-timeout-min", "0"])
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()


def test_main_rejects_zero_num_workers(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--num-workers", "0"])
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()


def test_main_rejects_checkpoint_with_init_weights(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", *REQUIRED, "--checkpoint", "x", "--init-weights", "y"]
    )
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()
