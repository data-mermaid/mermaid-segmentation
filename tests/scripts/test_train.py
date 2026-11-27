"""Training entrypoint parser."""

from __future__ import annotations

import pytest

import scripts.train as train_script
from mermaidseg.io import resolve_repo_path
from scripts.train import build_parser

REQUIRED = ["--num-workers", "8", "--cache-dir", "cache"]


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


def _stub_run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list:
    """Point ``sys.argv`` at ``argv`` and replace ``run``; returns the args ``run`` receives."""
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    received: list = []
    monkeypatch.setattr(train_script, "run", received.append)
    return received


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--cache-dir", "cache", "--num-workers", "0"], id="zero-num-workers"),
        pytest.param([*REQUIRED, "--nccl-timeout-min", "0"], id="zero-nccl-timeout"),
        pytest.param(
            [*REQUIRED, "--preempt-checkpoint-every-steps", "-1"], id="negative-preempt-every"
        ),
        pytest.param(
            [*REQUIRED, "--checkpoint", "x", "--init-weights", "y"], id="checkpoint-and-init"
        ),
    ],
)
def test_main_rejects_invalid_arguments_without_starting_run(monkeypatch, argv):
    received = _stub_run(monkeypatch, argv)
    with pytest.raises(SystemExit) as exit_info:
        train_script.main()
    assert exit_info.value.code == 2
    assert received == []


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(["--nccl-timeout-min", "0.01"], id="small-positive-nccl-timeout"),
        pytest.param(["--preempt-checkpoint-every-steps", "0"], id="preempt-disabled"),
        pytest.param(["--num-workers", "1"], id="single-data-loader-worker"),
    ],
)
def test_main_accepts_the_edge_of_each_valid_range(monkeypatch, extra):
    received = _stub_run(monkeypatch, [*REQUIRED, *extra])
    train_script.main()
    assert len(received) == 1
