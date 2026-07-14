"""Tests for the in-container processing entrypoint
(scripts/sagemaker_processing_entrypoint.py).

The entrypoint dispatches on --task and forwards the remaining args to the matching
routine. A task that isn't wired into the argparse choices fails the whole job at
container start (the "invalid choice: 'coralnet-resize'" CloudWatch error), so these
tests lock the dispatch table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sagemaker_processing_entrypoint as entrypoint  # type: ignore  # noqa: E402


def test_dispatches_coralnet_resize_and_forwards_args(monkeypatch):
    """--task=coralnet-resize routes to resize.main with the remaining args and exits its code."""
    received: dict[str, object] = {}

    def fake_resize_main(extra: list[str]) -> int:
        received["extra"] = extra
        return 0

    monkeypatch.setattr("mermaidseg.datasets.coralnet.preprocessing.resize.main", fake_resize_main)
    monkeypatch.setattr(sys, "argv", ["prog", "--task=coralnet-resize", "--run=foo", "--bucket=b"])

    with pytest.raises(SystemExit) as exc:
        entrypoint.main()

    assert exc.value.code == 0
    assert received["extra"] == ["--run=foo", "--bucket=b"]


def test_resize_nonzero_exit_propagates(monkeypatch):
    """A non-zero return from resize.main becomes the entrypoint's exit code (failed-job
    signal)."""
    monkeypatch.setattr("mermaidseg.datasets.coralnet.preprocessing.resize.main", lambda extra: 1)
    monkeypatch.setattr(sys, "argv", ["prog", "--task=coralnet-resize", "--run=foo"])

    with pytest.raises(SystemExit) as exc:
        entrypoint.main()

    assert exc.value.code == 1


def test_unknown_task_is_rejected(monkeypatch):
    """An unregistered task is rejected by argparse before any routine runs."""
    monkeypatch.setattr(sys, "argv", ["prog", "--task=does-not-exist"])

    with pytest.raises(SystemExit) as exc:
        entrypoint.main()

    # argparse exits with code 2 on invalid choice.
    assert exc.value.code == 2
