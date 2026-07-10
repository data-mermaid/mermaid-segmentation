"""Seam 2: Logger swallows only expected MLflow/IO failures — real bugs surface.

The logging try/excepts were `except Exception: warning(...)`, which silently swallowed
programming errors (the dropped/misnamed-metric class of bug). They now catch only
(MlflowException, OSError), so a genuine bug in a logging call propagates instead of
degrading observability unnoticed.
"""

from __future__ import annotations

import mlflow
import pytest

from mermaidseg.logger import Logger


def _enabled_logger(make_config, fake_meta_model) -> Logger:
    lgr = Logger(config=make_config(), meta_model=fake_meta_model, enable_mlflow=True)
    assert lgr.enabled is True
    return lgr


def test_log_propagates_programming_error(
    tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch
):
    """A non-MLflow error (e.g. ValueError) in a logging call must NOT be swallowed."""
    lgr = _enabled_logger(make_config, fake_meta_model)

    def boom(*_args, **_kwargs):
        raise ValueError("real bug, not a tracking failure")

    monkeypatch.setattr(mlflow, "log_metrics", boom)
    with pytest.raises(ValueError, match="real bug"):
        lgr.log({"train/loss": 1.0}, step=0)


def test_log_swallows_mlflow_exception(tmp_mlflow_uri, make_config, fake_meta_model, monkeypatch):
    """A transient MlflowException stays swallowed (warn + continue) — not fatal to
    training."""
    lgr = _enabled_logger(make_config, fake_meta_model)

    def boom(*_args, **_kwargs):
        raise mlflow.exceptions.MlflowException("transient tracking failure")

    monkeypatch.setattr(mlflow, "log_metrics", boom)
    lgr.log({"train/loss": 1.0}, step=0)  # must not raise
