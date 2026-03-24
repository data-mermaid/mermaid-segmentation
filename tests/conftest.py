"""Shared fixtures for mermaidseg test suite."""

from __future__ import annotations

from typing import Any

import mlflow
import pytest
import torch
import torch.nn as nn

from mermaidseg.io import ConfigDict


class FakeMetaModel:
    """Minimal MetaModel substitute used by Logger tests."""

    def __init__(
        self,
        run_name: str = "test-run",
        num_classes: int = 3,
        num_concepts: int | None = None,
        conceptid2labelid: dict[int, int] | None = None,
        concept_matrix: Any | None = None,
    ):
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.conceptid2labelid = conceptid2labelid
        self.concept_matrix = concept_matrix
        self.device = "cpu"
        self.model = nn.Linear(4, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)


@pytest.fixture(autouse=True)
def _cleanup_mlflow_run():
    """Ensure no active MLflow run leaks between tests."""
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture()
def tmp_mlflow_uri(tmp_path, monkeypatch):
    """Set MLFLOW_TRACKING_URI to a temporary directory for an isolated file store."""
    uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture()
def fake_meta_model():
    """Return a fresh FakeMetaModel on each call."""
    return FakeMetaModel()


@pytest.fixture()
def make_config():
    """Factory that builds a ConfigDict with the keys Logger expects.

    Override any nested value via keyword arguments, e.g.
    ``make_config(logger={"experiment_name": "custom"})``.
    """

    def _factory(**overrides: Any) -> ConfigDict:
        base: dict[str, Any] = {
            "logger": {
                "experiment_name": "unit-tests",
                "uri": None,
                "save_local_checkpoints": True,
                "save_local_models": True,
            },
            "model": {
                "name": "FakeLinear",
                "encoder": "resnet18",
            },
            "training": {
                "epochs": 2,
                "optimizer": {"type": "SGD", "lr": 0.01},
                "batch_size": 4,
            },
        }
        for section, values in overrides.items():
            if section in base and isinstance(values, dict):
                base[section].update(values)
            else:
                base[section] = values
        return ConfigDict(base)

    return _factory
