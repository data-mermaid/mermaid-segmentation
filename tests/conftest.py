"""Shared fixtures for mermaidseg test suite."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import mlflow
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mermaidseg.io import ConfigDict

NUM_CLASSES = 4
IMAGE_SIZE = (56, 56)


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
    """Factory that builds a ConfigDict.

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
                "encoder_name": "resnet18",
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


@pytest.fixture()
def minimal_config():
    """Minimal ConfigDict for MetaModel instantiation with a mocked DINOv3 encoder."""
    return ConfigDict(
        {
            "model": {
                "name": "LinearDINOv3",
                "encoder_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "input_size": IMAGE_SIZE,
            },
            "training": {
                "training_mode": "standard",
                "epochs": 1,
                "batch_size": 2,
                "optimizer": {"type": "SGD", "lr": 0.01},
                "loss": {"type": "CrossEntropyLoss"},
            },
        }
    )


@pytest.fixture()
def tiny_batch():
    """Synthetic batch for model tests: (images, masks)."""
    images = torch.randn(2, 3, *IMAGE_SIZE)
    masks = torch.randint(0, NUM_CLASSES, (2, *IMAGE_SIZE))
    return images, masks


@pytest.fixture()
def tiny_loader(tiny_batch):
    """DataLoader wrapping tiny_batch (1 batch of 2 samples)."""
    images, masks = tiny_batch
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=2, num_workers=0)


@pytest.fixture()
def mock_dinov3_encoder(monkeypatch):
    """AutoModel.from_pretrained mock returning controlled hidden states.

    Returns a factory that patches AutoModel on a target module. Uses hidden_size=64 (small for fast
    CI), patch_size=14, n_prefix_tokens=5 (CLS + 4 registers).
    """
    hidden_size = 64
    patch_size = 14
    n_prefix_tokens = 5

    def _make_mock(pattern="zeros"):
        _hs = hidden_size
        _ps = patch_size
        _np = n_prefix_tokens

        cfg = SimpleNamespace(hidden_size=_hs, patch_size=_ps)

        class _MockEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = cfg

            def forward(self, pixel_values, **_):
                B, _C, H, W = pixel_values.shape
                seq_len = _np + (H // _ps) * (W // _ps)
                if pattern == "arange":
                    states = (
                        torch.arange(
                            seq_len * _hs, dtype=pixel_values.dtype, device=pixel_values.device
                        )
                        .reshape(1, seq_len, _hs)
                        .expand(B, -1, -1)
                    )
                else:
                    states = torch.zeros(
                        B, seq_len, _hs, device=pixel_values.device, dtype=pixel_values.dtype
                    )
                return SimpleNamespace(last_hidden_state=states)

        return _MockEncoder()

    def _apply_mock(target_module, pattern="zeros"):
        mock_encoder = _make_mock(pattern)

        def _from_pretrained(*_args, **_kwargs):
            return mock_encoder

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained = _from_pretrained
        monkeypatch.setattr(target_module, "AutoModel", mock_auto_model)
        return mock_encoder

    return _apply_mock


@pytest.fixture()
def make_meta_model(mock_dinov3_encoder):
    """Factory that builds a real MetaModel with a mocked encoder.

    Eliminates the repeated import-mock-instantiate boilerplate in integration tests.
    """
    from mermaidseg.model.meta import MetaModel

    def _factory(
        minimal_config: ConfigDict,
        *,
        run_name: str = "test-run",
        num_classes: int = NUM_CLASSES,
        device: str = "cpu",
        **extra_kwargs: Any,
    ) -> MetaModel:
        import mermaidseg.model.models

        mock_dinov3_encoder(mermaidseg.model.models)
        return MetaModel(
            run_name=run_name,
            num_classes=num_classes,
            model_kwargs=minimal_config.model,
            training_kwargs=minimal_config.training,
            device=device,
            **extra_kwargs,
        )

    return _factory
