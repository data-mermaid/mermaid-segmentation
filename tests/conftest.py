"""Shared fixtures for mermaidseg test suite."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.io import ConfigDict

NUM_CLASSES = 4
IMAGE_SIZE = (56, 56)

# Two values per taxonomic rank plus one binary concept: 2 * 6 + 1 channels.
CBM_CONCEPT_VALUE2ID = {
    **{rank: {"not_given": [0, 0], "a": [2, 1], "b": [1, 2]} for rank in TAXONOMIC_CONCEPTS},
    "bleached": {"not_given": 0, False: 1, True: 2},
}
CBM_NUM_CONCEPTS = 2 * len(TAXONOMIC_CONCEPTS) + 1


def cbm_concept_lookup() -> torch.Tensor:
    """``(NUM_CLASSES, C)`` concept rows: label 1 is all ``a``, 2 all ``b``, 3 ``a`` unbleached."""
    a, b = [2.0, 1.0], [1.0, 2.0]
    rows = [
        [0.0] * CBM_NUM_CONCEPTS,
        a * len(TAXONOMIC_CONCEPTS) + [1.0],
        b * len(TAXONOMIC_CONCEPTS) + [2.0],
        a * len(TAXONOMIC_CONCEPTS) + [0.0],
    ]
    return torch.tensor(rows)


class FakeMetaModel:
    """Minimal MetaModel substitute used by Logger tests."""

    def __init__(
        self,
        run_name: str = "test-run",
        num_classes: int = 3,
        num_concepts: int | None = None,
        conceptid2labelid: dict[int, int] | None = None,
    ):
        self.run_name = run_name
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.conceptid2labelid = conceptid2labelid
        self.device = "cpu"
        self.device_type = "cpu"
        self.global_step = 0
        self._warmup_iters_completed = 0
        self.model = nn.Linear(4, num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)


@pytest.fixture()
def fake_meta_model():
    """Return a fresh FakeMetaModel on each call."""
    return FakeMetaModel()


@pytest.fixture()
def make_config():
    """Factory that builds a ConfigDict.

    Override any nested value via keyword arguments, e.g. ``make_config(logger={"wandb_project":
    "other"})``.
    """

    def _factory(**overrides: Any) -> ConfigDict:
        base: dict[str, Any] = {
            "logger": {
                "tracker": "wandb",
                "wandb_project": "mermaidseg",
                "wandb_entity": None,
                "train_log_interval": 1,
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


def cbm_test_config() -> ConfigDict:
    """Concept-bottleneck model and training sections with every key MetaModel reads.

    LoRA and other parameters get distinct learning rates, and the schedule keeps them
    constant (no warmup, LinearLR factors of 1).
    """
    return ConfigDict(
        {
            "model": {
                "name": "ConceptBottleneckDPTLoRADINOv3",
                "encoder_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "input_size": IMAGE_SIZE,
                "concept_feature_dim": 8,
                "out_indices": None,
                "neck_hidden_sizes": [8, 8, 8, 8],
                "fusion_hidden_size": 8,
                "reassemble_factors": [4, 2, 1, 0.5],
                "lora_r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj"],
                "lora_bias": "none",
            },
            "training": {
                "training_mode": "concept-bottleneck",
                "detach_concepts": True,
                "distributed": False,
                "world_size": 1,
                "concept_mapping_path": "configs/class_to_concepts.csv",
                "epochs": 1,
                "batch_size": 2,
                "stall_dump_sec": 180,
                "max_grad_norm": 1.0,
                "mixed_precision": False,
                "mixed_precision_dtype": "bfloat16",
                "grad_accum_steps": 1,
                "train_metrics_every": 1,
                "iterations_per_train_epoch": 1,
                "iterations_per_val_epoch": 1,
                "optimizer": {
                    "type": "AdamW",
                    "weight_decay": 0.0,
                    "lora_lr": 1e-5,
                    "dpt_lr": 5e-5,
                },
                "scheduler": {
                    "type": "LinearLR",
                    "start_factor": 1.0,
                    "end_factor": 1.0,
                    "total_iters": 1,
                    "step_every": "iteration",
                    "warmup_iters": 0,
                    "warmup_start_factor": 1.0,
                },
                "loss": {
                    "type": "ConceptBottleneckLoss",
                    "lambda_weight": 1.0,
                    "ignore_index": 0,
                    "per_pixel_loss_weight": 0.0,
                    "per_image_loss_weight": 1.0,
                },
            },
        }
    )


@pytest.fixture()
def minimal_config():
    """Minimal ConfigDict for MetaModel instantiation with a mocked DINOv3 encoder."""
    return cbm_test_config()


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
    """AutoModel.from_pretrained mock returning a tiny encoder with a LoRA target.

    Returns a factory that patches AutoModel on a target module. Uses hidden_size=16,
    patch_size=14, 4 layers and 5 prefix tokens (CLS + 4 registers). ``q_proj`` is the
    module LoRA wraps.
    """
    hidden_size = 16
    patch_size = 14
    num_layers = 4
    num_register_tokens = 4

    class _MockEncoderConfig(SimpleNamespace):
        def to_dict(self) -> dict[str, Any]:
            return dict(vars(self))

    class _MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _MockEncoderConfig(
                hidden_size=hidden_size,
                patch_size=patch_size,
                num_hidden_layers=num_layers,
                num_register_tokens=num_register_tokens,
            )
            self.embed = nn.Linear(3, hidden_size)
            self.q_proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, pixel_values, **_):
            patches = nn.functional.avg_pool2d(pixel_values, patch_size).flatten(2).transpose(1, 2)
            tokens = self.embed(patches)
            prefix = tokens.new_zeros(tokens.shape[0], 1 + num_register_tokens, tokens.shape[2])
            states = self.q_proj(torch.cat([prefix, tokens], dim=1))
            return SimpleNamespace(hidden_states=(states,) * (num_layers + 1))

    def _apply_mock(target_module):
        mock_encoder = _MockEncoder()

        def _from_pretrained(*_args, **_kwargs):
            return mock_encoder

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained = _from_pretrained
        monkeypatch.setattr(target_module, "AutoModel", mock_auto_model)
        return mock_encoder

    return _apply_mock


@pytest.fixture()
def make_meta_model(mock_dinov3_encoder):
    """Factory that builds a real concept-bottleneck MetaModel with a mocked encoder.

    The concept layout defaults to ``CBM_CONCEPT_VALUE2ID`` and ``cbm_concept_lookup``;
    pass ``num_concepts``, ``concept_value2id`` and ``source_to_concepts_lookup`` to override.
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
        extra_kwargs.setdefault("num_concepts", CBM_NUM_CONCEPTS)
        extra_kwargs.setdefault("concept_value2id", CBM_CONCEPT_VALUE2ID)
        extra_kwargs.setdefault("source_to_concepts_lookup", cbm_concept_lookup())
        return MetaModel(
            run_name=run_name,
            num_classes=num_classes,
            model_kwargs=minimal_config.model,
            training_kwargs=minimal_config.training,
            device=device,
            **extra_kwargs,
        )

    return _factory
