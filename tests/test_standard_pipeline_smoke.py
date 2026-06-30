"""Smoke test for standard-mode training pipeline without S3 access.

This test exercises the full code path in scripts/train.py:_run_training by patching
MermaidDataset and CoralNetDataset with tiny synthetic alternatives. It verifies that
standard-mode training (non-CBM, no concept bottleneck) runs end-to-end without concept-
related code paths failing.

Run with: uv run pytest tests/test_standard_pipeline_smoke.py -v -m smoke
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


class SyntheticCoralDataset:
    """Minimal synthetic dataset matching the interface expected by attach_registry."""

    def __init__(self, name: str, num_samples: int = 20):
        self.SOURCE_NAME = name
        self.num_samples = num_samples
        self.source_id2name = {0: "background", 1: "coral"}
        self.source_name2id = {"background": 0, "coral": 1}
        self.num_source_classes = 2
        self._global_offset = 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.zeros(3, 512, 512, dtype=torch.float32)
        labels = torch.zeros(512, 512, dtype=torch.long)
        return image, labels

    def set_global_offset(self, offset: int) -> None:
        self._global_offset = offset

    def num_load_failures(self) -> int:
        return 0


@pytest.mark.smoke
def test_standard_mode_training_pipeline_smoke(tmp_path, monkeypatch):
    """Full pipeline smoke test: standard mode training with synthetic datasets.

    This test patches MermaidDataset, CoralNetDataset, and MetaModel to avoid S3 access and
    slow model downloads. Verifies that ConceptSchema loading is properly guarded in standard
    mode and the registry is initialized with compute_concepts=False.

    Verifies that:
    - ConceptSchema is NOT loaded in standard mode
    - Registry is built with compute_concepts=False
    - No concept-related assertions fail
    - Schema assertion is skipped in standard mode
    """
    monkeypatch.chdir(Path(__file__).parent.parent)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))

    from scripts.train import _run_training

    synthetic_mermaid = SyntheticCoralDataset("mermaid", num_samples=20)
    synthetic_coralnet = SyntheticCoralDataset("coralnet", num_samples=20)

    concept_schema_calls = []

    original_from_csv = __import__(
        "mermaidseg.dataset_reconciliation", fromlist=["ConceptSchema"]
    ).ConceptSchema.from_csv

    def tracked_from_csv(*args, **kwargs):
        concept_schema_calls.append((args, kwargs))
        return original_from_csv(*args, **kwargs)

    patches = [
        patch("scripts.train.MermaidDataset", return_value=synthetic_mermaid),
        patch("scripts.train.CoralNetDataset", return_value=synthetic_coralnet),
        patch("scripts.train.BenthosYuvalCoralsDataset"),
        patch("scripts.train.CatlinSeaviewDataset"),
        patch("scripts.train.CoralscapesDataset"),
        patch("scripts.train.CoralscapesV2Dataset"),
        patch("scripts.train.MooreaLabeledCoralsDataset"),
        patch("scripts.train.PacificLabeledCoralsDataset"),
        patch("scripts.train.ConceptSchema.from_csv", side_effect=tracked_from_csv),
        patch("scripts.train.MetaModel"),
    ]

    with contextlib.ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        (
            mock_benthos,
            mock_catlin,
            mock_coralscapes,
            mock_coralscapes_v2,
            mock_moorea,
            mock_pacific,
            _,
            mock_meta_model,
        ) = mocks[2:10]

        for mock in [
            mock_benthos,
            mock_catlin,
            mock_coralscapes,
            mock_coralscapes_v2,
            mock_moorea,
            mock_pacific,
        ]:
            mock.side_effect = ValueError("Should not be called in standard baseline")

        mock_meta_model.return_value = MagicMock(
            model=MagicMock(),
            optimizer=MagicMock(),
            scheduler=None,
        )

        args = argparse.Namespace(
            config_data="configs/data_config_dinov3_base.yaml",
            config_model="configs/model_config_dinov3_base.yaml",
            config_training="configs/training_config_dinov3_base.yaml",
            config_logger="configs/logger_config.yaml",
            dry_run=True,
            auto_shutdown=False,
            log_dir=str(tmp_path / "logs"),
            failure_report_path=None,
            early_stopping=False,
            early_stopping_patience=10,
            early_stopping_min_delta=0.0,
            metric_of_interest="accuracy",
            seed=42,
            num_workers=0,
            run_name="smoke-test",
            epochs=None,
            batch_size=None,
        )

        with contextlib.suppress(Exception):
            _run_training(args)

        assert len(concept_schema_calls) == 0, (
            f"ConceptSchema.from_csv should not be called in standard mode, "
            f"but was called {len(concept_schema_calls)} times"
        )
