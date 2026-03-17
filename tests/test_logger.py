"""Tests for mermaidseg.logger – Phase 1 of the wandb separation plan."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from mermaidseg.logger import (
    LOCAL_DEFAULT_URI,
    Logger,
    WandbLogger,
    get_mlflow_tracking_uri,
    mlflow_connect,
)

from .conftest import FakeMetaModel


# ===================================================================
# get_mlflow_tracking_uri
# ===================================================================
class TestGetMlflowTrackingUri:
    """Test URI resolution priority: env var > config > default."""

    @pytest.mark.parametrize(
        "env_val,config_val,expected",
        [
            ("http://from-env:5000", "from-config", "http://from-env:5000"),
            (None, "segmentation", "segmentation"),
            (None, None, LOCAL_DEFAULT_URI),
        ],
        ids=["env-priority", "config-fallback", "default-fallback"],
    )
    def test_uri_resolution(self, monkeypatch, env_val, config_val, expected):
        if env_val:
            monkeypatch.setenv("MLFLOW_TRACKING_URI", env_val)
        else:
            monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert get_mlflow_tracking_uri(config_val) == expected


# ===================================================================
# mlflow_connect
# ===================================================================
class TestMlflowConnect:
    """Test MLflow connection establishment and error handling."""

    def test_successful_connect(self, tmp_mlflow_uri):
        duration = mlflow_connect(tmp_mlflow_uri)
        assert isinstance(duration, timedelta)
        assert duration.total_seconds() >= 0

    def test_arn_warning_when_sagemaker_missing(self, monkeypatch, caplog):
        arn = "arn:aws:sagemaker:us-east-1:123456:mlflow-tracking-server/my-app"

        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "sagemaker_mlflow":
                raise ImportError("no sagemaker_mlflow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with (
            patch("mermaidseg.logger.mlflow.set_tracking_uri"),
            patch("mermaidseg.logger.mlflow.search_experiments"),
        ):
            duration = mlflow_connect(arn)

        assert "sagemaker-mlflow is not installed" in caplog.text
        assert isinstance(duration, timedelta)

    def test_max_retries_raises_runtime_error(self, tmp_path):
        uri = str(tmp_path / "bogus")
        exc = mlflow.exceptions.MlflowException("Max retries exceeded with url")
        with patch("mermaidseg.logger.mlflow.set_tracking_uri"):
            with patch("mermaidseg.logger.mlflow.search_experiments", side_effect=exc):
                with pytest.raises(RuntimeError, match="Could not connect"):
                    mlflow_connect(uri)

    def test_other_mlflow_exception_reraised(self, tmp_path):
        uri = str(tmp_path / "bogus")
        exc = mlflow.exceptions.MlflowException("Something else broke")
        with patch("mermaidseg.logger.mlflow.set_tracking_uri"):
            with patch("mermaidseg.logger.mlflow.search_experiments", side_effect=exc):
                with pytest.raises(mlflow.exceptions.MlflowException, match="Something else"):
                    mlflow_connect(uri)


# ===================================================================
# Logger.__init__
# ===================================================================
class TestLoggerInit:
    """Test Logger initialization: MLflow connection, params/tags, config toggles."""

    def test_mlflow_run_started(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.enabled is True
        assert lgr.mlflow_run_id is not None
        assert mlflow.active_run() is not None

    def test_params_and_tags_logged(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        run = mlflow.get_run(lgr.mlflow_run_id)

        params = run.data.params
        assert params["num_classes"] == str(fake_meta_model.num_classes)
        assert params["run_name"] == fake_meta_model.run_name
        assert "model_encoder" in params
        assert params["training_epochs"] == "2"

        tags = run.data.tags
        assert tags["model_type"] == "FakeMetaModel"
        assert tags["framework"] == "pytorch"

    def test_save_local_checkpoints_from_config(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.save_local_checkpoints is False

    def test_disabled_when_no_experiment_name(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config(logger={"experiment_name": None})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.enabled is False
        assert lgr.mlflow_run_id is None

    def test_concept_metadata_logged(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel(num_concepts=5, conceptid2labelid={0: 0, 1: 1})
        config = make_config()
        lgr = Logger(
            config=config,
            meta_model=meta,
            id2label={0: "coral", 1: "sand", 2: "rock"},
            id2concept={0: "hard", 1: "soft"},
        )
        run = mlflow.get_run(lgr.mlflow_run_id)
        assert run.data.params["num_concepts"] == "5"

    def test_concept_matrix_logging_does_not_disable_logger(
        self, tmp_mlflow_uri, make_config, tmp_path
    ):
        missing_dir = tmp_path / "missing-dir"
        assert not missing_dir.exists()
        meta = FakeMetaModel(concept_matrix=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        lgr = Logger(config=make_config(), meta_model=meta, checkpoint_dir=str(missing_dir))
        assert lgr.enabled is True
        assert lgr.mlflow_run_id is not None

    def test_log_checkpoint_zero_rejected(self, tmp_mlflow_uri, make_config, fake_meta_model):
        with pytest.raises(ValueError, match="log_checkpoint must be > 0"):
            Logger(config=make_config(), meta_model=fake_meta_model, log_checkpoint=0)


# ===================================================================
# Logger.log
# ===================================================================
class TestLoggerLog:
    """Test metric logging: scalars, ndarrays with id maps, wandb fallback."""

    def test_ndarray_class_metrics_unpacked(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, id2label={0: "coral", 1: "sand", 2: "rock"})
        lgr.log({"iou": np.array([0.8, 0.6, 0.7])}, step=1)
        metrics = mlflow.get_run(lgr.mlflow_run_id).data.metrics
        assert metrics["iou/coral"] == pytest.approx(0.8)
        assert metrics["iou/sand"] == pytest.approx(0.6)
        assert metrics["iou/rock"] == pytest.approx(0.7)

    def test_ndarray_without_id_map_uses_index(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        lgr.log({"dice": np.array([0.1, 0.2])}, step=1)
        metrics = mlflow.get_run(lgr.mlflow_run_id).data.metrics
        assert metrics["dice/class_0"] == pytest.approx(0.1)
        assert metrics["dice/class_1"] == pytest.approx(0.2)

    def test_wandb_fallback_logging(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        lgr._wandb_logger = mock_wandb_logger

        lgr.log({"loss": 0.3}, step=5)
        mock_wandb_logger.log.assert_called_once_with({"loss": 0.3}, step=5)


# ===================================================================
# Logger.save_model_checkpoint
# ===================================================================
class TestSaveModelCheckpoint:
    """Test checkpoint saving: local files, MLflow artifacts, wandb artifacts."""

    def test_local_checkpoint_written(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-run")
        config = make_config()
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})
        files = list((tmp_path / "model_checkpoints" / "ckpt-run").iterdir())
        assert len(files) == 1, f"Expected 1 checkpoint file, found {len(files)}: {files}"
        assert "model_epoch50" in files[0].name, f"Expected epoch-based name, got {files[0].name}"

    def test_local_checkpoint_skipped_when_disabled(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-skip")
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})
        assert not (tmp_path / "model_checkpoints" / "ckpt-skip").exists()

    def test_mlflow_checkpoint_logged(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="mlf-ckpt")
        config = make_config()
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.3, "acc": 0.9})

        run = mlflow.get_run(lgr.mlflow_run_id)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="checkpoints")
        assert any("model_epoch50" in a.path for a in artifacts)

        assert run.data.tags.get("best_model_epoch") == "50"
        assert "best_model/loss" in run.data.metrics

        assert run.data.metrics["checkpoint/loss"] == pytest.approx(0.3)
        assert run.data.metrics["checkpoint/acc"] == pytest.approx(0.9)

    def test_mlflow_artifact_uploaded_when_local_disabled(
        self, tmp_mlflow_uri, tmp_path, make_config
    ):
        meta = FakeMetaModel(run_name="tmp-upload")
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.4})
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="checkpoints")
        assert len(list(artifacts)) >= 1

    def test_wandb_artifact_logged(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="wb-art")
        config = make_config()
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        lgr._wandb_logger = mock_wandb_logger

        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})

        mock_wandb_logger.save_checkpoint.assert_called_once()

    def test_save_local_models_false_skips_native_model_logging(
        self, tmp_mlflow_uri, tmp_path, make_config
    ):
        meta = FakeMetaModel(run_name="skip-native-model")
        config = make_config(logger={"save_local_models": False})
        lgr = Logger(
            config=config,
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
        )
        with patch("mermaidseg.logger.mlflow.pytorch.log_model") as mock_log_model:
            lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1, "acc": 0.95})
        mock_log_model.assert_not_called()

    def test_wandb_does_not_write_local_checkpoint_when_local_disabled(
        self, tmp_path, make_config, monkeypatch
    ):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        meta = FakeMetaModel(run_name="wandb-no-local")
        config = make_config(logger={"save_local_checkpoints": False})
        with (
            patch("mermaidseg.logger.WANDB_IMPORT_ERROR", None),
            patch("mermaidseg.logger.wandb") as mock_wandb,
            pytest.warns(DeprecationWarning, match="enable_wandb is deprecated"),
        ):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Artifact.return_value = MagicMock()
            lgr = Logger(
                config=config,
                meta_model=meta,
                checkpoint_dir=str(tmp_path),
                log_checkpoint=50,
                enable_mlflow=False,
                enable_wandb=True,
            )
            lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})

        assert not (tmp_path / "model_checkpoints" / "wandb-no-local").exists()

    def test_timestamp_based_name_for_non_checkpoint_epoch(
        self, tmp_mlflow_uri, tmp_path, make_config
    ):
        meta = FakeMetaModel(run_name="ts-name")
        config = make_config()
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=7, metrics_dict={"loss": 0.5})
        files = list((tmp_path / "model_checkpoints" / "ts-name").iterdir())
        assert len(files) == 1
        assert "model_epoch" not in files[0].name


# ===================================================================
# Scenario tests (mirror nbs/test_logger.ipynb scenarios A–D)
# ===================================================================
@pytest.mark.integration
class TestScenarios:
    """End-to-end integration tests covering full logger lifecycle."""

    @staticmethod
    def _run_logger_lifecycle(lgr, meta_model, epochs=3):
        for epoch in range(epochs):
            lgr.log(
                {"train/loss": 1.0 - (epoch * 0.2), "train/accuracy": 0.5 + (epoch * 0.1)},
                step=epoch,
            )
        lgr.save_model_checkpoint(
            meta_model, epoch=epochs - 1, metrics_dict={"accuracy": 0.66, "mean_iou": 0.76}
        )
        lgr.end_run()

    def test_scenario_a_local_only(self, tmp_path, make_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        meta = FakeMetaModel(run_name="scenario_a")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            log_epochs=1,
            log_checkpoint=2,
            checkpoint_dir=str(tmp_path),
            enable_mlflow=False,
            enable_wandb=False,
        )
        self._run_logger_lifecycle(lgr, meta)
        assert lgr.enabled is False
        ckpt_dir = tmp_path / "model_checkpoints" / "scenario_a"
        assert ckpt_dir.exists(), "Checkpoint directory should exist even when MLflow disabled"

    def test_scenario_b_local_plus_mlflow(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="scenario_b")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            log_epochs=1,
            log_checkpoint=2,
            checkpoint_dir=str(tmp_path),
            enable_mlflow=True,
            enable_wandb=False,
        )
        self._run_logger_lifecycle(lgr, meta)
        assert (tmp_path / "model_checkpoints" / "scenario_b").exists()
        assert lgr.mlflow_run_id is not None
        assert mlflow.get_run(lgr.mlflow_run_id).data.metrics["train/loss"] is not None

    def test_scenario_c_local_plus_wandb(self, tmp_path, make_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        meta = FakeMetaModel(run_name="scenario_c")
        with patch("mermaidseg.logger.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Artifact.return_value = MagicMock()
            with pytest.warns(DeprecationWarning, match="enable_wandb is deprecated"):
                lgr = Logger(
                    config=make_config(),
                    meta_model=meta,
                    log_epochs=1,
                    log_checkpoint=2,
                    checkpoint_dir=str(tmp_path),
                    enable_mlflow=False,
                    enable_wandb=True,
                )
            self._run_logger_lifecycle(lgr, meta)
        assert (tmp_path / "model_checkpoints" / "scenario_c").exists()
        assert lgr.enable_wandb is True

    def test_scenario_d_mlflow_unavailable(self, tmp_path, make_config, monkeypatch):
        """MLflow enabled but unreachable — graceful degradation, local checkpoint saved."""
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        meta = FakeMetaModel(run_name="scenario_d")
        lgr = Logger(
            config=make_config(logger={"uri": "http://localhost:9999"}),
            meta_model=meta,
            log_epochs=1,
            log_checkpoint=2,
            checkpoint_dir=str(tmp_path),
            enable_mlflow=True,
            enable_wandb=False,
        )
        assert lgr.enabled is False
        self._run_logger_lifecycle(lgr, meta)
        ckpt_dir = tmp_path / "model_checkpoints" / "scenario_d"
        assert (
            ckpt_dir.exists()
        ), "Bug fix verification: checkpoint must save despite MLflow failure"

    def test_scheduler_state_included_in_checkpoint(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="sched-check")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})
        ckpt_file = list((tmp_path / "model_checkpoints" / "sched-check").iterdir())[0]
        assert "scheduler_state_dict" in torch.load(ckpt_file, weights_only=False)


# ===================================================================
# Logger.end_run & context manager
# ===================================================================
class TestLoggerLifecycle:
    """Test logger cleanup: end_run and context manager protocol."""

    def test_end_run_closes_mlflow(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        assert mlflow.active_run() is not None
        lgr.end_run()
        assert mlflow.active_run() is None

    def test_end_run_closes_wandb(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        lgr._wandb_logger = mock_wandb_logger
        lgr.end_run()
        mock_wandb_logger.end_run.assert_called_once()

    def test_end_run_idempotent(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.end_run()
        lgr.end_run()
        assert mlflow.active_run() is None

    def test_context_manager_cleans_up_on_exception(
        self, tmp_mlflow_uri, make_config, fake_meta_model
    ):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        with pytest.raises(ValueError, match="boom"):
            with lgr:
                raise ValueError("boom")
        assert mlflow.active_run() is None
