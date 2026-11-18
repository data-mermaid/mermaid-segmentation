"""Tests for mermaidseg.logger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mermaidseg.logger import (
    Logger,
    WandbLogger,
    prune_rolling_checkpoints,
)
from tests.conftest import FakeMetaModel


def _wandb_logger(lgr: Logger) -> MagicMock:
    """Attach a WandbLogger double to ``lgr`` and return it."""
    lgr._wandb_logger = MagicMock(spec=WandbLogger)
    return lgr._wandb_logger


# ===================================================================
# Logger.__init__
# ===================================================================
class TestLoggerInit:
    """Test Logger initialization: config toggles."""

    def test_log_checkpoint_zero_rejected(self, make_config, fake_meta_model):
        with pytest.raises(ValueError, match="log_checkpoint must be > 0"):
            Logger(config=make_config(), meta_model=fake_meta_model, log_checkpoint=0)


# ===================================================================
# Logger.log
# ===================================================================
class TestLoggerLog:
    """Test metric logging: scalars and ndarrays with id maps."""

    def test_ndarray_class_metrics_unpacked(self, make_config):
        meta = FakeMetaModel()
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, id2label={0: "coral", 1: "sand", 2: "rock"})
        wandb_logger = _wandb_logger(lgr)
        lgr.log({"iou": np.array([0.8, 0.6, 0.7])}, step=1)
        metrics = wandb_logger.log.call_args.args[0]
        assert metrics["iou/coral"] == pytest.approx(0.8)
        assert metrics["iou/sand"] == pytest.approx(0.6)
        assert metrics["iou/rock"] == pytest.approx(0.7)

    def test_ndarray_without_id_map_uses_index(self, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        wandb_logger = _wandb_logger(lgr)
        lgr.log({"dice": np.array([0.1, 0.2])}, step=1)
        metrics = wandb_logger.log.call_args.args[0]
        assert metrics["dice/class_0"] == pytest.approx(0.1)
        assert metrics["dice/class_1"] == pytest.approx(0.2)

    def test_ndarray_concept_metrics_use_id2concept(self, make_config):
        meta = FakeMetaModel()
        config = make_config()
        lgr = Logger(
            config=config,
            meta_model=meta,
            id2label={0: "coral", 1: "sand"},
            id2concept={0: "hard", 1: "soft"},
        )
        wandb_logger = _wandb_logger(lgr)
        lgr.log({"concept_accuracy": np.array([0.9, 0.85])}, step=1)
        metrics = wandb_logger.log.call_args.args[0]
        assert metrics["concept_accuracy/hard"] == pytest.approx(0.9)
        assert metrics["concept_accuracy/soft"] == pytest.approx(0.85)

    def test_wandb_fallback_logging(self, make_config, fake_meta_model):
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        lgr._wandb_logger = mock_wandb_logger

        lgr.log({"loss": 0.3}, step=5)
        mock_wandb_logger.log.assert_called_once_with({"loss": 0.3}, step=5)


class TestBackendFlags:
    """Backend enable flags are config-driven."""

    def test_flags_read_from_config(self, make_config, fake_meta_model):
        config = make_config(logger={"enable_wandb": False})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.enable_wandb is False


class TestBackendDispatch:
    """log_params/log_dict/log_reconciliation reach wandb."""

    def _wandb_only_logger(self, make_config, fake_meta_model):
        config = make_config(logger={"enable_wandb": False})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        lgr._wandb_logger = MagicMock(spec=WandbLogger)
        return lgr

    def test_log_params_dispatches_to_wandb(self, make_config, fake_meta_model):
        lgr = self._wandb_only_logger(make_config, fake_meta_model)
        lgr.log_params({"lr": 0.1, "bs": 4})
        lgr._wandb_logger.log_params.assert_called_once_with({"lr": 0.1, "bs": 4})

    def test_log_dict_dispatches_to_wandb(self, make_config, fake_meta_model):
        lgr = self._wandb_only_logger(make_config, fake_meta_model)
        lgr.log_dict({"0": "coral"}, "metadata/concept_id2name.json")
        lgr._wandb_logger._log_json_artifact.assert_called_once()
        args, _ = lgr._wandb_logger._log_json_artifact.call_args
        assert args[0] == "concept_id2name"
        assert args[1] == {"0": "coral"}

    def test_log_reconciliation_dispatches_to_wandb(self, make_config, fake_meta_model):
        lgr = self._wandb_only_logger(make_config, fake_meta_model)
        registry = MagicMock()
        registry.global_id2source = {0: ("mermaid", "coral")}
        registry.target_id2label = {1: "coral"}
        registry.dataset_offsets = {"mermaid": 0}
        registry.source_to_target = torch.tensor([0, 1], dtype=torch.long)
        registry.concept_id2name = {0: "hard"}
        registry.num_target_classes = 2
        registry.num_global_source_classes = 2

        lgr.log_reconciliation(registry)

        artifact_names = {
            call.args[0] for call in lgr._wandb_logger._log_json_artifact.call_args_list
        }
        assert "target_id2label" in artifact_names
        assert "concept_id2name" in artifact_names
        lgr._wandb_logger.log_params.assert_called_once()


class TestSaveModelCheckpoint:
    """Test checkpoint saving to local files. Checkpoints are not uploaded to wandb."""

    def test_local_checkpoint_written(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-run")
        config = make_config()
        lgr = Logger(
            config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})
        files = list((tmp_path / "model_checkpoints" / "ckpt-run").iterdir())
        assert len(files) == 1, f"Expected 1 checkpoint file, found {len(files)}: {files}"
        assert "model_epoch50" in files[0].name, f"Expected epoch-based name, got {files[0].name}"

    def test_local_checkpoint_skipped_when_disabled(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-skip")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
            save_local_checkpoints=False,
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})
        assert not (tmp_path / "model_checkpoints" / "ckpt-skip").exists()

    def test_step_checkpoint_name_and_resume_flags(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="step-ckpt")
        meta.global_step = 25000
        meta._warmup_iters_completed = 4000
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=1)
        lgr.save_model_checkpoint(
            meta,
            epoch=0,
            metrics_dict={"loss": 0.1},
            checkpoint_name="model_step25000",
            epoch_complete=False,
        )
        path = tmp_path / "model_checkpoints" / "step-ckpt" / "model_step25000"
        assert path.is_file()
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["global_step"] == 25000
        assert ckpt["warmup_iters_completed"] == 4000
        assert ckpt["epoch_complete"] is False
        assert ckpt["epoch"] == 0

    def test_checkpoint_carries_the_loggers_lineage_only_when_set(self, tmp_path, make_config):
        lineage = {"git_sha": "abc", "git_dirty": False}
        for name, kwargs in (("with-lineage", {"lineage": lineage}), ("no-lineage", {})):
            meta = FakeMetaModel(run_name=name)
            lgr = Logger(
                config=make_config(),
                meta_model=meta,
                checkpoint_dir=str(tmp_path),
                log_checkpoint=1,
                **kwargs,
            )
            lgr.save_model_checkpoint(meta, epoch=1, metrics_dict={"loss": 0.1})
        saved = {
            name: torch.load(
                tmp_path / "model_checkpoints" / name / "model_epoch1", weights_only=True
            )
            for name in ("with-lineage", "no-lineage")
        }
        assert saved["with-lineage"]["lineage"] == lineage
        assert "lineage" not in saved["no-lineage"]

    def test_checkpoint_metrics_and_config_load_as_plain_python(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="plain-ckpt")
        lgr = Logger(
            config=make_config(), meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=1
        )
        metrics = {
            "iou": np.array([0.5, 0.25]),
            "acc": np.float32(0.75),
            "loss": torch.tensor(0.125),
            "step": 3,
        }
        lgr.save_model_checkpoint(meta, epoch=1, metrics_dict=metrics)
        path = tmp_path / "model_checkpoints" / "plain-ckpt" / "model_epoch1"
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["metrics"] == {"iou": [0.5, 0.25], "acc": 0.75, "loss": 0.125, "step": 3}
        assert type(ckpt["config"]) is dict
        assert type(ckpt["config"]["training"]) is dict
        assert ckpt["config"]["training"]["optimizer"] == {"type": "SGD", "lr": 0.01}

    def test_save_leaves_parameters_in_place(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="inplace-ckpt")
        weight = meta.model.weight
        ptr = weight.data_ptr()
        device = weight.device
        values = weight.detach().clone()
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=1)
        lgr.save_model_checkpoint(meta, epoch=1, metrics_dict={"loss": 0.1})

        assert meta.model.weight.data_ptr() == ptr
        assert meta.model.weight.device == device
        path = tmp_path / "model_checkpoints" / "inplace-ckpt" / "model_epoch1"
        ckpt = torch.load(path, weights_only=True)
        saved = ckpt["model_state_dict"]["weight"]
        assert saved.device.type == "cpu"
        torch.testing.assert_close(saved, values)

    def test_atomic_checkpoint_leaves_no_partial(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="atomic-ckpt")
        meta.global_step = 1000
        lgr = Logger(config=make_config(), meta_model=meta, checkpoint_dir=str(tmp_path))
        lgr.save_model_checkpoint(
            meta,
            epoch=0,
            metrics_dict={},
            checkpoint_name="model_preempt1000",
            epoch_complete=False,
            atomic=True,
        )
        directory = tmp_path / "model_checkpoints" / "atomic-ckpt"
        final = directory / "model_preempt1000"
        assert final.is_file()
        assert not (directory / "model_preempt1000.partial").exists()
        ckpt = torch.load(final, weights_only=True)
        assert ckpt["global_step"] == 1000
        assert ckpt["epoch_complete"] is False
        assert ckpt["optimizer_state_dict"]
        assert ckpt["scheduler_state_dict"]
        assert ckpt["model_state_dict"]

        (directory / "model_preempt500").write_bytes(b"stale")
        (directory / "model_preempt500.partial").write_bytes(b"partial")
        (directory / "model_step25000").write_bytes(b"keep")
        (directory / "model_epoch0").write_bytes(b"keep")
        prune_rolling_checkpoints(str(directory), keep_name="model_preempt1000")
        assert final.is_file()
        assert (directory / "model_step25000").is_file()
        assert (directory / "model_epoch0").is_file()
        assert not (directory / "model_preempt500").exists()
        assert not (directory / "model_preempt500.partial").exists()

    def test_wandb_run_id_env_resumes(self, monkeypatch):
        monkeypatch.setenv("WANDB_RUN_ID", "r1-cbm_lossw_pix0.5_img0.5")
        with (
            patch("mermaidseg.logger.WANDB_IMPORT_ERROR", None),
            patch("mermaidseg.logger.wandb") as mock_wandb,
        ):
            mock_wandb.init.return_value = MagicMock(id="r1-cbm_lossw_pix0.5_img0.5")
            WandbLogger(
                project="mermaidseg",
                run_name="cbm_lossw_pix0.5_img0.5",
                config={},
                num_classes=3,
            )
        assert mock_wandb.init.call_args.kwargs["id"] == "r1-cbm_lossw_pix0.5_img0.5"
        assert mock_wandb.init.call_args.kwargs["resume"] == "allow"

    def test_wandb_run_id_unset_starts_a_new_run(self, monkeypatch):
        monkeypatch.delenv("WANDB_RUN_ID", raising=False)
        with (
            patch("mermaidseg.logger.WANDB_IMPORT_ERROR", None),
            patch("mermaidseg.logger.wandb") as mock_wandb,
        ):
            mock_wandb.init.return_value = MagicMock(id="fresh")
            WandbLogger(project="mermaidseg", run_name="fresh", config={}, num_classes=3)
        assert mock_wandb.init.call_args.kwargs["id"] is None
        assert mock_wandb.init.call_args.kwargs["resume"] is None

    def test_wandb_does_not_write_local_checkpoint_when_local_disabled(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="wandb-no-local")
        config = make_config(logger={"enable_wandb": True})
        with (
            patch("mermaidseg.logger.WANDB_IMPORT_ERROR", None),
            patch("mermaidseg.logger.wandb") as mock_wandb,
        ):
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Artifact.return_value = MagicMock()
            lgr = Logger(
                config=config,
                meta_model=meta,
                checkpoint_dir=str(tmp_path),
                log_checkpoint=50,
                save_local_checkpoints=False,
            )
            lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})

        assert not (tmp_path / "model_checkpoints" / "wandb-no-local").exists()

    def test_scheduler_state_included_in_checkpoint(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="sched-check")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
        )
        meta.global_step = 25000
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})
        ckpt_file = list((tmp_path / "model_checkpoints" / "sched-check").iterdir())[0]
        checkpoint = torch.load(ckpt_file, weights_only=True)
        assert "scheduler_state_dict" in checkpoint
        assert checkpoint["global_step"] == 25000


# ===================================================================
# Scenario tests (scenarios A–D)
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

    def test_scenario_a_local_only(self, tmp_path, make_config):
        meta = FakeMetaModel(run_name="scenario_a")
        lgr = Logger(
            config=make_config(logger={"enable_wandb": False}),
            meta_model=meta,
            log_checkpoint=2,
            checkpoint_dir=str(tmp_path),
        )
        self._run_logger_lifecycle(lgr, meta)
        ckpt_dir = tmp_path / "model_checkpoints" / "scenario_a"
        assert ckpt_dir.exists()


# ===================================================================
# Logger.end_run & context manager
# ===================================================================
class TestLoggerLifecycle:
    """Test logger cleanup: end_run and context manager protocol."""

    def test_end_run_closes_wandb(self, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        mock_wandb_logger = MagicMock(spec=WandbLogger)
        lgr._wandb_logger = mock_wandb_logger
        lgr.end_run()
        mock_wandb_logger.end_run.assert_called_once()

    def test_context_manager_cleans_up_on_exception(self, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        wandb_logger = _wandb_logger(lgr)
        with pytest.raises(ValueError, match="boom"), lgr:
            raise ValueError("boom")
        wandb_logger.end_run.assert_called_once()
