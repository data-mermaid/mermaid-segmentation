"""Tests for mermaidseg.logger."""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from mermaidseg.logger import (
    LOCAL_DEFAULT_URI,
    Logger,
    _compute_class_by_source,
    _compute_class_counts,
    _compute_source_stats,
    _compute_train_summary,
    _resolve_annotations,
    get_mlflow_tracking_uri,
    mlflow_connect,
)
from tests._dataset_stubs import ConcatStub, make_coralnet_stub, make_mermaid_stub

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
        with (
            patch("mermaidseg.logger.mlflow.set_tracking_uri"),
            patch("mermaidseg.logger.mlflow.search_experiments", side_effect=exc),
            pytest.raises(RuntimeError, match="Could not connect"),
        ):
            mlflow_connect(uri)

    def test_other_mlflow_exception_reraised(self, tmp_path):
        uri = str(tmp_path / "bogus")
        exc = mlflow.exceptions.MlflowException("Something else broke")
        with (
            patch("mermaidseg.logger.mlflow.set_tracking_uri"),
            patch("mermaidseg.logger.mlflow.search_experiments", side_effect=exc),
            pytest.raises(mlflow.exceptions.MlflowException, match="Something else"),
        ):
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
        lgr = Logger(config=config, meta_model=fake_meta_model, log_epochs=2)
        run = mlflow.get_run(lgr.mlflow_run_id)

        params = run.data.params
        assert params["num_classes"] == str(fake_meta_model.num_classes)
        assert params["run_name"] == fake_meta_model.run_name
        assert "model_encoder_name" in params
        assert params["log_epochs"] == "2"

        tags = run.data.tags
        assert tags["model_type"] == "FakeMetaModel"
        assert tags["framework"] == "pytorch"

    def test_model_config_nested_values_logged(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config(
            model={
                "name": "FakeLinear",
                "encoder_name": "resnet18",
                "encoder_kwargs": {"depth": 4, "options": {"pretrained": False}},
            },
        )
        lgr = Logger(config=config, meta_model=fake_meta_model)
        run = mlflow.get_run(lgr.mlflow_run_id)

        params = run.data.params
        assert "model_config" in params
        parsed_config = json.loads(params["model_config"])
        assert parsed_config["encoder_kwargs"]["options"]["pretrained"] is False
        assert params["model_encoder_kwargs_depth"] == "4"

    def test_save_local_checkpoints_mirrors_deprecated_alias(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.save_local_checkpoints is False
        assert lgr.save_local_models is False

    def test_save_local_models_alias_sets_checkpoint_flag(self, tmp_mlflow_uri, make_config, fake_meta_model):
        config = make_config(logger={"save_local_checkpoints": None, "save_local_models": False})
        with pytest.deprecated_call(match="save_local_models is deprecated"):
            lgr = Logger(config=config, meta_model=fake_meta_model)
        assert lgr.save_local_checkpoints is False
        assert lgr.save_local_models is False

    @pytest.mark.parametrize(
        "logger_override",
        [{"experiment_name": None}, None],
        ids=["no-experiment-name", "no-logger-section"],
    )
    def test_disabled_when_experiment_not_configured(self, tmp_mlflow_uri, make_config, fake_meta_model, logger_override):
        config = make_config(logger=logger_override)
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

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="metadata")
        names = {a.path for a in artifacts}
        assert "metadata/id2label.json" in names
        assert "metadata/id2concept.json" in names
        assert "metadata/conceptid2labelid.json" in names

    def test_log_checkpoint_zero_rejected(self, tmp_mlflow_uri, make_config, fake_meta_model):
        with pytest.raises(ValueError, match="log_checkpoint must be > 0"):
            Logger(config=make_config(), meta_model=fake_meta_model, log_checkpoint=0)


# ===================================================================
# Logger._ensure_active_run
# ===================================================================
class TestEnsureActiveRun:
    """Test mid-session run recovery when the MLflow run is dropped."""

    def test_resumes_original_run_after_externally_ended(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        lgr = Logger(config=make_config(), meta_model=meta)
        original_run_id = lgr.mlflow_run_id
        assert mlflow.active_run() is not None

        mlflow.end_run()
        assert mlflow.active_run() is None

        lgr.log({"loss": 0.5}, step=1)
        assert mlflow.active_run() is not None
        assert lgr.mlflow_run_id == original_run_id

    def test_falls_back_to_new_run_when_resume_fails(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        lgr = Logger(config=make_config(), meta_model=meta)
        mlflow.end_run()

        lgr.mlflow_run_id = "nonexistent_run_id_12345"
        lgr.log({"loss": 0.5}, step=1)
        assert mlflow.active_run() is not None
        assert lgr.mlflow_run_id != "nonexistent_run_id_12345"
        tags = mlflow.get_run(lgr.mlflow_run_id).data.tags
        assert tags.get("resumed_from_run_id") == "nonexistent_run_id_12345"

    def test_returns_false_when_mlflow_disabled(self, tmp_path, make_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        meta = FakeMetaModel()
        lgr = Logger(config=make_config(), meta_model=meta, enable_mlflow=False)
        assert lgr._ensure_active_run() is False


# ===================================================================
# Logger.log
# ===================================================================
class TestLoggerLog:
    """Test metric logging: scalars, ndarrays with id maps."""

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

    def test_ndarray_concept_metrics_use_id2concept(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        config = make_config()
        lgr = Logger(
            config=config,
            meta_model=meta,
            id2label={0: "coral", 1: "sand"},
            id2concept={0: "hard", 1: "soft"},
        )
        lgr.log({"concept_accuracy": np.array([0.9, 0.85])}, step=1)
        metrics = mlflow.get_run(lgr.mlflow_run_id).data.metrics
        assert metrics["concept_accuracy/hard"] == pytest.approx(0.9)
        assert metrics["concept_accuracy/soft"] == pytest.approx(0.85)


class TestLogDataset:
    """Test MLflow dataset input logging: single, combined, and graceful fallback."""

    def _make_sub(self, name):
        ds = MagicMock()
        ds.annotations_path = f"s3://bucket/{name}.parquet"
        ds.source_bucket = "bucket"
        ds.df_images = pd.DataFrame({"image_id": [1]})
        ds.num_classes = 3
        ds.__class__ = type(name, (), {})
        return ds

    def test_log_datasets_iterates_sub_datasets(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        lgr = Logger(config=make_config(), meta_model=meta)

        combined = MagicMock()
        combined._datasets = [self._make_sub("DatasetA"), self._make_sub("DatasetB")]

        lgr.log_datasets(combined, context="training")

        inputs = mlflow.get_run(lgr.mlflow_run_id).inputs.dataset_inputs
        assert len(inputs) == 2
        names = {inp.dataset.name for inp in inputs}
        assert names == {"DatasetA", "DatasetB"}

    def test_log_datasets_supports_concat_dataset_attr(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel()
        lgr = Logger(config=make_config(), meta_model=meta)

        combined = MagicMock(spec=[])
        combined.datasets = [self._make_sub("SourceA"), self._make_sub("SourceB")]

        lgr.log_datasets(combined, context="training")

        inputs = mlflow.get_run(lgr.mlflow_run_id).inputs.dataset_inputs
        assert len(inputs) == 2
        names = {inp.dataset.name for inp in inputs}
        assert names == {"SourceA", "SourceB"}


# ===================================================================
# Logger.save_model_checkpoint
# ===================================================================
class TestSaveModelCheckpoint:
    """Test checkpoint saving: local files and MLflow artifacts."""

    def test_local_checkpoint_written(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-run")
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50)
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1})
        files = list((tmp_path / "model_checkpoints" / "ckpt-run").iterdir())
        assert len(files) == 1, f"Expected 1 checkpoint file, found {len(files)}: {files}"
        assert "model_epoch50" in files[0].name, f"Expected epoch-based name, got {files[0].name}"

    def test_local_checkpoint_skipped_when_disabled(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="ckpt-skip")
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50)
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.2})
        assert not (tmp_path / "model_checkpoints" / "ckpt-skip").exists()

    def test_mlflow_checkpoint_logged(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="mlf-ckpt")
        config = make_config()
        lgr = Logger(config=config, meta_model=meta, checkpoint_dir=str(tmp_path), log_checkpoint=50)
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.3, "acc": 0.9})

        run = mlflow.get_run(lgr.mlflow_run_id)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="checkpoints")
        assert any("model_epoch50" in a.path for a in artifacts)

        assert run.data.tags.get("best_model_epoch") == "50"
        assert "best_model/loss" in run.data.metrics

        assert run.data.metrics["checkpoint/loss"] == pytest.approx(0.3)
        assert run.data.metrics["checkpoint/acc"] == pytest.approx(0.9)

    def test_mlflow_model_logged_when_local_disabled(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="mlflow-no-local")
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(
            config=config,
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
        )
        with patch("mermaidseg.logger.mlflow.log_artifact") as mock_log_artifact:
            lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1, "acc": 0.95})
        best_model_calls = [c for c in mock_log_artifact.call_args_list if "best-model" in str(c)]
        assert len(best_model_calls) >= 1

    def test_model_logging_sets_best_model_tag(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="verify-model-tag")
        config = make_config(logger={"save_local_checkpoints": False})
        lgr = Logger(
            config=config,
            meta_model=meta,
            checkpoint_dir=str(tmp_path),
            log_checkpoint=50,
        )
        lgr.save_model_checkpoint(meta, epoch=50, metrics_dict={"loss": 0.1, "acc": 0.95})
        run = mlflow.get_run(lgr.mlflow_run_id)
        assert run.data.tags["best_model_logged"] == "true"
        assert run.data.tags["best_model_epoch"] == "50"

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
# Logger.save_safetensors_for_publish
# ===================================================================
class TestSaveSafetensorsForPublish:
    """Test SafeTensors export: artifacts, metadata, error handling."""

    def test_safetensors_artifacts_logged(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel(run_name="st-publish")
        lgr = Logger(config=make_config(), meta_model=meta)
        lgr.save_safetensors_for_publish(meta, epoch=10, metrics_dict={"loss": 0.15, "acc": 0.92})

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="publish")
        names = {a.path for a in artifacts}
        assert "publish/model_weights.safetensors" in names
        assert "publish/metadata.json" in names

    def test_safetensors_metadata_content(self, tmp_mlflow_uri, tmp_path, make_config):
        meta = FakeMetaModel(run_name="st-meta")
        lgr = Logger(config=make_config(), meta_model=meta)
        lgr.save_safetensors_for_publish(meta, epoch=5, metrics_dict={"loss": 0.2, "iou": np.array([0.8, 0.7, 0.6])})

        client = mlflow.tracking.MlflowClient()
        local = client.download_artifacts(lgr.mlflow_run_id, "publish/metadata.json", str(tmp_path))
        with open(local) as f:
            md = json.load(f)
        assert md["epoch"] == "5"
        assert md["run_name"] == "st-meta"
        assert md["model_class"] == "Linear"
        assert md["loss"] == "0.2"
        assert "iou" not in md

    def test_safetensors_custom_artifact_path(self, tmp_mlflow_uri, make_config):
        meta = FakeMetaModel(run_name="st-custom")
        lgr = Logger(config=make_config(), meta_model=meta)
        lgr.save_safetensors_for_publish(meta, epoch=1, metrics_dict={"loss": 0.3}, artifact_path="custom/export")

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(lgr.mlflow_run_id, path="custom/export")
        names = {a.path for a in artifacts}
        assert "custom/export/model_weights.safetensors" in names

    def test_safetensors_noop_when_mlflow_disabled(self, tmp_path, make_config, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        meta = FakeMetaModel(run_name="st-noop")
        lgr = Logger(
            config=make_config(),
            meta_model=meta,
            enable_mlflow=False,
        )
        lgr.save_safetensors_for_publish(meta, epoch=1, metrics_dict={"loss": 0.5})
        assert lgr.mlflow_run_id is None


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

    def test_end_run_idempotent(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.end_run()
        lgr.end_run()
        assert mlflow.active_run() is None

    def test_context_manager_cleans_up_on_exception(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        with pytest.raises(ValueError, match="boom"), lgr:
            raise ValueError("boom")
        assert mlflow.active_run() is None


# ===================================================================
# Logger.log_benchmark_context
# ===================================================================
class TestLogBenchmarkContext:
    """Test benchmark tag logging for before/after comparison filtering."""

    def test_sets_required_tags(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.log_benchmark_context(label="baseline")
        tags = mlflow.get_run(lgr.mlflow_run_id).data.tags
        assert tags["benchmark.label"] == "baseline"
        assert "benchmark.git_branch" in tags
        assert "benchmark.git_sha" in tags

    def test_dataset_variant_set_when_provided(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.log_benchmark_context(label="optimized", dataset_variant="mermaid_only")
        tags = mlflow.get_run(lgr.mlflow_run_id).data.tags
        assert tags["benchmark.dataset_variant"] == "mermaid_only"

    def test_dataset_variant_absent_when_not_provided(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.log_benchmark_context(label="baseline")
        tags = mlflow.get_run(lgr.mlflow_run_id).data.tags
        assert "benchmark.dataset_variant" not in tags

    def test_noop_when_mlflow_disabled(self, make_config, monkeypatch, fake_meta_model):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        lgr = Logger(config=make_config(logger={"experiment_name": None}), meta_model=fake_meta_model)
        lgr.log_benchmark_context(label="test")
        assert lgr.mlflow_run_id is None


# ===================================================================
# Logger.log_dataloader_params
# ===================================================================
class TestLogDataloaderParams:
    """Test DataLoader configuration logging as MLflow params."""

    def _make_loader(self, batch_size: int = 4, num_workers: int = 0) -> DataLoader:
        ds = TensorDataset(
            torch.zeros(8, 3, 4, 4),
            torch.zeros(8, 4, 4, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    def test_logs_all_expected_keys(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.log_dataloader_params(self._make_loader())
        params = mlflow.get_run(lgr.mlflow_run_id).data.params
        for key in ("dataloader_batch_size", "dataloader_num_workers", "dataloader_pin_memory", "dataloader_persistent_workers"):
            assert key in params, f"Missing expected param: {key}"

    def test_custom_prefix(self, tmp_mlflow_uri, make_config, fake_meta_model):
        lgr = Logger(config=make_config(), meta_model=fake_meta_model)
        lgr.log_dataloader_params(self._make_loader(), prefix="train_loader")
        params = mlflow.get_run(lgr.mlflow_run_id).data.params
        assert "train_loader_batch_size" in params

    def test_noop_when_mlflow_disabled(self, make_config, monkeypatch, fake_meta_model):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        lgr = Logger(config=make_config(logger={"experiment_name": None}), meta_model=fake_meta_model)
        lgr.log_dataloader_params(self._make_loader())
        assert lgr.mlflow_run_id is None


# ===================================================================
# _resolve_annotations
# ===================================================================
class TestResolveAnnotations:
    def test_plain_dataset_returns_attributes_unchanged(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Porites"],
                "img-2": ["Macroalgae"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Indonesia"},
            class_subset=["Acropora", "Porites", "Macroalgae"],
        )
        df_ann, df_img, id2label = _resolve_annotations(stub)
        assert df_ann is stub.df_annotations
        assert df_img is stub.df_images
        assert id2label is stub.id2label

    def test_subset_filters_annotations_by_subset_image_ids(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
                "img-3": ["Macroalgae"],
            },
            image_to_region={"img-1": "A", "img-2": "B", "img-3": "C"},
            class_subset=["Acropora", "Porites", "Macroalgae"],
        )

        subset = Subset(stub, [0, 2])  # img-1 and img-3
        df_ann, df_img, id2label = _resolve_annotations(subset)

        assert sorted(df_ann["image_id"].unique().tolist()) == ["img-1", "img-3"]
        assert sorted(df_img["image_id"].tolist()) == ["img-1", "img-3"]
        assert id2label == stub.id2label

    def test_concat_wrapper_concatenates_children(self):
        stub_a = make_mermaid_stub(
            image_to_classes={"a-1": ["Acropora"]},
            image_to_region={"a-1": "Indonesia"},
            class_subset=["Acropora", "Porites"],
        )
        stub_b = make_mermaid_stub(
            image_to_classes={"b-1": ["Porites"]},
            image_to_region={"b-1": "Caribbean"},
            class_subset=["Acropora", "Porites"],
        )

        wrapper = ConcatStub(_datasets=[stub_a, stub_b])

        df_ann, df_img, id2label = _resolve_annotations(wrapper)
        assert sorted(df_ann["image_id"].tolist()) == ["a-1", "b-1"]
        assert sorted(df_img["image_id"].tolist()) == ["a-1", "b-1"]
        assert id2label == stub_a.id2label

    def test_concat_wrapper_warns_on_id2label_mismatch(self, caplog):
        import logging

        stub_a = make_mermaid_stub(
            image_to_classes={"a-1": ["Acropora"]},
            image_to_region={"a-1": "X"},
            class_subset=["Acropora"],
        )
        stub_b = make_mermaid_stub(
            image_to_classes={"b-1": ["Porites"]},
            image_to_region={"b-1": "Y"},
            class_subset=["Porites"],
        )

        wrapper = ConcatStub(_datasets=[stub_a, stub_b])

        with caplog.at_level(logging.WARNING, logger="mermaidseg.logger"):
            _, _, id2label = _resolve_annotations(wrapper)

        assert id2label == stub_a.id2label  # first child wins
        assert any("id2label mismatch" in r.message for r in caplog.records)

    def test_unknown_shape_returns_none_and_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="mermaidseg.logger"):
            result = _resolve_annotations(object())
        assert result is None
        assert any("unsupported split shape" in r.message.lower() for r in caplog.records)


# ===================================================================
# _compute_class_counts
# ===================================================================
class TestComputeClassCounts:
    def _splits(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora"],
                "img-3": ["Other"],  # unclassified row
                "img-4": ["Porites"],
            },
            image_to_region={"img-1": "A", "img-2": "A", "img-3": "B", "img-4": "B"},
            class_subset=["Acropora", "Porites", "Other"],
        )

        return {
            "train": Subset(stub, [0, 1]),  # img-1, img-2
            "val": Subset(stub, [2]),  # img-3
            "test": Subset(stub, [3]),  # img-4
        }, stub

    def test_class_counts_schema_and_values(self):
        splits, stub = self._splits()
        resolved = {k: _resolve_annotations(v) for k, v in splits.items()}

        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)

        assert list(df.columns) == [
            "class_id",
            "class_name",
            "class_kind",
            "train_annotations",
            "val_annotations",
            "test_annotations",
            "train_images",
            "val_images",
            "test_images",
            "train_fraction",
            "val_fraction",
            "test_fraction",
        ]
        # Background row plus three classes.
        assert df["class_id"].tolist() == [0, 1, 2, 3]
        assert df["class_name"].tolist() == ["background", "Acropora", "Porites", "Other"]
        assert df["class_kind"].tolist() == ["background", "target", "target", "unclassified"]

        acropora = df.set_index("class_name").loc["Acropora"]
        assert acropora["train_annotations"] == 3  # 2 + 1
        assert acropora["val_annotations"] == 0
        assert acropora["test_annotations"] == 0
        assert acropora["train_images"] == 2

        # Fractions sum to 1.0 within each split (zero rows summed to total)
        assert abs(df["train_fraction"].sum() - 1.0) < 1e-9

    def test_class_kind_qualified_other_stays_target(self):
        stub = make_mermaid_stub(
            image_to_classes={"img-1": ["Other Invertebrates"]},
            image_to_region={"img-1": "A"},
            class_subset=["Other Invertebrates"],
        )

        resolved = {"train": _resolve_annotations(Subset(stub, [0]))}
        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)
        kind = df.set_index("class_name").loc["Other Invertebrates", "class_kind"]
        assert kind == "target"

    def test_empty_split_yields_zero_row_not_missing(self):
        splits, stub = self._splits()
        # Drop the 'val' split entirely.
        resolved = {"train": _resolve_annotations(splits["train"])}
        df = _compute_class_counts(resolved, parent_id2label=stub.id2label)
        # No val/test columns should appear in the schema when those splits aren't passed.
        for col in df.columns:
            assert "val_" not in col and "test_" not in col
        # Train-only fractions still sum to 1.
        assert abs(df["train_fraction"].sum() - 1.0) < 1e-9


# ===================================================================
# _compute_source_stats
# ===================================================================
class TestComputeSourceStats:
    def test_mermaid_region_rows(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Acropora", "Porites"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            class_subset=["Acropora", "Porites"],
        )

        resolved = {
            "train": _resolve_annotations(Subset(stub, [0])),
            "val": _resolve_annotations(Subset(stub, [1])),
        }
        df = _compute_source_stats(resolved)

        assert set(df["source_type"]) == {"region"}
        idx = df.set_index("source_key")
        assert idx.loc["Indonesia", "train_images"] == 1
        assert idx.loc["Indonesia", "val_images"] == 0
        assert idx.loc["Caribbean", "val_annotations"] == 2
        assert "test_images" not in df.columns  # test split not provided

    def test_coralnet_source_rows_cast_to_str(self):

        stub = make_coralnet_stub(
            image_to_classes={"img-1": ["Acropora"], "img-2": ["Porites"]},
            image_to_source={"img-1": 42, "img-2": 7},
            class_subset=["Acropora", "Porites"],
        )

        resolved = {"train": _resolve_annotations(Subset(stub, [0, 1]))}
        df = _compute_source_stats(resolved)

        assert set(df["source_type"]) == {"source"}
        assert set(df["source_key"]) == {"42", "7"}
        assert df["source_key"].dtype == object  # strings, not ints


# ===================================================================
# _compute_class_by_source
# ===================================================================
class TestComputeClassBySource:
    def test_long_format_omits_zero_rows(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
            },
            image_to_region={"img-1": "Indonesia", "img-2": "Caribbean"},
            class_subset=["Acropora", "Porites"],
        )

        resolved = {
            "train": _resolve_annotations(Subset(stub, [0])),
            "val": _resolve_annotations(Subset(stub, [1])),
        }
        df = _compute_class_by_source(resolved, parent_id2label=stub.id2label)

        assert list(df.columns) == [
            "source_key",
            "source_type",
            "class_id",
            "class_name",
            "split",
            "annotations",
            "images",
        ]
        # No zero-count rows: only (Indonesia, Acropora, train) and (Caribbean, Porites, val)
        assert len(df) == 2
        assert df.set_index(["source_key", "class_name", "split"]).loc[("Indonesia", "Acropora", "train"), "annotations"] == 1


# ===================================================================
# _compute_train_summary
# ===================================================================
class TestComputeTrainSummary:
    def test_summary_metrics_and_distribution(self):
        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora", "Acropora", "Porites"],
                "img-2": ["Acropora"],
                "img-3": ["Macroalgae"],
                "img-4": [],  # an image with zero annotations (loss-of-points scenario)
            },
            image_to_region={"img-1": "A", "img-2": "A", "img-3": "B", "img-4": "B"},
            class_subset=["Acropora", "Porites", "Macroalgae"],
        )
        # Re-add the zero-annotation image — make_mermaid_stub drops images with no rows
        # because df_images is built from df_annotations. Patch by hand:
        stub.df_images = pd.concat(
            [stub.df_images, pd.DataFrame([{"image_id": "img-4", "region_id": "B", "region_name": "B"}])],
            ignore_index=True,
        )

        resolved = {
            "train": _resolve_annotations(Subset(stub, [0, 1])),  # img-1, img-2
            "val": _resolve_annotations(Subset(stub, [2])),  # img-3
            "test": _resolve_annotations(Subset(stub, [3])),  # img-4 (no anns)
        }

        summary = _compute_train_summary(resolved, parent_id2label=stub.id2label, class_subset=["Acropora", "Porites", "Macroalgae"])

        assert summary["total_images"] == 4
        assert summary["total_annotations"] == 5
        assert summary["splits"]["train"] == {"images": 2, "annotations": 4}
        assert summary["splits"]["test"] == {"images": 1, "annotations": 0}
        assert summary["class_subset"] == ["Acropora", "Porites", "Macroalgae"]
        assert summary["num_classes"] == 4
        # eligible = target classes only (excludes background and unclassified).
        # Class subset is ["Acropora", "Porites", "Macroalgae"] — all targets.
        assert summary["eligible_num_classes"] == 3

        # top-K shares: train has Acropora=3, Porites=1 → top1=0.75
        assert abs(summary["top1_share"] - 0.75) < 1e-9
        # Only 2 eligible classes in train → top3 / top5 clamp to total = 1.0
        assert abs(summary["top3_share"] - 1.0) < 1e-9
        assert abs(summary["top5_share"] - 1.0) < 1e-9

        # effective_num_classes = exp(entropy), bounded above by eligible_num_classes
        assert 1.0 <= summary["effective_num_classes"] <= summary["eligible_num_classes"]

        # Annotations-per-image distribution catches the zero-annotation image
        assert summary["annotations_per_image"]["test"]["min"] == 0
        assert summary["annotations_per_image"]["train"]["max"] == 3
        assert summary["annotations_per_image"]["train"]["mean"] == 2.0


# ===================================================================
# Logger.log_dataset_statistics
# ===================================================================
class TestLogDatasetStatistics:
    def test_happy_path_writes_four_artifacts(self, tmp_mlflow_uri, make_config, fake_meta_model, tmp_path):

        stub = make_mermaid_stub(
            image_to_classes={
                "img-1": ["Acropora"],
                "img-2": ["Porites"],
                "img-3": ["Acropora", "Porites"],
            },
            image_to_region={"img-1": "A", "img-2": "B", "img-3": "B"},
            class_subset=["Acropora", "Porites"],
        )
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)

        try:
            lgr.log_dataset_statistics(
                {
                    "train": Subset(stub, [0]),
                    "val": Subset(stub, [1]),
                    "test": Subset(stub, [2]),
                }
            )
        finally:
            run_id = lgr.mlflow_run_id
            lgr.end_run()

        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(run_id, "dataset_stats")}
        assert artifacts == {
            "dataset_stats/class_counts.csv",
            "dataset_stats/source_stats.csv",
            "dataset_stats/class_by_source.csv",
            "dataset_stats/train_summary.yaml",
        }

    def test_artifact_failure_does_not_drop_others(self, tmp_mlflow_uri, make_config, fake_meta_model, caplog):
        from unittest.mock import patch

        stub = make_mermaid_stub(
            image_to_classes={"img-1": ["Acropora"]},
            image_to_region={"img-1": "A"},
            class_subset=["Acropora"],
        )
        config = make_config()
        lgr = Logger(config=config, meta_model=fake_meta_model)

        # Make log_text fail only for class_counts.csv; let the others through.
        real_log_text = mlflow.log_text

        def fake_log_text(text, artifact_file):
            if artifact_file.endswith("class_counts.csv"):
                raise RuntimeError("boom")
            return real_log_text(text, artifact_file)

        try:
            with patch("mermaidseg.logger.mlflow.log_text", side_effect=fake_log_text):
                lgr.log_dataset_statistics({"train": Subset(stub, [0])})
            run_id = lgr.mlflow_run_id
        finally:
            lgr.end_run()

        client = mlflow.MlflowClient()
        artifacts = {a.path for a in client.list_artifacts(run_id, "dataset_stats")}
        # class_counts failed; the other three should still be present.
        assert "dataset_stats/class_counts.csv" not in artifacts
        assert "dataset_stats/source_stats.csv" in artifacts
        assert "dataset_stats/class_by_source.csv" in artifacts
        assert "dataset_stats/train_summary.yaml" in artifacts
