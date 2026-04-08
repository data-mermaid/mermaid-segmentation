"""
Serialization strategy:
- Training checkpoints: torch.save (pickle) — full state for resume.
- MLflow best-model logging: torch.save to a temp dir then mlflow.log_artifact.
  We avoid mlflow.pytorch.log_model because its internal session flush triggers
  UniqueViolation on the SageMaker managed MLflow PostgreSQL backend.
  Load with torch.load('model.pt').
- Published models: SafeTensors — zero-copy, no arbitrary code execution.
"""

import contextlib
import copy
import json
import logging
import os
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from typing import Any

import mlflow
import numpy as np
import torch
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader

from mermaidseg.model.meta import MetaModel

logger = logging.getLogger(__name__)


try:
    import wandb

    WANDB_IMPORT_ERROR = None
except ImportError as err:
    wandb = None
    WANDB_IMPORT_ERROR = err

LOCAL_DEFAULT_URI = "./segmentation"


def get_mlflow_tracking_uri(config_uri: str | None = None) -> str:
    """Resolve MLflow tracking URI using a simple priority chain.

    Priority order:
        1. MLFLOW_TRACKING_URI environment variable (highest priority)
        2. config_uri parameter (from YAML config)
        3. Local file store default ("./segmentation")

    Args:
        config_uri: URI from configuration file (typically "segmentation" from YAML)

    Returns:
        str: MLflow tracking URI. Can be:
            - SageMaker MLflow App ARN (arn:aws:sagemaker:...)
            - HTTP(S) URL for remote tracking server
            - Local file path for file-based store

    Examples:
        >>> os.environ["MLFLOW_TRACKING_URI"] = "arn:aws:sagemaker:us-east-1:123:mlflow-app/my-app"
        >>> get_mlflow_tracking_uri("segmentation")
        'arn:aws:sagemaker:us-east-1:123:mlflow-app/my-app'

        >>> del os.environ["MLFLOW_TRACKING_URI"]
        >>> get_mlflow_tracking_uri("segmentation")
        'segmentation'

        >>> get_mlflow_tracking_uri(None)
        './segmentation'
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if config_uri:
        return config_uri

    return LOCAL_DEFAULT_URI


def mlflow_connect(uri: str | None = None) -> timedelta:
    """Set the MLflow tracking URI and verify connectivity.

    Returns the time taken to establish the connection. The connection test
    may take a long time to fail unless ``MLFLOW_HTTP_REQUEST_MAX_RETRIES``
    is set to a low number.
    """
    if uri is None:
        uri = get_mlflow_tracking_uri()

    # Check for sagemaker-mlflow plugin when using SageMaker MLflow App ARN
    if uri and uri.startswith("arn:"):
        try:
            import sagemaker_mlflow  # noqa: F401
        except ImportError:
            logger.warning("URI is a SageMaker ARN but sagemaker-mlflow is not installed. Install with: pip install sagemaker-mlflow")

    mlflow.set_tracking_uri(uri=uri)
    try:
        # Do something to test the server connection.
        time_before_connect = datetime.now()
        mlflow.search_experiments(max_results=1)
    except mlflow.exceptions.MlflowException as e:
        # Note that this may take a long time to reach
        # unless you set MLFLOW_HTTP_REQUEST_MAX_RETRIES to
        # a low number.
        if "Max retries exceeded" in str(e):
            raise RuntimeError("Could not connect to the MLflow tracking server. Is the tracking server up and running?") from e
        # If it's some other kind of MlflowException, just re-raise
        # for debugging purposes.
        raise

    time_after_connect = datetime.now()
    return time_after_connect - time_before_connect


def resume_run(run_id: str) -> mlflow.ActiveRun:
    """Re-attach to an existing MLflow run by run_id after a kernel restart.

    Ensures the tracking URI is set before attaching, so this works as long as
    ``MLFLOW_TRACKING_URI`` is in the environment (e.g. after re-running the
    notebook setup cell).

    Args:
        run_id: The MLflow run ID to resume, as printed after ``Logger`` init
                or visible in the MLflow UI.

    Returns:
        An ``mlflow.ActiveRun`` context manager. Use as::

            with resume_run("abc123") as run:
                mlflow.log_metric("extra_metric", value)

        Or assign it and manage context manually::

            active_run = resume_run("abc123").__enter__()

    Raises:
        RuntimeError: If the run cannot be found at the configured tracking URI.
    """
    uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        return mlflow.start_run(run_id=run_id)
    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(
            f"Could not resume MLflow run '{run_id}' at URI '{uri}'. Verify the run_id and that MLFLOW_TRACKING_URI is set correctly."
        ) from e


class Logger:
    """MLflow-focused logger for experiment tracking during training and evaluation.

    wandb support is deprecated; pass ``enable_wandb=True`` to use the legacy
    ``WandbLogger`` delegate during the deprecation period.
    """

    def __init__(
        self,
        config,
        meta_model,
        log_epochs=5,
        log_checkpoint=50,
        checkpoint_dir=".",
        enable_mlflow=True,
        enable_wandb=False,
        id2label=None,
        id2concept=None,
        save_local_checkpoints=None,
        save_local_models=None,
    ):

        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        if self.log_checkpoint <= 0:
            raise ValueError("log_checkpoint must be > 0")
        self.checkpoint_dir = checkpoint_dir
        self.run_name = meta_model.run_name
        self.enable_mlflow = enable_mlflow
        self.mlflow_run_id = None
        self.enabled = False
        self._wandb_logger = None
        self.id2label = id2label
        self.id2concept = id2concept
        logger_cfg = getattr(config, "logger", None)
        experiment_name = getattr(logger_cfg, "experiment_name", None)
        self._log_system_metrics = getattr(logger_cfg, "system_metrics", True)
        # Keep MLflow system metrics sampling explicit and stable across environments.
        self._system_metrics_interval = getattr(logger_cfg, "system_metrics_sampling_interval", 10)

        # Keep one source of truth for local persistence.
        if save_local_checkpoints is None and save_local_models is not None:
            warnings.warn(
                "save_local_models is deprecated; use save_local_checkpoints.",
                DeprecationWarning,
                stacklevel=2,
            )
            save_local_checkpoints = save_local_models

        if save_local_checkpoints is not None:
            self.save_local_checkpoints = save_local_checkpoints
        else:
            cfg_local = getattr(logger_cfg, "save_local_checkpoints", None)
            if cfg_local is None:
                legacy_cfg = getattr(logger_cfg, "save_local_models", None)
                if legacy_cfg is not None:
                    warnings.warn(
                        "config.logger.save_local_models is deprecated; use config.logger.save_local_checkpoints.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                cfg_local = legacy_cfg
            self.save_local_checkpoints = cfg_local if cfg_local is not None else True

        self.save_local_models = self.save_local_checkpoints

        if enable_mlflow:
            self.enabled = experiment_name is not None

            if not self.enabled:
                logger.info("MLflow logging is disabled (experiment_name not set)")

            if self.enabled:
                try:
                    # Disable PyTorch autologging. When enabled (the default),
                    # MLflow intercepts mlflow.pytorch.log_model and attempts to
                    # re-log training metrics that were already committed, causing
                    # UniqueViolation on the metric_pk in the SageMaker PostgreSQL
                    # backend. Must be called before mlflow_connect so no hooks are
                    # registered before the tracking URI is set.
                    mlflow.pytorch.autolog(disable=True)
                    config_uri = getattr(logger_cfg, "uri", None)
                    tracking_uri = get_mlflow_tracking_uri(config_uri)
                    logger.info("MLflow tracking URI: %s", tracking_uri)
                    duration = mlflow_connect(tracking_uri)
                    logger.info("Connected to MLflow in %s seconds", duration.seconds)
                    mlflow.set_experiment(experiment_name)
                    if self._log_system_metrics and self._system_metrics_interval is not None:
                        mlflow.set_system_metrics_sampling_interval(self._system_metrics_interval)

                    if mlflow.active_run() is None:
                        logger.info("Starting MLflow RUN: %s", self.run_name)
                        run = mlflow.start_run(
                            run_name=self.run_name,
                            log_system_metrics=self._log_system_metrics,
                        )
                        self.mlflow_run_id = run.info.run_id
                    else:
                        logger.info("MLflow run %s already active", self.run_name)
                        self.mlflow_run_id = mlflow.active_run().info.run_id

                    if config is not None:
                        logger.info("Logging config to MLflow...")
                        config_to_log = copy.deepcopy(config)
                        config_to_log["num_classes"] = int(meta_model.num_classes)
                        mlflow.log_dict(config_to_log, "config/config.json")

                        params = {
                            "num_classes": int(meta_model.num_classes),
                            "run_name": self.run_name,
                            "log_epochs": log_epochs,
                            "log_checkpoint": log_checkpoint,
                        }
                        if hasattr(config, "model"):
                            params.update({f"model_{k}": v for k, v in dict(config.model).items() if isinstance(v, str | int | float | bool)})
                        if hasattr(config, "training"):
                            params.update({f"training_{k}": v for k, v in dict(config.training).items() if isinstance(v, str | int | float | bool)})

                        mlflow.log_params(params)

                        tags = {
                            "model_type": meta_model.__class__.__name__,
                            "framework": "pytorch",
                        }
                        mlflow.set_tags(tags)
                        # Log concept metadata
                        self._log_concept_metadata(meta_model)

                except Exception as e:
                    logger.warning("Failed to initialize MLflow logging: %s", e)
                    if self.mlflow_run_id is not None:
                        with contextlib.suppress(Exception):
                            mlflow.end_run()
                        self.mlflow_run_id = None
                    self.enabled = False

        if enable_wandb:
            warnings.warn(
                "enable_wandb is deprecated and will be removed in a future release. Use the MLflow-based Logger workflow instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if WANDB_IMPORT_ERROR is not None:
                logger.warning("wandb is not installed. Wandb logging is disabled.")
            else:
                try:
                    self._wandb_logger = WandbLogger(
                        project=experiment_name or "mermaidseg",
                        run_name=self.run_name,
                        config=config,
                        num_classes=int(meta_model.num_classes),
                        _warn=False,
                    )
                except Exception as e:
                    logger.warning("Failed to initialize wandb logging: %s", e)

    @property
    def _mlflow_active(self) -> bool:
        """True when MLflow logging is enabled and operational."""
        return self.enable_mlflow and self.enabled

    def _log_concept_metadata(self, meta_model: MetaModel) -> None:
        """Log concept-related metadata artifacts when available."""
        if self.id2label:
            mlflow.log_dict(self.id2label, "metadata/id2label.json")
        if self.id2concept:
            mlflow.log_dict(self.id2concept, "metadata/id2concept.json")

        num_concepts = getattr(meta_model, "num_concepts", None)
        if num_concepts is not None:
            mlflow.log_param("num_concepts", int(num_concepts))

        conceptid2labelid = getattr(meta_model, "conceptid2labelid", None)
        if conceptid2labelid is not None:
            mlflow.log_dict(conceptid2labelid, "metadata/conceptid2labelid.json")

        concept_matrix = getattr(meta_model, "concept_matrix", None)
        if concept_matrix is not None:
            self._log_concept_matrix_artifact(concept_matrix)

    def _log_concept_matrix_artifact(self, concept_matrix: Any) -> None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = os.path.join(tmpdir, "concept_matrix.csv")
                concept_matrix.to_csv(csv_path)
                mlflow.log_artifact(csv_path, artifact_path="metadata")
                logger.info("Logged concept matrix to MLflow")
        except Exception as e:
            logger.warning("Failed to log concept matrix to MLflow: %s", e)

    def _unpack_metrics(self, metrics_dict: dict, key_prefix: str = "") -> dict[str, float]:
        """Unpack array-valued metrics into per-class/concept named scalars.

        Array values are expanded using id2concept (for concept metrics) or id2label
        (for class metrics). Scalar values are passed through as-is.

        Args:
            metrics_dict: Raw metrics dict, values may be float or np.ndarray.
            key_prefix: Optional prefix prepended to every key (e.g. "checkpoint").
        """
        prefix_sep = f"{key_prefix}/" if key_prefix else ""
        result: dict[str, float] = {}
        for k, v in (metrics_dict or {}).items():
            if isinstance(v, np.ndarray):
                if "concept" in k.lower():
                    id_map = self.id2concept or {}
                    fallback = "concept"
                else:
                    id_map = self.id2label or {}
                    fallback = "class"
                for idx, val in enumerate(v):
                    name = id_map.get(idx, f"{fallback}_{idx}")
                    result[f"{prefix_sep}{k}/{name}"] = float(val)
            else:
                result[f"{prefix_sep}{k}"] = float(v)
        return result

    def log_dataset(self, dataset, context: str = "training") -> None:
        """Log a single dataset as an MLflow dataset input with metadata tags.

        Args:
            dataset: Dataset instance (expects optional ``annotations_path``,
                ``source_bucket``, ``df_images``, ``num_classes`` when present).
            context: MLflow input context (e.g. ``"training"``).
        """
        if not self._ensure_active_run():
            return
        try:
            source = CodeDatasetSource(
                tags={
                    "annotations_path": getattr(dataset, "annotations_path", ""),
                    "source_bucket": getattr(dataset, "source_bucket", ""),
                    "num_images": str(len(getattr(dataset, "df_images", []))),
                    "num_classes": str(getattr(dataset, "num_classes", "")),
                }
            )
            meta = MetaDataset(
                source=source,
                name=dataset.__class__.__name__,
            )
            mlflow.log_input(meta, context=context)
        except Exception as e:
            logger.warning("Failed to log dataset to MLflow: %s", e)

    def log_datasets(self, dataset, context: str = "training") -> None:
        """Log one or more datasets: unwrap combined/concat wrappers or log as one.

        Args:
            dataset: A single dataset, or a wrapper with ``_datasets`` (e.g. combined)
                or ``datasets`` (e.g. PyTorch ``ConcatDataset``).
            context: MLflow input context passed through to ``log_dataset``.
        """
        sub = getattr(dataset, "_datasets", None) or getattr(dataset, "datasets", None)
        if sub is not None:
            for sub_dataset in sub:
                self.log_dataset(sub_dataset, context=context)
        else:
            self.log_dataset(dataset, context=context)

    def log_benchmark_context(
        self,
        label: str,
        dataset_variant: str | None = None,
    ) -> None:
        """Set MLflow tags enabling before/after benchmark filtering in the UI."""
        if not self._ensure_active_run():
            return
        try:
            git_branch = "unknown"
            git_sha = "unknown"
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            )
            if branch_result.returncode == 0:
                git_branch = branch_result.stdout.strip()
            sha_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            )
            if sha_result.returncode == 0:
                git_sha = sha_result.stdout.strip()

            tags = {
                "benchmark.label": label,
                "benchmark.git_branch": git_branch,
                "benchmark.git_sha": git_sha,
            }
            if dataset_variant is not None:
                tags["benchmark.dataset_variant"] = dataset_variant
            mlflow.set_tags(tags)
        except Exception as e:
            logger.warning("Failed to log benchmark context: %s", e)

    def log_dataloader_params(self, loader: DataLoader, prefix: str = "dataloader") -> None:
        """Log DataLoader configuration as MLflow params."""
        if not self._ensure_active_run():
            return
        try:
            mlflow.log_params(
                {
                    f"{prefix}_num_workers": loader.num_workers,
                    f"{prefix}_pin_memory": loader.pin_memory,
                    f"{prefix}_batch_size": loader.batch_size,
                    f"{prefix}_persistent_workers": getattr(loader, "persistent_workers", False),
                    f"{prefix}_prefetch_factor": getattr(loader, "prefetch_factor", None),
                }
            )
        except Exception as e:
            logger.warning("Failed to log dataloader params: %s", e)

    @property
    def enable_wandb(self) -> bool:
        """True when a ``WandbLogger`` delegate is active (deprecated)."""
        return self._wandb_logger is not None

    def _ensure_active_run(self) -> bool:
        """Guarantee an active MLflow run exists, resuming the original when possible."""
        if not self._mlflow_active:
            return False
        try:
            if mlflow.active_run() is not None:
                return True

            # Try to resume the original run so metrics stay in one place.
            original_run_id = self.mlflow_run_id
            if original_run_id is not None:
                try:
                    run = mlflow.start_run(
                        run_id=original_run_id,
                        log_system_metrics=self._log_system_metrics,
                    )
                    logger.info("Resumed MLflow run %s", original_run_id)
                    return True
                except Exception:
                    logger.warning(
                        "Could not resume MLflow run %s — starting a new run",
                        original_run_id,
                    )

            run = mlflow.start_run(
                run_name=self.run_name,
                log_system_metrics=self._log_system_metrics,
            )
            self.mlflow_run_id = run.info.run_id
            if original_run_id is not None:
                mlflow.set_tag("resumed_from_run_id", original_run_id)
            logger.warning("Started new MLflow run %s (was %s)", self.mlflow_run_id, original_run_id)
            return True
        except Exception as e:
            logger.warning("Failed to ensure active MLflow run: %s", e)
            return False

    def log(self, log_dict, step):
        if self._ensure_active_run():
            try:
                metrics_to_log = self._unpack_metrics(log_dict)
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning("Failed to log metrics to MLflow: %s", e)

        if self._wandb_logger is not None:
            self._wandb_logger.log(log_dict, step=step)

    def save_model_checkpoint(
        self,
        meta_model_run: MetaModel,
        epoch: int,
        metrics_dict: dict[str, float | np.ndarray],
    ):
        """Save a model checkpoint locally and/or to MLflow.

        Local persistence is controlled by ``save_local_checkpoints``.
        MLflow logging is independent of local persistence.

        The MLflow ``best-model`` artifact uses pickle (``torch.save``).
        For external distribution, use ``save_safetensors_for_publish`` instead.

        Checkpoint files are named ``model_epoch{epoch}`` — resuming from a
        previous epoch will overwrite the file unless ``checkpoint_dir`` or
        ``run_name`` differs.
        """
        timestamp = time.strftime("%Y%m%d%H%M%S")

        checkpoint: dict[str, Any] = {
            "config": self.config,
            "model_state_dict": meta_model_run.model.cpu().state_dict(),
            "optimizer_state_dict": meta_model_run.optimizer.state_dict(),
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics_dict,
        }

        meta_model_run.model = meta_model_run.model.to(meta_model_run.device)
        if hasattr(meta_model_run, "scheduler"):
            checkpoint["scheduler_state_dict"] = meta_model_run.scheduler.state_dict()

        model_path = f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/model_epoch{epoch}"

        local_checkpoint_saved = False
        if self.save_local_checkpoints:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if os.path.exists(model_path):
                logger.warning(
                    "Checkpoint file already exists and will be overwritten: %s. "
                    "Use a different checkpoint_dir or run_name when resuming training "
                    "to avoid silently overwriting prior checkpoints.",
                    model_path,
                )
            torch.save(checkpoint, model_path)  # type: ignore
            local_checkpoint_saved = True
            logger.info("Checkpoint saved locally: %s", model_path)
        else:
            logger.info("Local checkpoint saving disabled, skipping disk write")

        if self._ensure_active_run():
            try:
                if local_checkpoint_saved:
                    mlflow.log_artifact(model_path, artifact_path="checkpoints")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = os.path.join(tmpdir, os.path.basename(model_path))
                        torch.save(checkpoint, tmp_path)  # type: ignore
                        mlflow.log_artifact(tmp_path, artifact_path="checkpoints")

                checkpoint_metrics = self._unpack_metrics(metrics_dict, key_prefix="checkpoint")
                mlflow.log_metrics(checkpoint_metrics, step=epoch)

                scalar_metrics = {k: float(v) for k, v in metrics_dict.items() if not isinstance(v, np.ndarray)}

                # Log best model as a plain artifact instead of
                # mlflow.pytorch.log_model() which triggers an internal
                # PostgreSQL session flush on the SageMaker managed MLflow
                # backend, re-inserting already-committed metrics and
                # causing UniqueViolation on metric_pk.
                if local_checkpoint_saved:
                    mlflow.log_artifact(model_path, artifact_path="best-model")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        best_pt = os.path.join(tmpdir, "model.pt")
                        torch.save(checkpoint, best_pt)  # type: ignore
                        mlflow.log_artifact(best_pt, artifact_path="best-model")

                with tempfile.TemporaryDirectory() as tmpdir:
                    metadata_path = os.path.join(tmpdir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(
                            {
                                "epoch": epoch,
                                "timestamp": timestamp,
                                "metrics": scalar_metrics,
                                "run_name": meta_model_run.run_name,
                                "model_class": meta_model_run.model.__class__.__name__,
                                "serialization": "pickle",
                                "load_with": "ckpt = torch.load('model.pt'); model.load_state_dict(ckpt['model_state_dict'])",
                            },
                            f,
                            indent=2,
                        )
                    mlflow.log_artifact(metadata_path, artifact_path="best-model")
                mlflow.set_tag("best_model_logged", "true")
                mlflow.set_tag("best_model_epoch", str(epoch))
                mlflow.log_metrics(
                    {f"best_model/{k}": v for k, v in scalar_metrics.items()},
                    step=epoch,
                )

                logger.info("Checkpoint logged to MLflow (epoch %d): %s", epoch, model_path)
            except Exception as e:
                logger.warning("Failed to log checkpoint/model to MLflow: %s", e)
                with contextlib.suppress(Exception):
                    mlflow.set_tag("best_model_logged", "false")
                    mlflow.set_tag("best_model_log_error", str(e)[:500])

        if self._wandb_logger is not None:
            self._wandb_logger.save_checkpoint(
                model_path=model_path,
                epoch=epoch,
                metrics_dict=metrics_dict,
                checkpoint=checkpoint,
                local_checkpoint_saved=local_checkpoint_saved,
            )

    def save_safetensors_for_publish(
        self,
        meta_model_run: MetaModel,
        epoch: int,
        metrics_dict: dict[str, float | np.ndarray],
        artifact_path: str = "publish",
    ):
        """Export model weights as SafeTensors for secure, pickle-free sharing."""

        if not self._ensure_active_run():
            logger.warning("Cannot save safetensors: no active MLflow run")
            return

        try:
            state_dict = meta_model_run.model.state_dict()
            metadata = {
                "epoch": str(epoch),
                "run_name": meta_model_run.run_name,
                "model_class": meta_model_run.model.__class__.__name__,
                **{k: str(v) for k, v in metrics_dict.items() if not isinstance(v, np.ndarray)},
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                safetensors_path = os.path.join(tmpdir, "model_weights.safetensors")
                metadata_path = os.path.join(tmpdir, "metadata.json")

                save_safetensors(state_dict, safetensors_path, metadata=metadata)

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                mlflow.log_artifact(safetensors_path, artifact_path=artifact_path)
                mlflow.log_artifact(metadata_path, artifact_path=artifact_path)

            logger.info(
                "SafeTensors model exported to MLflow at '%s/model_weights.safetensors'",
                artifact_path,
            )
        except Exception as e:
            logger.warning("Failed to export SafeTensors model: %s", e)

    def end_run(self):
        """Properly end the MLflow run (and wandb delegate, if active).

        Should be called at the end of training to ensure proper cleanup.
        """
        if self.enable_mlflow and self.enabled:
            try:
                if mlflow.active_run() is not None:
                    mlflow.end_run()
                    logger.info("Ended MLflow run: %s", self.run_name)
            except Exception as e:
                logger.warning("Failed to end MLflow run: %s", e)

        if self._wandb_logger is not None:
            self._wandb_logger.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False


class WandbLogger:
    """Deprecated standalone wandb logger.

    .. deprecated::
        ``WandbLogger`` will be removed in a future release.
        Migrate to the MLflow-based :class:`Logger` workflow.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config,
        num_classes: int,
        *,
        _warn: bool = True,
    ):
        if _warn:
            warnings.warn(
                "WandbLogger is deprecated and will be removed in a future release. Use the MLflow-based Logger instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if WANDB_IMPORT_ERROR is not None:
            raise ImportError("wandb is required for WandbLogger but is not installed. Install with: pip install wandb") from WANDB_IMPORT_ERROR

        self.wandb_run = wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        self.wandb_run.config.update({"num_classes": num_classes})

    def log(self, log_dict, step):
        """Log metrics to the active wandb run."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(log_dict, step=step)
        except Exception as e:
            logger.warning("Failed to log metrics to wandb: %s", e)

    def save_checkpoint(
        self,
        model_path: str,
        epoch: int,
        metrics_dict: dict[str, Any],
        checkpoint: dict[str, Any],
        local_checkpoint_saved: bool,
    ):
        """Log a model checkpoint artifact to wandb."""
        if self.wandb_run is None:
            return
        temp_file_path = None
        try:
            artifact_file = model_path
            if not local_checkpoint_saved:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                    temp_file_path = tmp_file.name
                torch.save(checkpoint, temp_file_path)  # type: ignore
                artifact_file = temp_file_path
            artifact = wandb.Artifact(
                f"checkpoint-epoch{epoch}",
                type="model",
                metadata=metrics_dict,
            )
            artifact.add_file(artifact_file)
            self.wandb_run.log_artifact(artifact)
            logger.info("Checkpoint logged to wandb: %s", model_path)
        except Exception as e:
            logger.warning("Failed to log checkpoint to wandb: %s", e)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def end_run(self):
        """Finish the active wandb run."""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
            logger.info("Ended wandb run")
        except Exception as e:
            logger.warning("Failed to end wandb run: %s", e)
