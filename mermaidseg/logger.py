"""
title: mermaidseg.logger
abstract: Module that contains the mlflow and checkpoint logging functionality.
author: Viktor Domazetoski
date: 30-10-2025

Classes:
    Logger - A class for logging metrics and configurations to an MLflow tracking server.
    WandbLogger - Deprecated wandb logger; will be removed in a future release.
Functions:
    get_mlflow_tracking_uri() - Auto-detect MLflow URI based on execution environment.
    mlflow_connect() - Connect to the MLflow tracking server and return the connection time.
    save_model_checkpoint() - Save model checkpoints with relevant metadata.
"""

import copy
import logging
import os
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch

from mermaidseg.model.meta import MetaModel

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err

try:
    import wandb

    WANDB_IMPORT_ERROR = None
except ImportError as err:
    WANDB_IMPORT_ERROR = err


LOCAL_DEFAULT_URI = "./segmentation"


def get_mlflow_tracking_uri(config_uri: str | None = None) -> str:
    """
    Resolve MLflow tracking URI using a simple priority chain.

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
    """
    Establish connection to MLflow tracking server and measure connection time.
    Sets the MLflow tracking URI and tests the connection by performing a search
    operation. Measures and returns the time taken to establish the connection.
    Args:
        uri: The MLflow tracking server URI to connect to. Defaults to auto-detected URI.
    Returns:
        timedelta: The time taken to establish the connection to the MLflow server.
    Raises:
        RuntimeError: If the connection to the MLflow tracking server fails due to
                     max retries being exceeded, indicating the server may be down.
        mlflow.exceptions.MlflowException: For other MLflow-related errors that
                                         occur during connection or search operations.
    Note:
        The connection test may take a long time to fail unless
        MLFLOW_HTTP_REQUEST_MAX_RETRIES is set to a low number.
    """
    if uri is None:
        uri = get_mlflow_tracking_uri()

    # Check for sagemaker-mlflow plugin when using SageMaker MLflow App ARN
    if uri and uri.startswith("arn:"):
        try:
            import sagemaker_mlflow  # noqa: F401
        except ImportError:
            logger.warning(
                "URI is a SageMaker ARN but sagemaker-mlflow is not installed. "
                "Install with: pip install sagemaker-mlflow"
            )

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
            raise RuntimeError(
                "Could not connect to the MLflow tracking server."
                " Is the tracking server up and running?"
            ) from e
        # If it's some other kind of MlflowException, just re-raise
        # for debugging purposes.
        raise

    time_after_connect = datetime.now()
    return time_after_connect - time_before_connect


class Logger:
    """
    MLflow-focused logger for experiment tracking during training and evaluation.

    wandb support is deprecated; pass ``enable_wandb=True`` to use the legacy
    ``WandbLogger`` delegate during the deprecation period.
    Attributes:
        config (dict): Configuration dictionary for the logger.
        log_epochs (int): Frequency of logging metrics in terms of epochs.
        log_checkpoint (int): Frequency of saving checkpoints.
        checkpoint_dir (str): Directory to save checkpoints.
    Methods:
        __init__(config, meta_model, log_epochs=5, log_checkpoint=50, checkpoint_dir="."):
            Initializes the Logger instance with the given parameters and sets up the MLflow logger.
        log(log_dict, step):
            Logs a dictionary of metrics to the MLflow logger at a specific step.
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
        """
        Initializes the logger for tracking experiments and benchmarks.
        Args:
            config (dict, optional): Configuration dictionary for the experiment
            meta_model: Meta model object containing model metadata.
            log_epochs (int, optional): Frequency of logging epochs. Defaults to 5.
            log_checkpoint (int, optional): Frequency of logging checkpoints. Defaults to 50.
            checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to ".".
            enable_mlflow (bool, optional): Flag to enable or disable MLflow logging. Defaults to True.
            enable_wandb (bool, optional): **Deprecated.** If True, creates an internal
                ``WandbLogger`` delegate and emits a ``DeprecationWarning``. Defaults to False.
            id2label (dict, optional): Mapping from class IDs to class names for per-class metrics. Defaults to None.
            id2concept (dict, optional): Mapping from concept IDs to concept names for per-concept metrics. Defaults to None.
            save_local_checkpoints (bool, optional): Whether to save checkpoint files to disk.
                Reads from ``config.logger.save_local_checkpoints`` when None. Defaults to True.
            save_local_models (bool, optional): Whether to log native MLflow model artifacts
                (``mlflow.pytorch.log_model``) for best checkpoints. Reads from
                ``config.logger.save_local_models`` when None. Defaults to True.
        Attributes:
            config (dict): Stores the configuration dictionary for the experiment.
            log_epochs (int): Frequency of logging epochs.
            log_checkpoint (int): Frequency of logging checkpoints.
            checkpoint_dir (str): Directory to save checkpoints.
            run_name (str): Name of the current run, derived from the meta model.
            enable_mlflow (bool): Flag indicating whether MLflow logging is enabled.
            id2label (dict): Mapping from class IDs to class names for unpacking array metrics.
            id2concept (dict): Mapping from concept IDs to concept names for unpacking array metrics.
            save_local_checkpoints (bool): Whether to persist checkpoint files locally.
            save_local_models (bool): Whether to log native MLflow model artifacts.
        """

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

        if save_local_checkpoints is not None:
            self.save_local_checkpoints = save_local_checkpoints
        else:
            cfg_val = getattr(getattr(config, "logger", None), "save_local_checkpoints", None)
            self.save_local_checkpoints = cfg_val if cfg_val is not None else True

        if save_local_models is not None:
            self.save_local_models = save_local_models
        else:
            cfg_val = getattr(getattr(config, "logger", None), "save_local_models", None)
            self.save_local_models = cfg_val if cfg_val is not None else True

        if enable_mlflow:
            self.enabled = (
                self.config.logger.experiment_name is not None and MLFLOW_IMPORT_ERROR is None
            )

            if not self.enabled:
                logger.info(
                    "MLflow logging is disabled (experiment_name not set or mlflow not installed)"
                )

            if self.enabled:
                try:
                    config_uri = getattr(self.config.logger, "uri", None)
                    tracking_uri = get_mlflow_tracking_uri(config_uri)
                    logger.info("MLflow tracking URI: %s", tracking_uri)
                    duration = mlflow_connect(tracking_uri)
                    logger.info("Connected to MLflow in %s seconds", duration.seconds)
                    mlflow.set_experiment(self.config.logger.experiment_name)

                    if mlflow.active_run() is None:
                        logger.info("Starting MLflow RUN: %s", self.run_name)
                        run = mlflow.start_run(run_name=self.run_name)
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
                            params.update(
                                {
                                    f"model_{k}": v
                                    for k, v in dict(config.model).items()
                                    if isinstance(v, str | int | float | bool)
                                }
                            )
                        if hasattr(config, "training"):
                            params.update(
                                {
                                    f"training_{k}": v
                                    for k, v in dict(config.training).items()
                                    if isinstance(v, str | int | float | bool)
                                }
                            )

                        mlflow.log_params(params)

                        tags = {
                            "model_type": meta_model.__class__.__name__,
                            "framework": "pytorch",
                        }
                        mlflow.set_tags(tags)

                        # Log concept metadata artifacts if available
                        if self.id2label:
                            mlflow.log_dict(self.id2label, "metadata/id2label.json")
                        if self.id2concept:
                            mlflow.log_dict(self.id2concept, "metadata/id2concept.json")
                        if (
                            hasattr(meta_model, "num_concepts")
                            and meta_model.num_concepts is not None
                        ):
                            mlflow.log_param("num_concepts", int(meta_model.num_concepts))
                        if (
                            hasattr(meta_model, "conceptid2labelid")
                            and meta_model.conceptid2labelid is not None
                        ):
                            mlflow.log_dict(
                                meta_model.conceptid2labelid, "metadata/conceptid2labelid.json"
                            )
                        if (
                            hasattr(meta_model, "concept_matrix")
                            and meta_model.concept_matrix is not None
                        ):
                            concept_matrix_path = None
                            try:
                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix=".csv", delete=False
                                ) as tmp_file:
                                    concept_matrix_path = tmp_file.name
                                meta_model.concept_matrix.to_csv(concept_matrix_path)
                                mlflow.log_artifact(concept_matrix_path, artifact_path="metadata")
                                logger.info("Logged concept matrix to MLflow")
                            except Exception as e:
                                logger.warning("Failed to log concept matrix to MLflow: %s", e)
                            finally:
                                if concept_matrix_path and os.path.exists(concept_matrix_path):
                                    os.remove(concept_matrix_path)

                except Exception as e:
                    logger.warning("Failed to initialize MLflow logging: %s", e)
                    if self.mlflow_run_id is not None:
                        try:
                            mlflow.end_run()
                        except Exception:
                            pass
                        self.mlflow_run_id = None
                    self.enabled = False

        if enable_wandb:
            warnings.warn(
                "enable_wandb is deprecated and will be removed in a future release. "
                "Use the MLflow-based Logger workflow instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if WANDB_IMPORT_ERROR is not None:
                logger.warning("wandb is not installed. Wandb logging is disabled.")
            else:
                try:
                    self._wandb_logger = WandbLogger(
                        project=self.config.logger.experiment_name,
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

    @property
    def enable_wandb(self) -> bool:
        """True when a ``WandbLogger`` delegate is active (deprecated)."""
        return self._wandb_logger is not None

    def _ensure_active_run(self) -> bool:
        """Guarantee an active MLflow run exists. Returns True on success."""
        if not self._mlflow_active:
            return False
        try:
            if mlflow.active_run() is None:
                logger.warning("No active MLflow run, starting new run: %s", self.run_name)
                run = mlflow.start_run(run_name=self.run_name)
                self.mlflow_run_id = run.info.run_id
            return True
        except Exception as e:
            logger.warning("Failed to ensure active MLflow run: %s", e)
            return False

    def log(self, log_dict, step):
        """
        Logs the provided dictionary of metrics along with the current step.
        Args:
            log_dict (Dict[str, Any]): A dictionary containing the log data to be recorded.
            step (int): The current step or iteration associated with the log entry.
        """
        if self._ensure_active_run():
            try:
                # Batch log all metrics at once for better performance
                # Unpack array-valued metrics into per-class or per-concept named scalars
                metrics_to_log = {}
                for k, v in (log_dict or {}).items():
                    if isinstance(v, np.ndarray):
                        # Use id2concept for concept metrics, id2label for class metrics
                        if "concept" in k.lower():
                            id_map = self.id2concept or {}
                            prefix = "concept"
                        else:
                            id_map = self.id2label or {}
                            prefix = "class"

                        for idx, val in enumerate(v):
                            name = id_map.get(idx, f"{prefix}_{idx}")
                            metrics_to_log[f"{k}/{name}"] = float(val)
                    else:
                        metrics_to_log[k] = float(v)

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
        metrics_dict: dict[str, float],
    ):
        """
        Saves a model checkpoint to a specified directory and logs to MLflow.
        Args:
            meta_model_run (MetaModel): An instance of the MetaModel class containing
                the model, optimizer, and optionally a scheduler.
            epoch (int): The current epoch number.
            metrics_dict (Dict[str, float]): A dictionary containing metrics to be saved
                with the checkpoint.
        The checkpoint includes:
            - Configuration settings (`self.config`).
            - Model state dictionary (`model_state_dict`).
            - Optimizer state dictionary (`optimizer_state_dict`).
            - Scheduler state dictionary (`scheduler_state_dict`), if available.
            - Current epoch number.
            - Timestamp of the checkpoint creation.
            - Metrics dictionary.
        Checkpoints are saved in the directory:
            `{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/`
        Naming conventions for the checkpoint file:
            - If the epoch is a multiple of `self.log_checkpoint` and greater than 0:
              ``model_epoch{epoch}``
            - Otherwise:
              ``model_{timestamp}``
        The directory structure is created if it does not already exist.
        Note:
            The model is moved to the CPU before saving its state dictionary.
            Local checkpoint saving is controlled by ``save_local_checkpoints``.
            MLflow logging is independent of local persistence.
        """
        timestamp = time.strftime("%Y%m%d%H")

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

        model_path = (
            f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/model_{timestamp}"
        )
        if epoch % self.log_checkpoint == 0:
            model_path = f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/model_epoch{epoch}"

        # --- Local checkpoint persistence (optional) ---
        local_checkpoint_saved = False
        if self.save_local_checkpoints:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)  # type: ignore
            local_checkpoint_saved = True
            logger.info("Checkpoint saved locally: %s", model_path)
        else:
            logger.info("Local checkpoint saving disabled, skipping disk write")

        # --- MLflow artifact & model logging ---
        if self._ensure_active_run():
            try:
                if local_checkpoint_saved:
                    mlflow.log_artifact(model_path, artifact_path="checkpoints")
                else:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = os.path.join(tmpdir, os.path.basename(model_path))
                        torch.save(checkpoint, tmp_path)  # type: ignore
                        mlflow.log_artifact(tmp_path, artifact_path="checkpoints")

                # Log checkpoint metrics (unpack arrays into per-class or per-concept scalars)
                checkpoint_metrics = {}
                for k, v in metrics_dict.items():
                    if isinstance(v, np.ndarray):
                        if "concept" in k.lower():
                            id_map = self.id2concept or {}
                            prefix = "concept"
                        else:
                            id_map = self.id2label or {}
                            prefix = "class"

                        for idx, val in enumerate(v):
                            name = id_map.get(idx, f"{prefix}_{idx}")
                            checkpoint_metrics[f"checkpoint/{k}/{name}"] = float(val)
                    else:
                        checkpoint_metrics[f"checkpoint/{k}"] = float(v)

                mlflow.log_metrics(checkpoint_metrics, step=epoch)

                scalar_metrics = {
                    k: float(v) for k, v in metrics_dict.items() if not isinstance(v, np.ndarray)
                }
                if self.save_local_models:
                    mlflow.pytorch.log_model(
                        pytorch_model=meta_model_run.model,
                        name="best-model",
                        metadata={
                            "epoch": epoch,
                            "timestamp": timestamp,
                            "metrics": scalar_metrics,
                            "run_name": meta_model_run.run_name,
                        },
                    )
                    mlflow.set_tag("best_model_epoch", str(epoch))
                    mlflow.log_metrics(
                        {f"best_model/{k}": v for k, v in scalar_metrics.items()},
                        step=epoch,
                    )

                logger.info(
                    "Checkpoint + MLflow PyTorch model logged (epoch %d): %s",
                    epoch,
                    model_path,
                )
            except Exception as e:
                logger.warning("Failed to log checkpoint/model to MLflow: %s", e)

        if self._wandb_logger is not None:
            self._wandb_logger.save_checkpoint(
                model_path=model_path,
                epoch=epoch,
                metrics_dict=metrics_dict,
                checkpoint=checkpoint,
                local_checkpoint_saved=local_checkpoint_saved,
            )

    def end_run(self):
        """
        Properly end the MLflow run (and wandb delegate, if active).
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
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures runs are properly closed."""
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
                "WandbLogger is deprecated and will be removed in a future release. "
                "Use the MLflow-based Logger instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if WANDB_IMPORT_ERROR is not None:
            raise ImportError(
                "wandb is required for WandbLogger but is not installed. "
                "Install with: pip install wandb"
            ) from WANDB_IMPORT_ERROR

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
