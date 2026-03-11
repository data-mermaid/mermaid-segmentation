"""
title: mermaidseg.logger
abstract: Module that contains the mlflow and checkpoint logging functionality.
author: Viktor Domazetoski
date: 30-10-2025

Classes:
    Logger - A class for logging metrics and configurations to an MLflow tracking server.
Functions:
    get_mlflow_tracking_uri() - Auto-detect MLflow URI based on execution environment.
    mlflow_connect() - Connect to the MLflow tracking server and return the connection time.
    save_model_checkpoint() - Save model checkpoints with relevant metadata.
"""

import copy
import logging
import os
import time
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
    Logger class for managing mlflow logging and visualization during training and evaluation.
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
            enable_wandb (bool, optional): Flag to enable or disable Weights & Biases logging. Defaults to False.
            id2label (dict, optional): Mapping from class IDs to class names for per-class metrics. Defaults to None.
        Attributes:
            config (dict): Stores the configuration dictionary for the experiment.
            log_epochs (int): Frequency of logging epochs.
            log_checkpoint (int): Frequency of logging checkpoints.
            checkpoint_dir (str): Directory to save checkpoints.
            run_name (str): Name of the current run, derived from the meta model.
            enable_mlflow (bool): Flag indicating whether MLflow logging is enabled.
            enable_wandb (bool): Flag indicating whether Weights & Biases logging is enabled.
            id2label (dict): Mapping from class IDs to class names for unpacking array metrics.
        """

        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.run_name = meta_model.run_name
        self.enable_mlflow = enable_mlflow
        self.enable_wandb = enable_wandb
        self.mlflow_run_id = None
        self.enabled = False
        self.wandb_run = None
        self.id2label = id2label

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
                    config_uri = getattr(self.config, "uri", None)
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
            if WANDB_IMPORT_ERROR is not None:
                logger.warning("wandb is not installed. Wandb logging is disabled.")
                self.enable_wandb = False
            else:
                try:
                    self.wandb_run = wandb.init(
                        project=self.config.logger.experiment_name,
                        name=self.run_name,
                        config=config,
                    )
                    self.wandb_run.config.update({"num_classes": int(meta_model.num_classes)})
                except Exception as e:
                    logger.warning("Failed to initialize wandb logging: %s", e)
                    self.enable_wandb = False

    def log(self, log_dict, step):
        """
        Logs the provided dictionary of metrics along with the current step.
        Args:
            log_dict (Dict[str, Any]): A dictionary containing the log data to be recorded.
            step (int): The current step or iteration associated with the log entry.
        """
        if self.enable_mlflow and self.enabled:
            try:
                # Ensure there is an active mlflow run while logging metrics/artifacts
                if mlflow.active_run() is None:
                    logger.warning("No active MLflow run, starting new run: %s", self.run_name)
                    run = mlflow.start_run(run_name=self.run_name)
                    self.mlflow_run_id = run.info.run_id

                # Batch log all metrics at once for better performance
                # Unpack array-valued metrics into per-class named scalars
                metrics_to_log = {}
                for k, v in (log_dict or {}).items():
                    if isinstance(v, np.ndarray):
                        for class_id, class_val in enumerate(v):
                            class_name = (self.id2label or {}).get(class_id, f"class_{class_id}")
                            metrics_to_log[f"{k}/{class_name}"] = float(class_val)
                    else:
                        metrics_to_log[k] = float(v)

                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning("Failed to log metrics to MLflow: %s", e)

        if self.enable_wandb and self.wandb_run is not None:
            try:
                self.wandb_run.log(log_dict, step=step)
            except Exception as e:
                logger.warning("Failed to log metrics to wandb: %s", e)

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
              `model_epoch{epoch}`
            - Otherwise:
              `model_{timestamp}`
        The directory structure is created if it does not already exist.
        Note:
            The model is moved to the CPU before saving its state dictionary.
            Local checkpoint is always saved; MLflow upload is optional.
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

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(checkpoint, model_path)  # type: ignore
        logger.info("Checkpoint saved locally: %s", model_path)

        # Log checkpoint to MLflow if enabled
        if self.enable_mlflow and self.enabled:
            try:
                # Log the checkpoint file as an artifact
                mlflow.log_artifact(model_path, artifact_path="checkpoints")

                # Log checkpoint metrics
                checkpoint_metrics = {f"checkpoint/{k}": v for k, v in metrics_dict.items()}
                mlflow.log_metrics(checkpoint_metrics, step=epoch)

                logger.info("Checkpoint logged to MLflow: %s", model_path)
            except Exception as e:
                logger.warning("Failed to log checkpoint to MLflow: %s", e)

        if self.enable_wandb and self.wandb_run is not None:
            try:
                artifact = wandb.Artifact(
                    f"checkpoint-epoch{epoch}",
                    type="model",
                    metadata=metrics_dict,
                )
                artifact.add_file(model_path)
                self.wandb_run.log_artifact(artifact)
                logger.info("Checkpoint logged to wandb: %s", model_path)
            except Exception as e:
                logger.warning("Failed to log checkpoint to wandb: %s", e)

    def end_run(self):
        """
        Properly end the MLflow and wandb runs.
        Should be called at the end of training to ensure proper cleanup.
        """
        if self.enable_mlflow and self.enabled:
            try:
                if mlflow.active_run() is not None:
                    mlflow.end_run()
                    logger.info("Ended MLflow run: %s", self.run_name)
            except Exception as e:
                logger.warning("Failed to end MLflow run: %s", e)

        if self.enable_wandb and self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                logger.info("Ended wandb run: %s", self.run_name)
            except Exception as e:
                logger.warning("Failed to end wandb run: %s", e)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures runs are properly closed."""
        self.end_run()
        return False
