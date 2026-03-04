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
import os
import time
from datetime import datetime, timedelta
from typing import Any

import torch

from mermaidseg.model.meta import MetaModel

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


def get_mlflow_tracking_uri():
    """
    Auto-detect MLflow tracking URI based on execution environment.

    Detects whether code is running on:
    - SageMaker JupyterLab/Notebook instance
    - Local development machine

    Returns:
        str: MLflow tracking URI appropriate for the environment.

    Environment Detection:
        - SageMaker: Checks for /opt/ml/metadata/resource-metadata.json
        - Falls back to MLFLOW_TRACKING_URI environment variable
        - Defaults to "segmentation" for local development
    """
    # Check if running on SageMaker (JupyterLab, Notebook, or Training Job)
    if os.path.exists("/opt/ml/metadata/resource-metadata.json"):
        # SageMaker environment detected
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        return mlflow_uri
    else:
        # Local development environment
        return os.getenv("MLFLOW_TRACKING_URI", "segmentation")


def mlflow_connect(uri: str | None = None) -> timedelta:
    """
    Establish connection to MLflow tracking server and measure connection time.
    Sets the MLflow tracking URI to "segmentation" and tests the connection by
    performing a search operation. Measures and returns the time taken to establish
    the connection.
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
        Attributes:
            config (dict): Stores the configuration dictionary for the experiment.
            log_epochs (int): Frequency of logging epochs.
            log_checkpoint (int): Frequency of logging checkpoints.
            checkpoint_dir (str): Directory to save checkpoints.
            run_name (str): Name of the current run, derived from the meta model.
            enable_mlflow (bool): Flag indicating whether MLflow logging is enabled.
            enable_wandb (bool): Flag indicating whether Weights & Biases logging is enabled.
        """

        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.run_name = meta_model.run_name
        self.enable_mlflow = enable_mlflow
        self.enable_wandb = enable_wandb
        self.mlflow_run_id = None

        if enable_mlflow:
            self.enabled = (
                self.config.logger.experiment_name is not None
                and MLFLOW_IMPORT_ERROR is None
            )

            # If mlflow is available, ensure there is an active run and log basic params/tags.
            if not self.enabled:
                print("MLflow logging is disabled (experiment_name not set or mlflow not installed)")
                return

            try:
                duration = mlflow_connect(self.config.uri)
                print(f"Connected to MLflow in {duration.seconds} seconds")
                mlflow.set_experiment(self.config.logger.experiment_name)
                
                # If no active run, start one
                if mlflow.active_run() is None:
                    print(f"Starting MLflow RUN: {str(self.run_name)}")
                    run = mlflow.start_run(run_name=self.run_name)
                    self.mlflow_run_id = run.info.run_id
                else:
                    print(f"MLflow run {str(self.run_name)} already active")
                    self.mlflow_run_id = mlflow.active_run().info.run_id

                if config is not None:
                    print("Logging config to MLflow...")
                    # Create a copy to avoid mutating input config
                    config_to_log = copy.deepcopy(config)
                    config_to_log["num_classes"] = int(meta_model.num_classes)
                    mlflow.log_dict(config_to_log, "config/config.json")
                    
                    # Log key parameters
                    params = {
                        "num_classes": int(meta_model.num_classes),
                        "run_name": self.run_name,
                        "log_epochs": log_epochs,
                        "log_checkpoint": log_checkpoint,
                    }
                    # Extract model and training params from config if available
                    if hasattr(config, "model"):
                        params.update({f"model_{k}": v for k, v in config.model.items() if isinstance(v, (str, int, float, bool))})
                    if hasattr(config, "training"):
                        params.update({f"training_{k}": v for k, v in config.training.items() if isinstance(v, (str, int, float, bool))})
                    
                    mlflow.log_params(params)
                    
                    # Set useful tags
                    tags = {
                        "model_type": meta_model.__class__.__name__,
                        "framework": "pytorch",
                    }
                    mlflow.set_tags(tags)
                    
            except Exception as e:
                print(f"Warning: Failed to initialize MLflow logging: {e}")
                self.enabled = False

        if enable_wandb:
            if WANDB_IMPORT_ERROR is not None:
                print("Warning: wandb is not installed. Wandb logging is disabled.")
                self.enable_wandb = False
                return
            
            try:
                self.logger = wandb.init(
                    project=self.config.logger.experiment_name,
                    name=self.run_name,
                    config=config,
                )
                self.logger.config.update({"num_classes": int(meta_model.num_classes)})
            except Exception as e:
                print(f"Warning: Failed to initialize wandb logging: {e}")
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
                    print(f"Warning: No active MLflow run, starting new run: {self.run_name}")
                    run = mlflow.start_run(run_name=self.run_name)
                    self.mlflow_run_id = run.info.run_id

                # Batch log all metrics at once for better performance
                metrics_to_log = {k: float(v) for k, v in (log_dict or {}).items()}
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to MLflow: {e}")

        if self.enable_wandb:
            try:
                self.logger.log(log_dict, step=step)
            except Exception as e:
                print(f"Warning: Failed to log metrics to wandb: {e}")

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
        """
        if not self.enable_wandb and (not self.enable_mlflow or not self.enabled):
            return
        
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
        
        # Log checkpoint to MLflow
        if self.enable_mlflow and self.enabled:
            try:
                # Log the checkpoint file as an artifact
                mlflow.log_artifact(model_path, artifact_path="checkpoints")
                
                # Log checkpoint metrics
                checkpoint_metrics = {
                    f"checkpoint/{k}": v for k, v in metrics_dict.items()
                }
                mlflow.log_metrics(checkpoint_metrics, step=epoch)
                
                print(f"Checkpoint logged to MLflow: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to log checkpoint to MLflow: {e}")
    
    def end_run(self):
        """
        Properly end the MLflow and wandb runs.
        Should be called at the end of training to ensure proper cleanup.
        """
        if self.enable_mlflow and self.enabled:
            try:
                if mlflow.active_run() is not None:
                    mlflow.end_run()
                    print(f"Ended MLflow run: {self.run_name}")
            except Exception as e:
                print(f"Warning: Failed to end MLflow run: {e}")
        
        if self.enable_wandb:
            try:
                wandb.finish()
                print(f"Ended wandb run: {self.run_name}")
            except Exception as e:
                print(f"Warning: Failed to end wandb run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures runs are properly closed."""
        self.end_run()
        return False
