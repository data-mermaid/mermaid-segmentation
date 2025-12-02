"""
title: mermaidseg.logger
abstract: Module that contains the mlflow and checkpoint logging functionality.
author: Viktor Domazetoski
date: 30-10-2025

Classes:
    Logger - A class for logging metrics and configurations to an MLflow tracking server.
Functions:
    mlflow_connect() - Connect to the MLflow tracking server and return the connection time.
    save_model_checkpoint() - Save model checkpoints with relevant metadata.
"""

from typing import Any, Dict

try:
    import mlflow
    from mlflow.models import infer_signature

    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err

try:
    import wandb

    WANDB_IMPORT_ERROR = None
except ImportError as err:
    WANDB_IMPORT_ERROR = err

import os
import time
from datetime import datetime, timedelta

import torch
from mermaidseg.model.meta import MetaModel

URI = "segmentation"  # Update as an argument in config


def mlflow_connect(uri=URI) -> timedelta:
    """
    Establish connection to MLflow tracking server and measure connection time.
    Sets the MLflow tracking URI to "segmentation" and tests the connection by
    performing a search operation. Measures and returns the time taken to establish
    the connection.
    Args:
        uri: The MLflow tracking server URI to connect to. Defaults to URI constant.
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
            )
        # If it's some other kind of MlflowException, just re-raise
        # for debugging purposes.
        raise e

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

        if enable_mlflow:
            self.enabled = (self.config.logger.experiment_name is not None) and (
                MLFLOW_IMPORT_ERROR is None
            )

            # If mlflow is available, ensure there is an active run and log basic params/tags.
            if not self.enabled:
                return

            duration = mlflow_connect(self.config.uri)
            print(f"Connected in {duration.seconds} seconds")
            mlflow.set_experiment(self.config.logger.experiment_name)
            # If no active run, start one
            if mlflow.active_run() is None:
                print(f"Starting RUN: {str(self.run_name)}")
                mlflow.start_run(run_name=self.run_name)
            else:
                print(f"Run {str(self.run_name)} already active")

            if config is not None:
                print("Logging config...")
                config["num_classes"] = int(meta_model.num_classes)
                mlflow.log_dict(config, "config/config.json")

        if enable_wandb:
            self.logger = wandb.init(
                project=self.config.logger.experiment_name,
                name=self.run_name,
                config=config,
            )
            self.logger.config.update({"num_classes": int(meta_model.num_classes)})

    def log(self, log_dict, step):
        """
        Logs the provided dictionary of metrics along with the current step.
        Args:
            log_dict (Dict[str, Any]): A dictionary containing the log data to be recorded.
            step (int): The current step or iteration associated with the log entry.
        """
        if self.enable_mlflow and  not self.enabled:
            # Ensure there is an active mlflow run while logging metrics/artifacts
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=self.run_name)

            for k, v in (log_dict or {}).items():
                mlflow.log_metric(k, float(v), step=step)

        if self.enable_wandb:
            self.logger.log(log_dict, step = step)

    def save_model_checkpoint(
        self,
        meta_model_run: MetaModel,
        epoch: int,
        metrics_dict: Dict[str, float],
    ):
        """
        Saves a model checkpoint to a specified directory.
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

        checkpoint: Dict[str, Any] = {
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

        model_path = f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/model_{timestamp}"
        if epoch % self.log_checkpoint == 0:
            model_path = f"{self.checkpoint_dir}/model_checkpoints/{meta_model_run.run_name}/model_epoch{epoch}"

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(checkpoint, model_path)  # type: ignore
