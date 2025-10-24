"""
title: mermaidseg.logger
abstract: Module that contains the mlflow logging functionality.
author: Viktor Domazetoski
date: 24-10-2025

Classes:
    Logger - A class for logging metrics and configurations to an MLflow tracking server.
Functions:
    mlflow_connect() - Connect to the MLflow tracking server and return the connection time.
"""

try:
    import mlflow
    from mlflow.models import infer_signature

    MLFLOW_IMPORT_ERROR = None
except ImportError as err:
    MLFLOW_IMPORT_ERROR = err

from datetime import datetime, timedelta

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
        self, config, meta_model, log_epochs=5, log_checkpoint=50, checkpoint_dir="."
    ):
        """
        Initializes the logger for tracking experiments and benchmarks.
        Args:
            config (dict, optional): Configuration dictionary for the experiment
            meta_model: Meta model object containing model metadata.
            log_epochs (int, optional): Frequency of logging epochs. Defaults to 5.
            log_checkpoint (int, optional): Frequency of logging checkpoints. Defaults to 50.
            checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to ".".
        Attributes:
            config (dict): Stores the configuration dictionary for the experiment.
            log_epochs (int): Frequency of logging epochs.
            log_checkpoint (int): Frequency of logging checkpoints.
            checkpoint_dir (str): Directory to save checkpoints.
        """

        self.config = config
        self.log_epochs = log_epochs
        self.log_checkpoint = log_checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.run_name = meta_model.run_name

        self.enabled = (self.config.experiment_name is not None) and (
            MLFLOW_IMPORT_ERROR is None
        )

        # If mlflow is available, ensure there is an active run and log basic params/tags.
        if not self.enabled:
            return

        duration = mlflow_connect(self.config.uri)
        print(f"Connected in {duration.seconds} seconds")
        mlflow.set_experiment(self.config.experiment_name)
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

    def log(self, log_dict, step):
        """
        Logs the provided dictionary of metrics along with the current step.
        Args:
            log_dict (Dict[str, Any]): A dictionary containing the log data to be recorded.
            step (int): The current step or iteration associated with the log entry.
        """
        if not self.enabled:
            return
        # Ensure there is an active mlflow run while logging metrics/artifacts
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=self.run_name)

        for k, v in (log_dict or {}).items():
            mlflow.log_metric(k, float(v), step=step)
