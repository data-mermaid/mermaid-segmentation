import csv
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.metric_policy import (
    canonical_metric_name,
    extract_metric_value,
    metric_direction,
)


def build_epoch_metrics(
    prefix: str,
    total_loss: float,
    metric_results: dict[str, float | NDArray[np.float64]],
) -> dict[str, float]:
    """Build scalar train/validation metrics for logging and CSV export."""
    metrics: dict[str, float] = {f"{prefix}/loss/total": float(total_loss)}
    for key, value in metric_results.items():
        if not isinstance(value, (int, float, np.floating)):
            continue
        if key.startswith("loss/") or key.startswith("accuracy/"):
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


class LocalMetricsWriter:
    """Append per-epoch training metrics to a local CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._fieldnames: list[str] | None = None

    def write(self, epoch: int, metrics: dict[str, float]) -> None:
        row: dict[str, float | int | str] = {"epoch": epoch, **metrics}
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            self._fieldnames = ["epoch"] + sorted(metrics)
            with self.csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ["epoch"])
            rows = [dict(record) for record in reader]

        new_keys = sorted(set(row) - set(fieldnames))
        if new_keys:
            fieldnames.extend(new_keys)
            for record in rows:
                for key in new_keys:
                    record.setdefault(key, "")
        self._fieldnames = fieldnames
        rows.append({key: row.get(key, "") for key in fieldnames})

        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def train_model(
    meta_model: MetaModel,
    evaluator: Evaluator,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]
    | None = None,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]
    | None = None,
    logger: Logger | None = None,
    start_epoch: int = -1,
    end_epoch: int = -1,
    metric_of_interest: str = "accuracy",
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
):
    """Train a model, logging losses and metrics per epoch.

    Args:
        meta_model (MetaModel): The meta-model to be trained, which includes the model,
            optimizer, scheduler, and training configurations.
        evaluator (Evaluator): An evaluator object used to compute metrics during
            evaluation phases.
        train_loader (DataLoader): DataLoader for the training dataset. It should yield
            either tuples of tensors or dictionaries of tensors.
        val_loader (Optional[DataLoader], optional): DataLoader for the validation dataset.
            Defaults to None. If provided, the model will be evaluated on the validation
            set after each epoch.
        test_loader (Optional[DataLoader], optional): DataLoader for the test dataset.
            Defaults to None. If provided, the model is evaluated periodically according
            to ``logger.log_epochs`` (or every epoch when logger is None), plus the final epoch.
        logger (Optional[Logger], optional): Logger object for logging metrics and saving
            model checkpoints. Defaults to None.
        start_epoch (int, optional): The starting epoch for training. Defaults to -1, which
            will be set to 0 if not specified.
        end_epoch (int, optional): The ending epoch for training. Defaults to -1, which
            will be set based on the meta-model's training configuration if not specified.
        metric_of_interest (str, optional): Metric used for checkpointing and early
            stopping. Must resolve to ``loss`` or ``accuracy`` (classification accuracy).
            Defaults to "accuracy".
        early_stopping (bool, optional): Enables early stopping on validation
            `metric_of_interest`. Defaults to False.
        early_stopping_patience (int, optional): Number of consecutive epochs with no
            improvement allowed before stopping early. Defaults to 10.
        early_stopping_min_delta (float, optional): Minimum metric improvement required
            to reset patience. Defaults to 0.0.
    Returns:
        dict[int, dict]: Per-epoch metrics keyed by epoch number, containing
            ``train_metrics``, ``validation_metrics`` (if ``val_loader`` is provided), and
            ``loss``. The ``loss`` sub-dict includes training loss plus timing metrics
            kept locally for debugging.
    """

    metric_name = canonical_metric_name(metric_of_interest)
    direction = metric_direction(metric_name)
    best_metric_value = float("inf") if direction == "min" else float("-inf")
    best_results = {"epoch": -1, metric_name: best_metric_value}
    epochs_without_improvement = 0

    if early_stopping and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be > 0 when early_stopping is enabled.")
    if early_stopping and val_loader is None:
        raise ValueError("early_stopping requires val_loader.")

    if start_epoch == -1:
        start_epoch = 0
    if end_epoch == -1:
        end_epoch = start_epoch + meta_model.training_kwargs.epochs
    metrics_epoch = {}
    training_start = time.perf_counter()

    local_metrics_writer: LocalMetricsWriter | None = None
    if logger is not None:
        csv_path = (
            Path(logger.checkpoint_dir) / "model_checkpoints" / meta_model.run_name / "metrics.csv"
        )
        local_metrics_writer = LocalMetricsWriter(csv_path)

    for epoch in range(start_epoch, end_epoch):
        should_stop_early = False
        epoch_loss_dict: dict[str, float] = {}
        epoch_training_metrics: dict[str, float] = {}
        epoch_start_time = time.time()
        logging.info("EPOCH: %d", epoch)

        meta_model.model.train(True)
        train_loss, train_metric_results, train_timing = meta_model.train_epoch(
            train_loader, evaluator
        )
        logging.info("LOSS train %s", train_loss)
        logging.info("TRAIN METRICS: %s", train_metric_results)
        epoch_loss_dict["train/loss"] = train_loss
        epoch_loss_dict["train/data_loading_sec"] = train_timing["data_loading_sec"]
        epoch_loss_dict["train/forward_sec"] = train_timing["forward_sec"]
        epoch_loss_dict["train/backward_sec"] = train_timing["backward_sec"]
        metrics_epoch[epoch] = {"train_metrics": train_metric_results}
        epoch_training_metrics.update(
            build_epoch_metrics("train", train_loss, train_metric_results)
        )

        scheduler = getattr(meta_model, "scheduler", None)
        metric_value: float | None = None

        if val_loader is not None:
            meta_model.model.eval()
            val_start = time.time()
            val_loss, val_metric_results = meta_model.validation_epoch(val_loader, evaluator)
            epoch_loss_dict["validation/time_taken"] = time.time() - val_start
            logging.info("LOSS valid %s", val_loss)
            logging.info("VALID METRICS: %s", val_metric_results)

            epoch_loss_dict["validation/loss"] = val_loss
            metrics_epoch[epoch]["validation_metrics"] = val_metric_results
            epoch_training_metrics.update(
                build_epoch_metrics("validation", val_loss, val_metric_results)
            )

            metric_value = extract_metric_value(metric_of_interest, val_loss, val_metric_results)
            if direction == "min":
                improved = metric_value < (best_metric_value - early_stopping_min_delta)
            else:
                improved = metric_value > (best_metric_value + early_stopping_min_delta)

            if improved:
                best_metric_value = metric_value
                best_results[metric_name] = metric_value
                best_results["epoch"] = epoch
                epochs_without_improvement = 0
                if logger is not None:
                    logger.save_model_checkpoint(meta_model, epoch, val_metric_results)
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement >= early_stopping_patience:
                logging.info(
                    "Early stopping triggered: no '%s' improvement for %d epoch(s).",
                    metric_name,
                    early_stopping_patience,
                )
                should_stop_early = True

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metric_value is not None:
                    scheduler.step(metric_value)
                else:
                    warnings.warn(
                        "Skipping ReduceLROnPlateau step because no validation metric is available.",
                        stacklevel=2,
                    )
            else:
                scheduler.step()

            if logger is not None:
                current_lr = meta_model.optimizer.param_groups[0]["lr"]
                logger.log({"train/lr": current_lr}, step=epoch)

        epoch_wall = time.time() - epoch_start_time
        epoch_loss_dict["train/time_taken"] = epoch_wall
        epoch_loss_dict["train/samples_per_sec"] = train_timing["num_samples"] / epoch_wall
        epoch_loss_dict["train/data_loading_pct"] = (
            train_timing["data_loading_sec"] / epoch_wall * 100
        )

        if torch.cuda.is_available():
            epoch_loss_dict["train/gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()

        if epoch == end_epoch - 1:
            epoch_loss_dict["train/total_training_sec"] = time.perf_counter() - training_start

        if logger is not None:
            logger.log_training_metrics(epoch_training_metrics, step=epoch)

        if local_metrics_writer is not None:
            local_metrics_writer.write(epoch, epoch_training_metrics)

        metrics_epoch[epoch]["loss"] = epoch_loss_dict
        log_every = max(logger.log_epochs, 1) if logger is not None else 1

        if should_stop_early:
            if test_loader is not None:
                _ = evaluate_and_log(evaluator, test_loader, meta_model, epoch, "test")
            break

        if epoch % log_every > 0 and epoch < (end_epoch - 1):
            continue

        if test_loader is not None:
            test_start = time.time()
            _ = evaluate_and_log(evaluator, test_loader, meta_model, epoch, "test")
            epoch_loss_dict["test/time_taken"] = time.time() - test_start
    return metrics_epoch


def evaluate_and_log(
    evaluator: Evaluator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    meta_model: MetaModel,
    epoch: int,
    split: str = "train",
) -> dict[str, float | NDArray[np.float64]]:
    """Evaluate a split and log metrics locally (stderr only, not MLflow).

    Args:
        evaluator (Evaluator): The evaluator object used to compute metrics and evaluate the model.
        loader (DataLoader): A data loader providing the dataset for evaluation. The dataset can be
            either a tuple of tensors or a dictionary of tensors.
        meta_model (MetaModel): The model to be evaluated.
        epoch (int): The current epoch number, used for logging purposes.
        split (str, optional): The dataset split being evaluated (e.g., "train", "validation", "test").
            Defaults to "train".
    Returns:
        Dict[str, Union[float, NDArray[np.float64]]]: A dictionary containing the evaluation metrics.
    """
    metric_results = evaluator.evaluate_model(
        loader,
        meta_model,
    )
    logging.info("%s metrics (epoch %d): %s", split, epoch, metric_results)

    return metric_results
