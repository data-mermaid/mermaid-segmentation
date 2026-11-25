"""Training loop that logs losses and metrics per epoch and checkpoints the best model."""

import logging
import time
from collections.abc import Callable
from typing import Any

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


def train_model(
    meta_model: MetaModel,
    evaluator: Evaluator,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]
    | None = None,
    logger: Logger | None = None,
    start_epoch: int = -1,
    end_epoch: int = -1,
    metric_of_interest: str = "accuracy",
    train_log_interval: int = 1,
    on_optimizer_step: Callable[[MetaModel], None] | None = None,
    save_best_checkpoint: bool = True,
) -> dict[int, dict[str, Any]]:
    """Train a model, logging losses and metrics per epoch.

    Args:
        meta_model: The meta-model to be trained, which includes the model,
            optimizer, scheduler, and training configurations.
        evaluator: An evaluator object used to compute metrics during
            evaluation phases.
        train_loader: DataLoader for the training dataset. It should yield
            either tuples of tensors or dictionaries of tensors.
        val_loader: DataLoader for the validation dataset. If provided, the model is
            evaluated on the validation set after each epoch.
        logger: Logger object for logging metrics and saving model checkpoints.
        start_epoch: The starting epoch for training. ``-1`` is treated as 0.
        end_epoch: The ending epoch for training. ``-1`` is set from the meta-model's
            training configuration.
        metric_of_interest: Metric used for checkpointing. Must resolve to ``loss`` or
            ``accuracy`` (classification accuracy).
        train_log_interval: Log per-batch train losses to the logger every
            ``train_log_interval`` iterations (1 = every batch).
        on_optimizer_step: Called from ``train_epoch`` after each successful optimizer update.
        save_best_checkpoint: When True, write ``model_epoch{N}`` whenever the validation
            metric improves. Step-checkpointed runs pass False so only ``model_step*`` is saved.

    Returns:
        Per-epoch metrics keyed by epoch number, containing
            ``train_metrics``, ``validation_metrics`` (if ``val_loader`` is provided), and
            ``loss``. The ``loss`` sub-dict includes training loss plus timing metrics
            kept locally for debugging.
    """
    metric_name = canonical_metric_name(metric_of_interest)
    direction = metric_direction(metric_name)
    best_metric_value = float("inf") if direction == "min" else float("-inf")
    best_results = {"epoch": -1, metric_name: best_metric_value}

    if start_epoch == -1:
        start_epoch = 0
    if end_epoch == -1:
        end_epoch = start_epoch + meta_model.training_kwargs.epochs
    metrics_epoch: dict[int, dict[str, Any]] = {}
    training_start = time.perf_counter()

    for epoch in range(start_epoch, end_epoch):
        epoch_loss_dict: dict[str, float] = {}
        epoch_start_time = time.time()
        logging.info("EPOCH: %d", epoch)

        meta_model.model.train(True)
        train_loss, train_metric_results, train_timing = meta_model.train_epoch(
            train_loader,
            evaluator,
            run_logger=logger,
            log_interval=train_log_interval,
            epoch=epoch,
            on_optimizer_step=on_optimizer_step,
        )
        # Monotonic global-step (iteration) axis for the experiment tracker.
        step = meta_model.global_step
        logging.info("LOSS train %s", train_loss)
        logging.info("TRAIN METRICS: %s", train_metric_results)
        epoch_loss_dict["train/loss"] = train_loss
        epoch_loss_dict["train/data_loading_sec"] = train_timing["data_loading_sec"]
        epoch_loss_dict["train/forward_sec"] = train_timing["forward_sec"]
        epoch_loss_dict["train/backward_sec"] = train_timing["backward_sec"]
        metrics_epoch[epoch] = {"train_metrics": train_metric_results}
        _log_metric_dict(logger, "train", train_metric_results, step)

        metric_value: float | None = None

        if val_loader is not None:
            val_start = time.time()
            val_loss, val_metric_results = meta_model.validation_epoch(val_loader, evaluator)
            epoch_loss_dict["validation/time_taken"] = time.time() - val_start
            logging.info("LOSS valid %s", val_loss)
            logging.info("VALID METRICS: %s", val_metric_results)

            epoch_loss_dict["validation/loss"] = val_loss
            metrics_epoch[epoch]["validation_metrics"] = val_metric_results
            _log_metric_dict(logger, "validation", val_metric_results, step)

            metric_value = extract_metric_value(metric_of_interest, val_loss, val_metric_results)
            if direction == "min":
                improved = metric_value < (best_metric_value)
            else:
                improved = metric_value > best_metric_value

            if improved:
                best_metric_value = metric_value
                best_results[metric_name] = metric_value
                best_results["epoch"] = epoch
                if logger is not None and save_best_checkpoint:
                    logger.save_model_checkpoint(meta_model, epoch, val_metric_results)

        if logger is not None:
            current_lr = meta_model.optimizer.param_groups[0]["lr"]
            logger.log({"train/lr": current_lr}, step=step)

        epoch_wall = time.time() - epoch_start_time
        epoch_loss_dict["train/time_taken"] = epoch_wall
        epoch_loss_dict["train/samples_per_sec"] = train_timing["num_samples"] / epoch_wall
        epoch_loss_dict["train/data_loading_pct"] = (
            train_timing["data_loading_sec"] / epoch_wall * 100
        )

        if meta_model.device_type == "cuda":
            epoch_loss_dict["train/gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()

        if epoch == end_epoch - 1:
            epoch_loss_dict["train/total_training_sec"] = time.perf_counter() - training_start

        # Surface the epoch index on the iteration axis so it is available in the experiment tracker.
        epoch_loss_dict["epoch"] = float(epoch)
        if logger is not None:
            logger.log(epoch_loss_dict, step=step)

        metrics_epoch[epoch]["loss"] = epoch_loss_dict
    return metrics_epoch


def _log_metric_dict(
    logger: Logger | None,
    prefix: str,
    metric_results: dict[str, float | NDArray[np.float64]],
    step: int,
) -> None:
    if logger is None or not metric_results:
        return
    logger.log({f"{prefix}/{name}": value for name, value in metric_results.items()}, step=step)
