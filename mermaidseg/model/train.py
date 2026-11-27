"""One training epoch that logs losses and metrics."""

import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel


def run_epoch(
    meta_model: MetaModel,
    evaluator: Evaluator,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    val_loader: DataLoader[tuple[torch.Tensor, ...]] | None,
    logger: Logger | None,
    *,
    epoch: int,
    train_log_interval: int,
    on_optimizer_step: Callable[[MetaModel], None] | None,
) -> dict[str, Any]:
    """Train one epoch, then validate, logging losses and metrics.

    Args:
        meta_model: The meta-model to be trained, which includes the model,
            optimizer, scheduler, and training configurations.
        evaluator: An evaluator object used to compute metrics during
            evaluation phases.
        train_loader: DataLoader for the training dataset. It should yield
            either tuples of tensors or dictionaries of tensors.
        val_loader: DataLoader for the validation dataset. When given, the model is
            evaluated on it after the epoch.
        logger: Logger for metrics, or None to log nothing.
        epoch: Index of the epoch.
        train_log_interval: Log per-batch train losses to the logger every
            ``train_log_interval`` iterations (1 = every batch).
        on_optimizer_step: Called from ``train_epoch`` after each successful optimizer update.

    Returns:
        This epoch's metrics: ``train_metrics``, ``validation_metrics`` (if ``val_loader`` is
        given), and ``loss``. The ``loss`` sub-dict includes training loss plus timing metrics
        kept locally for debugging.
    """
    epoch_start_time = time.time()
    epoch_start_perf = time.perf_counter()
    epoch_loss_dict: dict[str, float] = {}
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
    epoch_metrics: dict[str, Any] = {"train_metrics": train_metric_results}
    _log_metric_dict(logger, "train", train_metric_results, step)

    if val_loader is not None:
        val_start = time.time()
        val_loss, val_metric_results = meta_model.validation_epoch(val_loader, evaluator)
        epoch_loss_dict["validation/time_taken"] = time.time() - val_start
        logging.info("LOSS valid %s", val_loss)
        logging.info("VALID METRICS: %s", val_metric_results)

        epoch_loss_dict["validation/loss"] = val_loss
        epoch_metrics["validation_metrics"] = val_metric_results
        _log_metric_dict(logger, "validation", val_metric_results, step)

    if logger is not None:
        current_lr = meta_model.optimizer.param_groups[0]["lr"]
        logger.log({"train/lr": current_lr}, step=step)

    epoch_wall = time.time() - epoch_start_time
    epoch_loss_dict["train/time_taken"] = epoch_wall
    epoch_loss_dict["train/samples_per_sec"] = train_timing["num_samples"] / epoch_wall
    epoch_loss_dict["train/data_loading_pct"] = train_timing["data_loading_sec"] / epoch_wall * 100

    if meta_model.device_type == "cuda":
        epoch_loss_dict["train/gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

    epoch_loss_dict["train/total_training_sec"] = time.perf_counter() - epoch_start_perf

    # Surface the epoch index on the iteration axis so it is available in the experiment tracker.
    epoch_loss_dict["epoch"] = float(epoch)
    if logger is not None:
        logger.log(epoch_loss_dict, step=step)

    epoch_metrics["loss"] = epoch_loss_dict
    return epoch_metrics


def _log_metric_dict(
    logger: Logger | None,
    prefix: str,
    metric_results: dict[str, float],
    step: int,
) -> None:
    if logger is None or not metric_results:
        return
    logger.log({f"{prefix}/{name}": value for name, value in metric_results.items()}, step=step)
