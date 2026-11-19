"""Training loop that logs losses and metrics per epoch and checkpoints the best model."""

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mermaidseg.datasets.local_cache import LocalS3Cache
from mermaidseg.datasets.utils import emit_cache_stats, emit_load_stage_stats
from mermaidseg.logger import Logger
from mermaidseg.model.distributed import all_reduce_sum
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel
from mermaidseg.model.metric_policy import (
    canonical_metric_name,
    extract_metric_value,
    metric_direction,
)


def _enforce_load_failure_rate(
    failures_before: int | None,
    num_loaded_samples: int,
    max_rate: float | None,
    epoch: int,
    split: str = "train",
) -> None:
    """Raise on every rank if this epoch's dataset load-failure rate exceeds ``max_rate``.

    A high rate signals a systemic data problem (bad credentials, missing files, truncated image
    lists) rather than a few corrupt samples, so we fail fast instead of silently training on a
    shrunken/biased dataset. The rate is failed load attempts over all load attempts this epoch,
    summed over ranks: ``failures / (num_loaded_samples + failures)``. Every rank sees the same
    rate and raises together instead of leaving the others in the next collective, so this is a
    collective call that every rank must make.

    The rate is approximate at the epoch boundary: loader workers prefetch, so a few batches
    loaded during validation or before the next epoch's first read can land in the neighbouring
    epoch's window (both loaders share one failure counter).

    Args:
        failures_before: ``LocalS3Cache`` failure count read before the epoch, or None if untracked.
        num_loaded_samples: Samples the loaders delivered this epoch, summed over ranks.
        max_rate: Largest tolerated failure rate, or None to disable the guard.
        epoch: Epoch number, for the error message.
        split: Split name, for the error message.

    Raises:
        RuntimeError: If the epoch's failure rate is above ``max_rate``.
    """
    if max_rate is None:
        return
    failures_after = LocalS3Cache.get().load_failure_count()
    delta = (
        0 if failures_before is None or failures_after is None else failures_after - failures_before
    )
    epoch_failures = int(all_reduce_sum(torch.tensor([delta], dtype=torch.int64)).item())
    attempts = num_loaded_samples + epoch_failures
    if attempts == 0:
        return
    rate = epoch_failures / attempts
    if rate > max_rate:
        raise RuntimeError(
            f"Epoch {epoch}: {split} load-failure rate {rate:.1%} ({epoch_failures}/{attempts}) "
            f"exceeds max_load_failure_rate={max_rate:.1%}. This usually indicates a systemic "
            f"data problem (credentials, missing files, truncated image lists) rather than a few "
            f"corrupt samples. Inspect the dataset warnings the loader workers emitted; pass "
            f"max_load_failure_rate=None to disable this guard."
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
    max_load_failure_rate: float | None = 0.05,
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
        max_load_failure_rate: If set, raise once the per-epoch
            dataset load-failure rate exceeds this fraction (e.g. 0.05 = 5%), to fail fast on
            systemic data problems instead of silently training on a shrunken dataset. Set to
            ``None`` to disable. Only enforced when the shared load stats are attached (see
            ``setup_local_cache``); otherwise there is no failure count to read.
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
        failures_before = LocalS3Cache.get().load_failure_count()
        train_loss, train_metric_results, train_timing = meta_model.train_epoch(
            train_loader,
            evaluator,
            run_logger=logger,
            log_interval=train_log_interval,
            epoch=epoch,
            on_optimizer_step=on_optimizer_step,
        )
        _enforce_load_failure_rate(
            failures_before, int(train_timing["num_loaded_samples"]), max_load_failure_rate, epoch
        )
        # Monotonic global-step (iteration) axis for the experiment tracker.
        step = meta_model.global_step
        logging.info("LOSS train %s", train_loss)
        logging.info("TRAIN METRICS: %s", train_metric_results)
        train_cache_stats = LocalS3Cache.get().snapshot_stats()
        emit_cache_stats(
            f"EPOCH {epoch} train cache: {train_cache_stats.s3_fetches} fetched from S3, "
            f"{train_cache_stats.local_hits} served from local"
        )
        emit_load_stage_stats(f"EPOCH {epoch} train", train_cache_stats)
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
            val_cache_stats = LocalS3Cache.get().snapshot_stats()
            emit_cache_stats(
                f"EPOCH {epoch} val cache: {val_cache_stats.s3_fetches} fetched from S3, "
                f"{val_cache_stats.local_hits} served from local"
            )
            emit_load_stage_stats(f"EPOCH {epoch} val", val_cache_stats)

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
