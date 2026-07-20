import logging
import time
import warnings

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


def _dataset_load_failures(dataset: object) -> int | None:
    """Cumulative load-failure count for ``dataset``, or ``None`` if untracked.

    Unwraps ``torch.utils.data.ConcatDataset`` (and anything exposing ``.datasets``) by
    summing the tracked children — the real training path wraps per-source
    ``BaseCoralDataset`` instances in a ``ConcatDataset``, which itself has no
    ``num_load_failures``.
    """
    if dataset is None:
        return None
    if hasattr(dataset, "num_load_failures"):
        try:
            return int(dataset.num_load_failures())
        except Exception:
            return None
    children = getattr(dataset, "datasets", None)
    if children:
        counts = [_dataset_load_failures(child) for child in children]
        tracked = [c for c in counts if c is not None]
        return sum(tracked) if tracked else None
    return None


def _loader_load_failure_count(loader: object) -> int | None:
    """Return the cumulative load-failure count of a loader's dataset, or None if
    untracked."""
    return _dataset_load_failures(getattr(loader, "dataset", None))


def _enforce_load_failure_rate(
    loader: object,
    failures_before: int | None,
    max_rate: float | None,
    epoch: int,
    samples_processed: int,
    split: str = "train",
) -> None:
    """Raise if this epoch's load-failure rate exceeds ``max_rate``.

    Rate = ``failures_this_epoch / attempts_this_epoch`` where ``attempts`` is
    ``samples_processed`` (items that collated through successfully this epoch) plus the
    failures this epoch. The denominator is per-epoch attempts, NOT ``len(dataset)``:
    ``iterations_per_*_epoch`` samples only a fraction of a large dataset each epoch, so
    dividing by the full dataset size would make the rate ~100x too small to ever trip.

    A high rate signals a systemic data problem (bad credentials, missing files, truncated
    image lists) rather than a few corrupt samples, so we fail fast instead of silently
    training on a shrunken/biased dataset. No-op when disabled or when failures are untracked.
    """
    if max_rate is None or failures_before is None:
        return
    after = _loader_load_failure_count(loader)
    if after is None:
        return
    epoch_failures = max(after - failures_before, 0)
    attempts = samples_processed + epoch_failures
    if attempts <= 0:
        return
    rate = epoch_failures / attempts
    if rate > max_rate:
        raise RuntimeError(
            f"Epoch {epoch}: {split} load-failure rate {rate:.1%} ({epoch_failures}/{attempts}) "
            f"exceeds max_load_failure_rate={max_rate:.1%}. This usually indicates a systemic "
            f"data problem (credentials, missing files, truncated image lists) rather than a few "
            f"corrupt samples. Inspect the dataset load-failure report; pass "
            f"max_load_failure_rate=None to disable this guard."
        )


def train_model(
    meta_model: MetaModel,
    evaluator: Evaluator,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]
    | None = None,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]
    | None = None,
    dataset_val_loaders: (
        dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]]] | None
    ) = None,
    logger: Logger | None = None,
    start_epoch: int = -1,
    end_epoch: int = -1,
    metric_of_interest: str = "miou",
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    max_load_failure_rate: float | None = 0.05,
):
    """Train a model, logging losses and metrics per epoch.

    Args:
        meta_model (MetaModel): The meta-model to be trained, which includes the model,
            optimizer, scheduler, and training configurations.
        evaluator (Evaluator): An evaluator object used to compute metrics during
            evaluation phases.
        train_loader (DataLoader): DataLoader for the training dataset. It should yield
            either tuples of tensors or dictionaries of tensors.
        val_loader (Optional[DataLoader], optional): Combined validation DataLoader used for
            checkpoint selection / early stopping. Defaults to None.
        test_loader (Optional[DataLoader], optional): DataLoader for the test dataset.
            Defaults to None. If provided, the model is evaluated periodically according
            to ``logger.log_epochs`` (or every epoch when logger is None), plus the final epoch.
        dataset_val_loaders (dict[str, DataLoader] | None): Optional per-dataset validation
            loaders (e.g. ``{\"mermaid\": ..., \"coralnet\": ...}``). Logged as
            ``validation/{name}/*`` without affecting checkpoint selection.
        logger (Optional[Logger], optional): Logger object for logging metrics and saving
            model checkpoints. Defaults to None. When ``logger.log_checkpoint`` is set, a
            periodic (non-improvement) checkpoint is also saved every ``log_checkpoint``
            epochs, so a resumable snapshot exists even if the validation metric never
            improves. Periodic checkpoints are skipped on epochs already covered by an
            improvement-triggered save.
        start_epoch (int, optional): The starting epoch for training. Defaults to -1, which
            will be set to 0 if not specified.
        end_epoch (int, optional): The ending epoch for training. Defaults to -1, which
            will be set based on the meta-model's training configuration if not specified.
        metric_of_interest (str, optional): Metric used for checkpointing and early
            stopping. One of ``loss``, ``accuracy``, ``miou``, ``f1-score``. Defaults to
            "miou" — mean IoU is a more reliable segmentation metric than pixel accuracy,
            which is dominated by majority classes (e.g. background) in imbalanced data.
        early_stopping (bool, optional): Enables early stopping on validation
            `metric_of_interest`. Defaults to False.
        early_stopping_patience (int, optional): Number of consecutive epochs with no
            improvement allowed before stopping early. Defaults to 10.
        early_stopping_min_delta (float, optional): Minimum metric improvement required
            to reset patience. Defaults to 0.0.
        max_load_failure_rate (float | None, optional): If set, raise once the per-epoch
            dataset load-failure rate exceeds this fraction (e.g. 0.05 = 5%), to fail fast on
            systemic data problems instead of silently training on a shrunken dataset. Set to
            ``None`` to disable. Only enforced when the loader's dataset tracks load failures.
            Defaults to 0.05.
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
    checkpoint_interval = getattr(logger, "log_checkpoint", None) if logger is not None else None

    for epoch in range(start_epoch, end_epoch):
        should_stop_early = False
        checkpoint_saved_this_epoch = False
        epoch_loss_dict: dict[str, float] = {}
        epoch_start_time = time.time()
        logging.info("EPOCH: %d", epoch)

        meta_model.model.train(True)
        failures_before = _loader_load_failure_count(train_loader)
        train_loss, train_metric_results, train_timing = meta_model.train_epoch(
            train_loader, evaluator
        )
        _enforce_load_failure_rate(
            train_loader,
            failures_before,
            max_load_failure_rate,
            epoch,
            train_timing["num_samples"],
        )
        logging.info("LOSS train %s", train_loss)
        logging.info("TRAIN METRICS: %s", train_metric_results)
        epoch_loss_dict["train/loss"] = train_loss
        epoch_loss_dict["train/data_loading_sec"] = train_timing["data_loading_sec"]
        epoch_loss_dict["train/forward_sec"] = train_timing["forward_sec"]
        epoch_loss_dict["train/backward_sec"] = train_timing["backward_sec"]
        metrics_epoch[epoch] = {"train_metrics": train_metric_results}
        _log_metric_dict(logger, "train", train_metric_results, epoch)

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
            _log_metric_dict(logger, "validation", val_metric_results, epoch)

            if dataset_val_loaders:
                per_dataset_metrics: dict[str, object] = {}
                for dataset_name, ds_loader in sorted(dataset_val_loaders.items()):
                    ds_loss, ds_metrics = meta_model.validation_epoch(ds_loader, evaluator)
                    prefix = f"validation/{dataset_name}"
                    logging.info("VALID [%s] loss=%s metrics=%s", dataset_name, ds_loss, ds_metrics)
                    epoch_loss_dict[f"{prefix}/loss"] = ds_loss
                    _log_metric_dict(logger, prefix, ds_metrics, epoch)
                    per_dataset_metrics[dataset_name] = {"loss": ds_loss, "metrics": ds_metrics}
                metrics_epoch[epoch]["validation_per_dataset"] = per_dataset_metrics

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
                    checkpoint_saved_this_epoch = True
            else:
                epochs_without_improvement += 1

            if early_stopping and epochs_without_improvement >= early_stopping_patience:
                logging.info(
                    "Early stopping triggered: no '%s' improvement for %d epoch(s).",
                    metric_name,
                    early_stopping_patience,
                )
                should_stop_early = True

        if (
            logger is not None
            and checkpoint_interval
            and checkpoint_interval > 0
            and not checkpoint_saved_this_epoch
            and epoch % checkpoint_interval == 0
        ):
            periodic_metrics = (
                val_metric_results if val_loader is not None else train_metric_results
            )
            logger.save_model_checkpoint(meta_model, epoch, periodic_metrics, is_best=False)

        warmup_complete = (
            getattr(meta_model, "warmup_iters", 0) == 0
            or meta_model._warmup_iters_completed >= meta_model.warmup_iters
        )
        if scheduler is not None and warmup_complete:
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

        if scheduler is not None and logger is not None:
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
            logger.log(epoch_loss_dict, step=epoch)

        metrics_epoch[epoch]["loss"] = epoch_loss_dict
        log_every = max(logger.log_epochs, 1) if logger is not None else 1

        if should_stop_early:
            if test_loader is not None:
                _ = evaluate_and_log(evaluator, test_loader, meta_model, logger, epoch, "test")
            break

        if epoch % log_every > 0 and epoch < (end_epoch - 1):
            continue

        if test_loader is not None:
            test_start = time.time()
            _ = evaluate_and_log(evaluator, test_loader, meta_model, logger, epoch, "test")
            test_time = time.time() - test_start
            epoch_loss_dict["test/time_taken"] = test_time
            # epoch_loss_dict was already logged above (before test eval ran), so log this
            # timing metric directly to keep it in MLflow (matches main's behavior).
            if logger is not None:
                logger.log({"test/time_taken": test_time}, step=epoch)
    return metrics_epoch


def _log_metric_dict(
    logger: Logger | None,
    prefix: str,
    metric_results: dict[str, float | NDArray[np.float64]],
    epoch: int,
) -> None:
    if logger is None or not metric_results:
        return
    logger.log({f"{prefix}/{name}": value for name, value in metric_results.items()}, step=epoch)


def evaluate_and_log(
    evaluator: Evaluator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    meta_model: MetaModel,
    logger: Logger | None,
    epoch: int,
    split: str = "train",
) -> dict[str, float | NDArray[np.float64]]:
    """Evaluate a split and optionally log its metrics.

    Args:
        evaluator (Evaluator): The evaluator object used to compute metrics and evaluate the model.
        loader (DataLoader): A data loader providing the dataset for evaluation. The dataset can be
            either a tuple of tensors or a dictionary of tensors.
        meta_model (MetaModel): The model to be evaluated.
        logger (Logger | None): Optional logger used to log metrics and image predictions.
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
    _log_metric_dict(logger, split, metric_results, epoch)
    logging.info("%s metrics (epoch %d): %s", split, epoch, metric_results)

    return metric_results
