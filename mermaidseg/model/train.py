"""
title: mermaidseg.model.train
abstract: Module that contains the function used to train a model end-to-end and log the metrics & checkpoints.
author: Viktor Domazetoski
date: 22-10-2025

Functions:
    train_model()
        Trains a model using the provided data loaders and logs the results.
"""

import time

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mermaidseg.logger import Logger
from mermaidseg.model.eval import Evaluator
from mermaidseg.model.meta import MetaModel


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
):
    """
    Trains a model using the provided data loaders and logs the results.
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
            Defaults to None. If provided, the model will be evaluated on the test set
            after each epoch.
        logger (Optional[Logger], optional): Logger object for logging metrics and saving
            model checkpoints. Defaults to None.
        start_epoch (int, optional): The starting epoch for training. Defaults to -1, which
            will be set to 0 if not specified.
        end_epoch (int, optional): The ending epoch for training. Defaults to -1, which
            will be set based on the meta-model's training configuration if not specified.
        metric_of_interest (str, optional): The primary metric used to determine the best
            model during validation. Defaults to "accuracy".
    Returns:
        None
    """

    best_results = {"epoch": -1, metric_of_interest: 0}

    if start_epoch == -1:
        start_epoch = 0
        end_epoch = meta_model.training_kwargs.epochs
    if end_epoch == -1:
        end_epoch = start_epoch + meta_model.training_kwargs.epochs
    metrics_epoch = {}
    for epoch in range(start_epoch, end_epoch):
        epoch_loss_dict = {}
        epoch_start_time = time.time()
        print(f"EPOCH: {epoch}")

        meta_model.model.train(True)
        train_loss, train_metric_results = meta_model.train_epoch(train_loader, evaluator)
        print(f"LOSS train {train_loss}")
        print(f"TRAIN METRICS: {train_metric_results}")
        epoch_loss_dict["train/loss"] = train_loss
        metrics_epoch[epoch] = {"train_metrics": train_metric_results}
        if logger is not None and len(train_metric_results) > 0:
            logger.log(
                {
                    f"train/{metric_name}": metric
                    for metric_name, metric in train_metric_results.items()
                },
                step=epoch,
            )
        epoch_loss_dict["train/time_taken"] = time.time() - epoch_start_time

        if hasattr(meta_model, "scheduler"):
            meta_model.scheduler.step()

        if val_loader is not None:
            meta_model.model.eval()
            val_loss, val_metric_results = meta_model.validation_epoch(val_loader, evaluator)
            print(f"LOSS valid {val_loss}")
            print(f"VALID METRICS: {val_metric_results}")

            epoch_loss_dict["validation/loss"] = val_loss
            metrics_epoch[epoch]["validation_metrics"] = val_metric_results
            if logger is not None and len(val_metric_results) > 0:
                logger.log(
                    {
                        f"validation/{metric_name}": metric
                        for metric_name, metric in val_metric_results.items()
                    },
                    step=epoch,
                )

                best_model_flag = (
                    best_results[metric_of_interest] < val_metric_results[metric_of_interest]
                )

                if best_model_flag:
                    best_results[metric_of_interest] = val_metric_results[metric_of_interest]
                    best_results["epoch"] = epoch

                    logger.save_model_checkpoint(meta_model, epoch, val_metric_results)

        logger.log(
            epoch_loss_dict,
            step=epoch,
        )

        metrics_epoch[epoch]["loss"] = epoch_loss_dict
        if epoch % logger.log_epochs > 0 and epoch < (end_epoch - 1):
            continue

        if test_loader is not None:
            _ = evaluate_and_log(evaluator, test_loader, meta_model, logger, epoch, "test")
    return metrics_epoch


def evaluate_and_log(
    evaluator: Evaluator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
    meta_model: MetaModel,
    logger: Logger,
    epoch: int,
    split: str = "train",
) -> dict[str, float | NDArray[np.float64]]:
    """
    Evaluates a model using the provided evaluator, logs the metrics, and logs image predictions.
    Args:
        evaluator (Evaluator): The evaluator object used to compute metrics and evaluate the model.
        loader (DataLoader): A data loader providing the dataset for evaluation. The dataset can be
            either a tuple of tensors or a dictionary of tensors.
        meta_model (MetaModel): The model to be evaluated.
        logger (Logger): The logger object used to log metrics and image predictions.
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
    if logger is not None:
        logger.log(
            {f"{split}/{metric_name}": metric for metric_name, metric in metric_results.items()},
            step=epoch,
        )
    print(f"{split} metrics")
    print(metric_results)

    # logger.log_image_predictions(
    #     *evaluator.evaluate_image(
    #         loader, meta_model, epoch=epoch, log_epochs=logger.log_epochs
    #     ),
    #     epoch,
    #     split=split,
    # )

    return metric_results
