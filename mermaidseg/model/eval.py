"""
title: mermaidseg.model.eval
abstract: Module that contains the Evaluator classes which are used to calculate model performance metrics.
author: Viktor Domazetoski
date: 22-08-2025

Classes:
    Evaluator
        __init__
        evaluate_model()
            Evaluates the performance of a given meta-model on a dataset provided by the dataloader.
        evaluate_image()
            Evaluates the given model on one image of the dataloader.

    EvaluatorSemanticSegmentation(Evaluator)
        __init__
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from mermaidseg.model.meta import MetaModel
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex
from torchmetrics.metric import Metric


class Evaluator:
    """
    Evaluator class for evaluating machine learning models.
    This class provides methods to evaluate the performance of a meta-model on a dataset
    or a single image using various metrics. It supports automatic mixed precision (AMP)
    and can handle both multiclass and binary classification tasks.
    Attributes:
        metric_dict (Dict[str, Metric]): A dictionary of metrics to evaluate the model.
        epoch (int): The current epoch number.
        device (Union[str, torch.device]): The device to run the evaluation on (e.g., "cuda" or "cpu").
        num_classes (int): The number of classes in the classification task.
    Methods:
        __init__(num_classes: int, device: Union[str, torch.device] = "cuda",
                 metric_dict: Optional[Dict[str, Metric]] = None, ignore_index: int = 0, **kwargs: Any):
            Initializes the Evaluator with the specified number of classes, device, and metrics.
        evaluate_model(dataloader: DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
                       meta_model: MetaModel) -> Dict[str, Union[float, NDArray[np.float64]]]:
            Evaluates the performance of a given meta-model on a dataset provided by the dataloader.
        evaluate_image(dataloader: DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
                       meta_model: MetaModel, epoch: int = 0, log_epochs: int = 5) -> tuple[
                           NDArray[np.float64], Union[NDArray[np.int_], int, str], Union[NDArray[np.int_], int, str]]:
            Evaluates the given model on one image from the dataloader and returns the image, label, and prediction.
    """

    metric_dict: Dict[str, Metric]

    def __init__(
        self,
        num_classes: int,
        device: Union[str, torch.device] = "cuda",
        metric_dict: Optional[Dict[str, Metric]] = None,
        ignore_index: int = 0,
        **kwargs: Any
    ):

        self.epoch = 0
        self.device = device
        self.num_classes = num_classes

        if metric_dict:
            self.metric_dict = metric_dict
        else:
            self.metric_dict = {
                "accuracy": Accuracy(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=int(num_classes),
                    ignore_index=ignore_index,
                    **kwargs
                ).to(self.device),
            }

    @torch.no_grad()
    def evaluate_model(
        self,
        dataloader: DataLoader[
            Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        meta_model: MetaModel,
    ) -> Dict[str, Union[float, NDArray[np.float64]]]:
        """
        Evaluates the performance of a given meta-model on a dataset provided by the dataloader.
        Args:
            dataloader (DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]):
                A DataLoader object that provides batches of data. Each batch can either be a tuple
                of input tensors and labels or a dictionary containing "pixel_values" and "labels".
            meta_model (MetaModel):
                The meta-model to be evaluated. It contains the model and additional configurations
                such as whether to use automatic mixed precision (AMP).
        Returns:
            Dict[str, Union[float, NDArray[np.float64]]]:
                A dictionary containing the computed metrics. Each key corresponds to a metric name,
                and the value is either a float (for scalar metrics) or a NumPy array (for metrics
                with multiple dimensions).
        """
        meta_model.model.eval()
        metric_results: Dict[str, Union[float, NDArray[np.float64]]] = {}

        for data in dataloader:
            inputs, labels = data
            labels = labels.long().to(self.device)
            outputs = meta_model.batch_predict(inputs)
            outputs = outputs.argmax(dim=1)
            ## Update metrics
            for metric in self.metric_dict.values():
                metric.update(outputs, labels)
        ## Compute metrics
        for metric_name in self.metric_dict:
            metric_results[metric_name] = (
                self.metric_dict[metric_name].compute().cpu().numpy()
            )
            if metric_results[metric_name].ndim == 0:
                metric_results[metric_name] = metric_results[metric_name].item()
            self.metric_dict[metric_name].reset()

        self.epoch += 1
        return metric_results

    @torch.no_grad()
    def evaluate_image(
        self,
        dataloader: DataLoader[
            Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        meta_model: MetaModel,
        epoch: int = 0,
        log_epochs: int = 5,
        proba: bool = False,
    ) -> tuple[
        NDArray[np.float64],
        Union[NDArray[np.int_], int],
        Union[NDArray[np.int_], int],
    ]:
        """
        Evaluates the given model on one image of the dataloader.
        Args:
            dataloader (DataLoader[Union[tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]]):
                A DataLoader object that provides batches of data. Each batch can either be a tuple
                of input tensors and labels or a dictionary containing "pixel_values" and "labels".
            meta_model (MetaModel):
                The meta-model to be evaluated. It contains the model and additional configurations
                such as whether to use automatic mixed precision (AMP).
            epoch (int, optional): The epoch number for logging purposes. Defaults to 0.
        Returns:
            dict: A dictionary containing the computed metric results.
        """

        meta_model.model.eval()
        with torch.no_grad():
            data = next(iter(dataloader))
            inputs, labels = data

            outputs = meta_model.batch_predict(inputs)
            if not proba:
                outputs = outputs.argmax(dim=1)

        image_counter = (
            epoch % (log_epochs * 5) // 5
        )  # Rotating 5 images (assuming batch size above 5)
        image_counter = image_counter % inputs.size(
            dim=0
        )  # In case we use a smaller batch size

        image: NDArray[np.float64] = inputs[image_counter].cpu().numpy()
        label: NDArray[np.int_] = labels[image_counter].cpu().numpy()
        pred: Union[NDArray[np.int_], NDArray[np.float_]] = (
            outputs[image_counter].cpu().numpy()
        )

        return image, label, pred


class EvaluatorSemanticSegmentation(Evaluator):
    """
    A class for evaluating segmentation models, inheriting from the base `Evaluator` class.
    Attributes:
        metric_dict (dict): A dictionary of metrics used for evaluation. Defaults to
            accuracy and mean IoU, configured for either binary or multiclass tasks
            based on the number of classes.
    Args:
        num_classes (int): The number of classes in the classification task.
        device (Union[str, torch.device], optional): The device on which the metrics will be computed.
            Defaults to "cuda".
        metric_dict (Optional[Dict[str, Metric]], optional): A dictionary of pre-defined metrics. If not provided,
            default metrics (accuracy and F1 score) are initialized based on the number of classes.
        ignore_index (int, optional): Specifies a target value that is ignored during metric computation.
            Defaults to 0.
    """

    def __init__(
        self,
        num_classes: int,
        device: Union[str, torch.device] = "cuda",
        metric_dict: Optional[Dict[str, Metric]] = None,
        ignore_index: int = 0,
        **kwargs: Any
    ):

        super().__init__(
            num_classes=num_classes,
            device=device,
            metric_dict=metric_dict,
            ignore_index=ignore_index,
            **kwargs
        )

        if metric_dict:
            self.metric_dict = metric_dict
        else:
            self.metric_dict = {
                "accuracy": Accuracy(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ).to(device),
                "mean_iou": JaccardIndex(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ).to(device),
            }
        self.metric_dict = {
            metric_name: metric.to(device)
            for metric_name, metric in self.metric_dict.items()
        }
