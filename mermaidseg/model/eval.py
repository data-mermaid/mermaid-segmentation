from typing import Any

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex
from torchmetrics.metric import Metric

from mermaidseg.model.meta import MetaModel


class Evaluator:
    """Base evaluator for machine learning models.

    Accumulates metrics across batches and supports both multiclass and binary tasks.
    Operates on ``(inputs, source_labels)`` batches and converts the source
    labels to target-space labels via ``meta_model._to_target_labels`` for
    metric computation.

    Attributes:
        metric_dict (dict[str, Metric]): Metrics to accumulate and compute.
        epoch (int): Current epoch counter (incremented after each `evaluate_model` call).
        device (str | torch.device): Device metrics tensors are moved to.
        num_classes (int): Number of output (target) classes.
    """

    metric_dict: dict[str, Metric]

    def __init__(
        self,
        num_classes: int,
        device: str | torch.device = "cuda",
        metric_dict: dict[str, Metric] | None = None,
        calculate_concept_metrics: bool = False,
        concept_metric_dict: dict[str, Metric] | None = None,
        ignore_index: int = 0,
        **kwargs: Any,
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
                    **kwargs,
                ).to(self.device),
            }

        if calculate_concept_metrics:
            if concept_metric_dict:
                self.concept_metric_dict = concept_metric_dict
            else:
                self.concept_metric_dict = {
                    "f1_concept": F1Score(
                        task="multiclass", average="none", num_classes=3, ignore_index=0
                    ).to(self.device)
                }

    @torch.no_grad()
    def evaluate_model(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        meta_model: MetaModel,
    ) -> dict[str, float | NDArray[np.float64]]:
        """Evaluate ``meta_model`` over ``dataloader``.

        Args:
            dataloader: DataLoader yielding ``(inputs, source_labels)`` batches.
            meta_model: The model wrapper providing ``batch_predict`` and the
                source/target lookup tensors.
        Returns:
            Dict of metric_name -> scalar/array result.
        """
        meta_model.model.eval()
        metric_results: dict[str, float | NDArray[np.float64]] = {}
        for data in tqdm.tqdm(dataloader):
            inputs, source_labels = data
            source_labels = source_labels.long().to(self.device)
            target_labels = meta_model._to_target_labels(source_labels)
            outputs, concept_outputs = meta_model.batch_predict(inputs)

            if outputs.ndim > 3:
                outputs = outputs.argmax(dim=1)
            for metric in self.metric_dict.values():
                metric.update(outputs, target_labels)

            if meta_model.training_mode in ("concept-bottleneck", "concept"):
                concept_outputs = (concept_outputs > 0.5).float()
                concept_outputs += 1
                concept_outputs *= target_labels.unsqueeze(1) != 0

                concept_labels = meta_model._to_concept_labels(source_labels)
                concept_labels += 1
                concept_labels *= target_labels.unsqueeze(1) != 0

                for metric in self.concept_metric_dict.values():
                    metric.update(concept_outputs, concept_labels)

        for metric_name in self.metric_dict:
            metric_results[metric_name] = self.metric_dict[metric_name].compute().cpu().numpy()
            if metric_results[metric_name].ndim == 0:
                metric_results[metric_name] = metric_results[metric_name].item()
            self.metric_dict[metric_name].reset()

        if meta_model.training_mode in ("concept-bottleneck", "concept"):
            for metric_name in self.concept_metric_dict:
                metric_results[metric_name] = (
                    self.concept_metric_dict[metric_name].compute().cpu().numpy()
                )
                if metric_results[metric_name].ndim == 0:
                    metric_results[metric_name] = metric_results[metric_name].item()
                self.concept_metric_dict[metric_name].reset()

        self.epoch += 1
        return metric_results

    @torch.no_grad()
    def evaluate_image(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]],
        meta_model: MetaModel,
        epoch: int = 0,
        log_epochs: int = 5,
        proba: bool = False,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int_] | int,
        NDArray[np.int_] | int,
    ]:
        """Return one image, its ground-truth target label, and the model prediction."""
        meta_model.model.eval()
        with torch.no_grad():
            data = next(iter(dataloader))
            inputs, source_labels = data
            target_labels = meta_model._to_target_labels(source_labels.long().to(self.device))

            outputs, concept_outputs = meta_model.batch_predict(inputs)
            if not proba:
                outputs = outputs.argmax(dim=1)

        image_counter = (
            epoch % (log_epochs * 5) // 5
        )  # Rotating 5 images (assuming batch size above 5)
        image_counter = image_counter % inputs.size(dim=0)  # In case we use a smaller batch size

        image: NDArray[np.float64] = inputs[image_counter].cpu().numpy()
        label: NDArray[np.int_] = target_labels[image_counter].cpu().numpy()
        pred: NDArray[np.int_] | NDArray[np.float_] = outputs[image_counter].cpu().numpy()

        return image, label, pred


class EvaluatorSemanticSegmentation(Evaluator):
    """A class for evaluating segmentation models, inheriting from the base `Evaluator` class.

    Attributes:
        metric_dict (dict): A dictionary of metrics used for evaluation. Defaults to
            accuracy and mean IoU, configured for either binary or multiclass tasks
            based on the number of classes.
    Args:
        num_classes (int): The number of (target) classes in the classification task.
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
        device: str | torch.device = "cuda",
        metric_dict: dict[str, Metric] | None = None,
        ignore_index: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            num_classes=num_classes,
            device=device,
            metric_dict=metric_dict,
            ignore_index=ignore_index,
            **kwargs,
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
                "per_class_iou": JaccardIndex(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=ignore_index,
                ).to(device),
                "per_class_f1": F1Score(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=ignore_index,
                ).to(device),
            }
        self.metric_dict = {
            metric_name: metric.to(device) for metric_name, metric in self.metric_dict.items()
        }
