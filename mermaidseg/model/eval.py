import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchmetrics.metric import Metric

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.concept_metrics import (
    map_taxonomy_predictions_to_dense,
    map_taxonomy_to_dense,
)
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
        concept_value2id: dict[str, dict[str, int]] | None = None,
        ignore_index: int = 0,
        include_classification: bool = True,
    ):
        self.epoch = 0
        self.device = device
        self.num_classes = num_classes
        self.concept_value2id = concept_value2id
        self._binary_accuracy_metrics: dict[str, Metric] = {}

        if metric_dict:
            self.metric_dict = metric_dict
        elif include_classification:
            self.metric_dict = {
                "accuracy": Accuracy(
                    task="multiclass" if num_classes > 2 else "binary",
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ).to(device),
            }
        else:
            self.metric_dict = {}

        self.metric_dict = {
            metric_name: metric.to(device) for metric_name, metric in self.metric_dict.items()
        }

        if calculate_concept_metrics:
            if concept_metric_dict:
                self.concept_metric_dict = concept_metric_dict
            else:
                self.concept_metric_dict = {}
                if self.concept_value2id is not None:
                    for concept in TAXONOMIC_CONCEPTS:
                        if concept not in self.concept_value2id:
                            continue
                        concept_values = self.concept_value2id[concept]
                        order_concept_length = len(list(concept_values.values())[0]) + 2
                        self.concept_metric_dict[f"accuracy/{concept}"] = Accuracy(
                            task="multiclass",
                            num_classes=order_concept_length,
                            ignore_index=0,
                        ).to(self.device)
                    for concept in self.concept_value2id:
                        if concept not in TAXONOMIC_CONCEPTS:
                            self._binary_accuracy_metrics[concept] = Accuracy(
                                task="multiclass",
                                num_classes=3,
                                ignore_index=0,
                            ).to(self.device)

            self.concept_metric_dict = {
                metric_name: metric.to(device)
                for metric_name, metric in self.concept_metric_dict.items()
            }
            self._binary_accuracy_metrics = {
                metric_name: metric.to(device)
                for metric_name, metric in self._binary_accuracy_metrics.items()
            }
        else:
            self.concept_metric_dict = {}
            self._binary_accuracy_metrics = {}

    def compute_concept_metric_results(self) -> dict[str, float]:
        return self._compute_concept_metric_results()

    def _compute_concept_metric_results(self) -> dict[str, float]:
        if not self.concept_metric_dict and not self._binary_accuracy_metrics:
            return {}
        metric_results: dict[str, float] = {}
        for metric_name in self.concept_metric_dict:
            value = self.concept_metric_dict[metric_name].compute().cpu().numpy()
            if value.ndim == 0:
                metric_results[metric_name] = float(value.item())
            else:
                metric_results[metric_name] = float(value.mean())
            self.concept_metric_dict[metric_name].reset()

        if self._binary_accuracy_metrics:
            binary_values = []
            for metric in self._binary_accuracy_metrics.values():
                value = metric.compute().cpu().numpy()
                binary_values.append(float(value.item() if value.ndim == 0 else value.mean()))
                metric.reset()
            metric_results["accuracy/multi_hot"] = sum(binary_values) / len(binary_values)

        return metric_results

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

            if self.metric_dict:
                if outputs.ndim > 3:
                    outputs = outputs.argmax(dim=1)
                for metric in self.metric_dict.values():
                    metric.update(outputs, target_labels)

            if meta_model.training_mode in ("concept-bottleneck", "concept"):
                concept_labels = meta_model._to_concept_labels(source_labels)
                self.evaluate_concepts(concept_outputs, concept_labels)

        for metric_name in self.metric_dict:
            metric_results[metric_name] = self.metric_dict[metric_name].compute().cpu().numpy()
            if metric_results[metric_name].ndim == 0:
                metric_results[metric_name] = metric_results[metric_name].item()
            self.metric_dict[metric_name].reset()

        if meta_model.training_mode in ("concept-bottleneck", "concept"):
            metric_results.update(self._compute_concept_metric_results())

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

    def evaluate_concepts(self, concept_outputs, concept_labels):
        """Update concept accuracy metrics for taxonomic and binary concepts."""
        if self.concept_value2id is None:
            return

        offset = 0

        for concept in TAXONOMIC_CONCEPTS:
            if concept not in self.concept_value2id:
                continue
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            concept_labels_order = concept_labels[:, offset : offset + order_concept_length, ...]
            concept_outputs_order = concept_outputs[:, offset : offset + order_concept_length, ...]
            offset += order_concept_length

            concept_labels_order = map_taxonomy_to_dense(concept_labels_order)
            concept_outputs_order = map_taxonomy_predictions_to_dense(concept_outputs_order)
            metric_key = f"accuracy/{concept}"
            if metric_key not in self.concept_metric_dict:
                continue
            self.concept_metric_dict[metric_key].update(concept_outputs_order, concept_labels_order)

        concept_labels_binary = concept_labels[:, offset:, ...]
        concept_outputs_binary = concept_outputs[:, offset:, ...]
        concept_outputs_binary = (concept_outputs_binary > 0.5).float()
        concept_outputs_binary += 1
        binary_offset = 0
        for concept in self.concept_value2id:
            if concept not in TAXONOMIC_CONCEPTS:
                if concept not in self._binary_accuracy_metrics:
                    binary_offset += 1
                    continue
                self._binary_accuracy_metrics[concept].update(
                    concept_outputs_binary[:, binary_offset : binary_offset + 1, ...],
                    concept_labels_binary[:, binary_offset : binary_offset + 1, ...],
                )
                binary_offset += 1


EvaluatorSemanticSegmentation = Evaluator
