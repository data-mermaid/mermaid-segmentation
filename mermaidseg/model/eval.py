"""Metric accumulation for segmentation and concept predictions."""

import torch
from torchmetrics.classification import Accuracy
from torchmetrics.metric import Metric

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.model.concept_metrics import (
    map_taxonomy_predictions_to_dense,
    map_taxonomy_to_dense,
)


class Evaluator:
    """Base evaluator for machine learning models.

    Accumulates metrics across batches and supports both multiclass and binary tasks.
    ``MetaModel`` updates the metrics with target-space labels each batch.

    Attributes:
        metric_dict: Metrics to accumulate and compute.
        device: Device metrics tensors are moved to.
        num_classes: Number of output (target) classes.
    """

    metric_dict: dict[str, Metric]

    def __init__(
        self,
        num_classes: int,
        device: str | torch.device,
        calculate_concept_metrics: bool = False,
        concept_value2id: dict[str, dict[str, list[int]]] | None = None,
    ) -> None:
        """Build the accuracy metrics on the target device.

        Args:
            num_classes: Number of output (target) classes.
            device: Device metrics tensors are moved to.
            calculate_concept_metrics: Whether to build per-concept accuracy metrics.
            concept_value2id: Concept name to value-to-id mapping, required for concept metrics.
        """
        self.device = device
        self.num_classes = num_classes
        self.concept_value2id = concept_value2id
        self._binary_accuracy_metrics: dict[str, Metric] = {}

        self.metric_dict = {
            "accuracy": Accuracy(
                task="multiclass" if num_classes > 2 else "binary",
                num_classes=num_classes,
                ignore_index=0,
            ).to(device),
        }

        self.metric_dict = {
            metric_name: metric.to(device) for metric_name, metric in self.metric_dict.items()
        }

        if calculate_concept_metrics:
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
        """Compute and reset the concept metrics, returning results by metric name."""
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

    def evaluate_concepts(
        self, concept_outputs: torch.Tensor, concept_labels: torch.Tensor
    ) -> None:
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
