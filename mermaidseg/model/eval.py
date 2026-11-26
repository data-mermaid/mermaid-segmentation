"""Metric accumulation for segmentation and concept predictions."""

import torch
from torchmetrics.classification import Accuracy
from torchmetrics.metric import Metric

from mermaidseg.dataset_reconciliation.concept_ranks import (
    TAXONOMIC_CONCEPTS,
    binary_offset,
    rank_slices,
)
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
        concept_value2id: dict[str, dict[str, list[int]]],
    ) -> None:
        """Build the accuracy and per-concept accuracy metrics on the target device.

        Args:
            num_classes: Number of output (target) classes.
            device: Device metrics tensors are moved to.
            concept_value2id: Concept name to value-to-id mapping.
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

        self.concept_metric_dict: dict[str, Metric] = {}
        for concept, _, width in rank_slices(self.concept_value2id):
            self.concept_metric_dict[f"accuracy/{concept}"] = Accuracy(
                task="multiclass",
                num_classes=width + 2,
                ignore_index=0,
            ).to(self.device)
        for concept in self.concept_value2id:
            if concept not in TAXONOMIC_CONCEPTS:
                self._binary_accuracy_metrics[concept] = Accuracy(
                    task="multiclass",
                    num_classes=3,
                    ignore_index=0,
                ).to(self.device)

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
        for concept, offset, width in rank_slices(self.concept_value2id):
            concept_labels_order = concept_labels[:, offset : offset + width, ...]
            concept_outputs_order = concept_outputs[:, offset : offset + width, ...]

            concept_labels_order = map_taxonomy_to_dense(concept_labels_order)
            concept_outputs_order = map_taxonomy_predictions_to_dense(concept_outputs_order)

            metric_key = f"accuracy/{concept}"
            if metric_key not in self.concept_metric_dict:
                continue
            self.concept_metric_dict[metric_key].update(concept_outputs_order, concept_labels_order)

        tail = binary_offset(self.concept_value2id)
        concept_labels_binary = concept_labels[:, tail:, ...]
        concept_outputs_binary = concept_outputs[:, tail:, ...]
        concept_outputs_binary = (concept_outputs_binary > 0.5).float()
        concept_outputs_binary += 1
        channel = 0
        for concept in self.concept_value2id:
            if concept not in TAXONOMIC_CONCEPTS:
                if concept not in self._binary_accuracy_metrics:
                    channel += 1
                    continue
                self._binary_accuracy_metrics[concept].update(
                    concept_outputs_binary[:, channel : channel + 1, ...],
                    concept_labels_binary[:, channel : channel + 1, ...],
                )
                channel += 1
