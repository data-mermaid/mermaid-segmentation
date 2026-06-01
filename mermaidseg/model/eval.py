import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex
from torchmetrics.metric import Metric

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.meta import MetaModel


def map_taxonomy_to_dense(binary_taxonomy: torch.Tensor) -> torch.Tensor:
    """Map taxonomic one-hot/binary labels to dense IDs.

    Maps binary/one-hot taxonomy labels to dense IDs where:
    - 0 represents None (all zeros)
    - 1 represents Not Given (all ones)
    - 2+ represents the argmax index + 2 (for other categories)
    """
    sum_ = binary_taxonomy.sum(dim=1)
    is_none = sum_.eq(0)  # if all zeros, then sum == 0 which represents None
    is_not_given = sum_.eq(
        binary_taxonomy.size(1)
    )  # if all ones, then sum == C which represents Not Given

    taxonomy_id = (
        binary_taxonomy.argmax(dim=1) + 2
    )  # The remaining values start from 2, since 0 is None and 1 is Not Given. So if the argmax is 0, we want to assign it to 2, if it's 1, we want to assign it to 3, and so on.
    return torch.where(
        is_none,
        torch.zeros_like(taxonomy_id),
        torch.where(is_not_given, torch.ones_like(taxonomy_id), taxonomy_id),
    )


def map_taxonomy_predictions_to_dense(pred_taxonomy: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """Map predicted taxonomic scores to dense IDs.

    Converts predicted taxonomy scores to dense IDs:
    - If max probability > threshold: assign argmax + 2 (to account for None and Not Given)
    - If max probability <= threshold: assign 1 (Not Given)
    """
    max_vals, argmax_idx = pred_taxonomy.max(dim=1)
    return torch.where(
        max_vals
        > threshold,  # If the probability across the prediction is smaller than threshold, we will assign it to Not Given (1), otherwise we will assign it to the argmax + 2 (to account for None and Not Given)
        argmax_idx + 2,
        torch.ones_like(argmax_idx),
    )


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
    ):
        self.epoch = 0
        self.device = device
        self.num_classes = num_classes
        self.concept_value2id = concept_value2id

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

        if calculate_concept_metrics:
            if concept_metric_dict:
                self.concept_metric_dict = concept_metric_dict
            else:
                self.concept_metric_dict = {
                    "mean_concept_f1": F1Score(
                        task="multiclass", average="none", num_classes=3, ignore_index=0
                    ).to(self.device)
                }
                if self.concept_value2id is not None:
                    for concept in self.concept_value2id:
                        if concept in TAXONOMIC_CONCEPTS:
                            concept_values = self.concept_value2id[concept]
                            order_concept_length = (
                                len(list(concept_values.values())[0]) + 2
                            )  # +2 to account for the "None" and "Not Given" classes that we add to the taxonomic concepts
                            self.concept_metric_dict[f"{concept}_f1"] = F1Score(
                                task="multiclass",
                                average="none",
                                num_classes=order_concept_length,
                                ignore_index=0,
                            ).to(self.device)
                        else:
                            self.concept_metric_dict[f"{concept}_f1"] = F1Score(
                                task="multiclass", average="none", num_classes=3, ignore_index=0
                            ).to(self.device)

            self.concept_metric_dict = {
                metric_name: metric.to(device)
                for metric_name, metric in self.concept_metric_dict.items()
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

    def evaluate_concepts(self, concept_outputs, concept_labels):
        """Update concept metrics for taxonomic and binary concepts."""
        concept_outputs_mean = (concept_outputs > 0.5).float()
        concept_outputs_mean += 1
        for metric_name, metric in self.concept_metric_dict.items():
            if "mean_concept" in metric_name:
                metric.update(concept_outputs_mean, concept_labels)

        offset = 0

        ### Calculate taxonomic concept metrics
        for concept in TAXONOMIC_CONCEPTS:
            concept_values = self.concept_value2id[concept]
            order_concept_length = len(list(concept_values.values())[0])
            concept_labels_order = concept_labels[:, offset : offset + order_concept_length, ...]
            concept_outputs_order = concept_outputs[:, offset : offset + order_concept_length, ...]
            offset += order_concept_length

            concept_labels_order = map_taxonomy_to_dense(concept_labels_order)
            concept_outputs_order = map_taxonomy_predictions_to_dense(concept_outputs_order)
            self.concept_metric_dict[f"{concept}_f1"].update(
                concept_outputs_order, concept_labels_order
            )

        ### Calculate binary concept metrics
        concept_labels_binary = concept_labels[:, offset:, ...]
        concept_outputs_binary = concept_outputs[:, offset:, ...]
        concept_outputs_binary = (concept_outputs_binary > 0.5).float()
        concept_outputs_binary += 1
        for concept in self.concept_value2id:
            if concept not in TAXONOMIC_CONCEPTS:
                self.concept_metric_dict[f"{concept}_f1"].update(
                    concept_outputs_binary[:, offset : offset + 1, ...],
                    concept_labels_binary[:, offset : offset + 1, ...],
                )
                offset += 1
