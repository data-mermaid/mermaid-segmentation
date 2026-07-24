import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex
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

    Args:
        per_class_metrics: When True (and no explicit ``metric_dict`` is given), also
            accumulates ``f1_per_class`` and ``iou_per_class`` (macro components,
            per-class vectors) alongside scalar ``accuracy``/``miou``. Off by default
            since class cardinality (especially for concept ranks like genus) can make
            per-class series numerous; callers should default this based on model type
            (e.g. on for standard segmentation, off for concept/concept-bottleneck).
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
        per_class_metrics: bool = False,
    ):
        self.epoch = 0
        self.device = device
        self.num_classes = num_classes
        self.concept_value2id = concept_value2id
        self._binary_accuracy_metrics: dict[str, Metric] = {}

        if metric_dict:
            self.metric_dict = metric_dict
        elif include_classification:
            task = "multiclass" if num_classes > 2 else "binary"
            shared_kwargs = {
                "task": task,
                "num_classes": num_classes,
                "ignore_index": ignore_index,
            }
            self.metric_dict = {
                "accuracy": Accuracy(**shared_kwargs).to(device),
                # Mean IoU (macro-averaged over classes, excluding ignore_index) — a more
                # informative segmentation metric than pixel accuracy, which is dominated by
                # majority classes (e.g. background) in class-imbalanced coral reef data.
                "miou": JaccardIndex(**shared_kwargs, average="macro").to(device),
                # `miou` (macro) weights every *present* class equally — a rare class counts as
                # much as a common one (torchmetrics macro already excludes classes absent from
                # the target). `miou_weighted` weights each class's IoU by its support, so comparing
                # the two shows whether a change helps rare vs common classes. It is a scalar (like
                # accuracy/miou) and MUST always be produced when classification metrics are on:
                # metric_policy advertises it as a selectable metric_of_interest, so gating it (e.g.
                # behind per_class_metrics) would crash checkpoint/early-stopping when it's selected.
                "miou_weighted": JaccardIndex(**shared_kwargs, average="weighted").to(device),
            }
            if per_class_metrics:
                # average="none" returns a per-class vector instead of a scalar; Logger
                # unpacks these into named metrics (e.g. "f1_per_class/acropora") via
                # id2label, so class-level regressions are visible without re-running eval.
                self.metric_dict["f1_per_class"] = F1Score(**shared_kwargs, average="none").to(
                    device
                )
                self.metric_dict["iou_per_class"] = JaccardIndex(
                    **shared_kwargs, average="none"
                ).to(device)
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

    def accumulate(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the classification metric bank with one batch.

        ``preds`` may be raw logits (``(B, C, H, W)`` — argmaxed to class ids) or
        already class-id maps (``(B, H, W)`` — used as- is). Callers pass the mode-
        appropriate ``(preds, targets)`` pair (e.g. ``(logits, target_labels)`` for
        standard/CBM, or ``(concept_class_preds, target_concept_preds)`` for concept
        mode). No-op when the classification bank is empty. This is the single owner of
        the metric-update lifecycle, shared by the train/val loops and
        :meth:`evaluate_model`.
        """
        if not self.metric_dict:
            return
        if preds.ndim > 3:
            preds = preds.argmax(dim=1)
        preds = preds.detach()
        for metric in self.metric_dict.values():
            metric.update(preds, targets)

    def compute_and_reset(
        self, include_concepts: bool = False
    ) -> dict[str, float | NDArray[np.float64]]:
        """Compute every accumulated metric, reset the bank(s), and return the results.

        Scalars are returned as Python floats; per-class metrics stay as arrays. When
        ``include_concepts`` is True, concept metrics are computed + reset and merged
        in.
        """
        metric_results: dict[str, float | NDArray[np.float64]] = {}
        for metric_name in self.metric_dict:
            value = self.metric_dict[metric_name].compute().cpu().numpy()
            metric_results[metric_name] = value.item() if value.ndim == 0 else value
            self.metric_dict[metric_name].reset()
        if include_concepts:
            metric_results.update(self._compute_concept_metric_results())
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
        is_concept = meta_model.has_concepts
        for data in tqdm.tqdm(dataloader):
            inputs, source_labels = data
            source_labels = source_labels.long().to(self.device)
            target_labels = meta_model._to_target_labels(source_labels)
            outputs, concept_outputs = meta_model.batch_predict(inputs)

            self.accumulate(outputs, target_labels)
            if is_concept:
                concept_labels = meta_model._to_concept_labels(source_labels)
                self.evaluate_concepts(concept_outputs, concept_labels)

        metric_results = self.compute_and_reset(include_concepts=is_concept)
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
