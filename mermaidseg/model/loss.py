from collections.abc import Iterator
from typing import Any

import torch

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.concept_metrics import (
    calculate_multi_hot_concept_loss,
    calculate_taxonomic_rank_loss,
)


def iter_concept_slices(
    concept_value2id: dict[str, dict[str, int]],
    concept_labels: torch.Tensor,
    concept_outputs: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (name, labels_slice, outputs_slice) for each taxonomic rank and the binary tail."""
    offset = 0
    for concept in TAXONOMIC_CONCEPTS:
        if concept not in concept_value2id:
            continue
        concept_values = concept_value2id[concept]
        order_concept_length = len(list(concept_values.values())[0])
        yield (
            concept,
            concept_labels[:, offset : offset + order_concept_length, ...],
            concept_outputs[:, offset : offset + order_concept_length, ...],
        )
        offset += order_concept_length
    yield (
        "multi_hot",
        concept_labels[:, offset:, ...],
        concept_outputs[:, offset:, ...],
    )


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """CrossEntropyLoss is a wrapper of `torch.nn.CrossEntropyLoss` that allows for additional
    customization.

    Attributes:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            This is useful for masking certain values in the target tensor. Defaults to -1.
        kwargs: Additional keyword arguments that are passed to the base `torch.nn.CrossEntropyLoss` class.
    """

    def __init__(self, ignore_index: int = 0, **kwargs: Any) -> None:
        super().__init__(ignore_index=ignore_index, **kwargs)

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = super().forward(outputs, target_labels)
        return loss, {"classification": loss.item()}


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """BCE loss for concept prediction that masks background pixels before averaging.

    Wraps `torch.nn.BCEWithLogitsLoss` with `reduction="none"` and applies a foreground mask derived
    from `labels` so background pixels (label == 0) do not contribute to the mean.
    """

    def __init__(
        self,
        reduction: str = "none",
        concept_value2id: dict[str, dict[str, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.concept_value2id = concept_value2id

    def _slice_loss(
        self,
        name: str,
        concept_outputs: torch.Tensor,
        concept_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        foreground_mask = target_labels > 0
        if name == "multi_hot":
            return calculate_multi_hot_concept_loss(
                concept_outputs,
                concept_labels,
                self,
                from_logits=True,
                foreground_mask=foreground_mask,
            )
        return calculate_taxonomic_rank_loss(
            concept_outputs,
            concept_labels,
            from_logits=True,
            foreground_mask=foreground_mask,
        )

    def forward(
        self,
        concept_outputs: torch.Tensor,
        concept_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute masked concept loss over foreground pixels.

        Args:
            concept_outputs (torch.Tensor): Concept logits with shape (B, C, H, W).
            concept_labels (torch.Tensor): Binary concept targets with shape (B, C, H, W).
            target_labels (torch.Tensor): Target-space segmentation labels with shape
                (B, H, W); background == 0.
        Returns:
            A 2-tuple of (total_loss, loss_components) for logging.
        """
        if self.concept_value2id is None:
            foreground_mask = target_labels > 0
            total_loss = calculate_multi_hot_concept_loss(
                concept_outputs,
                concept_labels,
                self,
                from_logits=True,
                foreground_mask=foreground_mask,
            )
            return total_loss, {"concepts": total_loss.item()}

        loss_components: dict[str, float] = {}
        concept_loss_value = torch.tensor(
            0.0, device=concept_outputs.device, dtype=concept_outputs.dtype
        )
        for name, labels_slice, outputs_slice in iter_concept_slices(
            self.concept_value2id, concept_labels, concept_outputs
        ):
            slice_loss = self._slice_loss(name, outputs_slice, labels_slice, target_labels)
            loss_components[name] = slice_loss.item()
            concept_loss_value = concept_loss_value + slice_loss

        loss_components["concepts"] = concept_loss_value.item()
        return concept_loss_value, loss_components


class ConceptBottleneckLoss(torch.nn.Module):
    """ConceptBottleneckLoss combines a classification loss with a concept prediction loss.

    It computes the total loss as the sum of the classification loss and a weighted concept loss.
    The concept loss operates on the model's raw concept *logits* (pre-activation): taxonomic
    groups use cross-entropy and the binary tail uses ``binary_cross_entropy_with_logits``. This
    is numerically stable under mixed precision, unlike running BCE on post-sigmoid activations.

    Attributes:
        class_loss (torch.nn.Module): The loss function used for classification. Defaults to
            `torch.nn.CrossEntropyLoss`.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient
            for the classification loss. Defaults to -1.
        lambda_weight (float): The weight applied to the concept loss when computing the total loss. Defaults to 1.0.
        kwargs: Additional keyword arguments that are passed to the classification loss.
    """

    def __init__(
        self,
        concept_value2id: dict[str, dict[str, int]],
        class_loss: torch.nn.Module = torch.nn.CrossEntropyLoss,
        ignore_index: int = 0,
        lambda_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.class_loss = class_loss(ignore_index=ignore_index, **kwargs)
        self.lambda_weight = lambda_weight
        self.concept_value2id = concept_value2id

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the total loss as the sum of the classification loss and a weighted concept
        loss.

        Args:
            outputs (torch.Tensor): The model's output logits for classification.
            target_labels (torch.Tensor): The target-space ground truth labels for classification.
            concept_logits (torch.Tensor): The model's raw concept logits (pre-activation).
                Taxonomic ranks are scored with cross-entropy and the binary tail with
                ``binary_cross_entropy_with_logits`` for numerical stability.
            concept_labels (torch.Tensor): The ground truth labels for concept prediction.
        Returns:
            A 2-tuple of (total_loss, loss_components) where total_loss is the
            scalar tensor for backprop and loss_components is a dict of detached
            component values for logging.
        """
        class_loss_value = self.class_loss(outputs, target_labels)
        if (target_labels > 0).sum() == 0:
            class_loss_value = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

        loss_components: dict[str, float] = {"classification": class_loss_value.item()}
        concept_loss_value = torch.tensor(
            0.0, device=concept_logits.device, dtype=concept_logits.dtype
        )

        for name, labels_slice, outputs_slice in iter_concept_slices(
            self.concept_value2id, concept_labels, concept_logits
        ):
            if name == "multi_hot":
                slice_loss = calculate_multi_hot_concept_loss(
                    outputs_slice, labels_slice, from_logits=True
                )
            else:
                slice_loss = calculate_taxonomic_rank_loss(
                    outputs_slice, labels_slice, from_logits=True
                ) / len(TAXONOMIC_CONCEPTS)
            loss_components[name] = slice_loss.item()
            concept_loss_value = concept_loss_value + slice_loss

        loss_components["concepts"] = concept_loss_value.item()
        total_loss = class_loss_value + self.lambda_weight * concept_loss_value
        return total_loss, loss_components
