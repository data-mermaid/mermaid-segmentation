"""Classification and concept-bottleneck losses."""

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.model.concept_metrics import (
    calculate_multi_hot_concept_loss,
    calculate_taxonomic_rank_loss,
    reduce_masked_loss,
)


def iter_concept_slices(
    concept_value2id: dict[str, dict[str, list[int]]],
    concept_labels: torch.Tensor,
    concept_outputs: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (name, labels_slice, outputs_slice) per taxonomic rank and for the binary tail."""
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


def binary_concept_losses(
    concept_logits: torch.Tensor,
    concept_labels: torch.Tensor,
    names: list[str],
    *,
    per_pixel_loss_weight: float,
    per_image_loss_weight: float,
) -> dict[str, float]:
    """The ``multi_hot`` loss of each binary channel on its own, keyed ``multi_hot/<name>``.

    Same masked BCE and pixel/image blend as ``calculate_multi_hot_concept_loss``, reduced
    per channel. Channels with no labelled pixel in the batch are left out.
    """
    if concept_logits.shape[1] != len(names):
        raise ValueError(
            f"{len(names)} binary concept names for {concept_logits.shape[1]} channels"
        )
    mask = concept_labels.gt(0).to(concept_logits.dtype)
    targets = torch.clamp(concept_labels - 1, 0.01, 0.99)
    per_element = F.binary_cross_entropy_with_logits(concept_logits, targets, reduction="none")
    sums = (per_element * mask).flatten(2).sum(dim=2)
    counts = mask.flatten(2).sum(dim=2)
    channel_counts = counts.sum(dim=0)
    per_pixel = sums.sum(dim=0) / channel_counts.clamp(min=1)
    has_valid = counts > 0
    per_image_values = (sums / counts.clamp(min=1)) * has_valid
    per_image = per_image_values.sum(dim=0) / has_valid.sum(dim=0).clamp(min=1)
    values = per_pixel_loss_weight * per_pixel + per_image_loss_weight * per_image
    return {
        f"multi_hot/{name}": value
        for name, value, count in zip(names, values.tolist(), channel_counts.tolist(), strict=True)
        if count > 0
    }


def _masked_cross_entropy(
    outputs: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    ignore_index: int,
    label_smoothing: float,
    per_pixel_loss_weight: float,
    per_image_loss_weight: float,
) -> torch.Tensor:
    valid_mask = target_labels != ignore_index
    per_pixel = F.cross_entropy(
        outputs,
        target_labels,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return reduce_masked_loss(
        per_pixel,
        valid_mask,
        per_pixel_loss_weight=per_pixel_loss_weight,
        per_image_loss_weight=per_image_loss_weight,
    )


class ConceptBottleneckLoss(torch.nn.Module):
    """Combine a classification loss with a concept prediction loss.

    It computes the total loss as the sum of the classification loss and a weighted concept loss.
    The concept loss operates on the model's raw concept *logits* (pre-activation): taxonomic
    groups use cross-entropy and the binary tail uses ``binary_cross_entropy_with_logits``. This
    is numerically stable under mixed precision, unlike running BCE on post-sigmoid activations.
    """

    def __init__(
        self,
        concept_value2id: dict[str, dict[str, list[int]]],
        *,
        ignore_index: int,
        lambda_weight: float,
        per_pixel_loss_weight: float,
        per_image_loss_weight: float,
        class_loss: type[torch.nn.Module] = torch.nn.CrossEntropyLoss,
        taxonomic_label_smoothing: float = 0.01,
        binary_concept_names: list[str] | None = None,
    ) -> None:
        """Build the classification loss and store the concept-loss settings.

        Args:
            concept_value2id: Concept name to value-to-id mapping.
            class_loss: Loss class used for classification.
            ignore_index: Target value that is ignored and does not contribute to the input
                gradient for the classification loss.
            lambda_weight: Weight applied to the concept loss in the total loss.
            per_pixel_loss_weight: Weight on the batch-wide per-pixel mean of the loss, where all
                valid pixels are averaged equally (densely annotated images dominate).
            per_image_loss_weight: Weight on the per-image mean of the loss, where each image is
                first averaged over its valid pixels and then images are averaged equally
                (sparsely annotated images are not down-weighted).
            taxonomic_label_smoothing: Label smoothing for the taxonomic rank losses.
            binary_concept_names: Names of the binary channels, for per-channel loss components.
        """
        super().__init__()
        self.class_loss = class_loss(ignore_index=ignore_index)
        self.binary_concept_names = binary_concept_names
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight
        self.concept_value2id = concept_value2id
        self.per_pixel_loss_weight = per_pixel_loss_weight
        self.per_image_loss_weight = per_image_loss_weight
        self.taxonomic_label_smoothing = taxonomic_label_smoothing

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the classification loss plus the weighted concept loss.

        Args:
            outputs: The model's output logits for classification.
            target_labels: The target-space ground truth labels for classification.
            concept_logits: The model's raw concept logits (pre-activation).
                Taxonomic ranks are scored with cross-entropy and the binary tail with
                ``binary_cross_entropy_with_logits`` for numerical stability.
            concept_labels: The ground truth labels for concept prediction.

        Returns:
            A 2-tuple of (total_loss, loss_components) where total_loss is the
            scalar tensor for backprop and loss_components holds detached 0-d
            component tensors for logging, plus host-float per-channel
            ``multi_hot/<name>`` losses when ``binary_concept_names`` is set.
        """
        class_loss_value = _masked_cross_entropy(
            outputs,
            target_labels,
            weight=getattr(self.class_loss, "weight", None),
            ignore_index=self.ignore_index,
            label_smoothing=getattr(self.class_loss, "label_smoothing", 0.0),
            per_pixel_loss_weight=self.per_pixel_loss_weight,
            per_image_loss_weight=self.per_image_loss_weight,
        )

        loss_components: dict[str, torch.Tensor | float] = {
            "classification": class_loss_value.detach()
        }
        concept_loss_value = torch.tensor(
            0.0, device=concept_logits.device, dtype=concept_logits.dtype
        )

        for name, labels_slice, outputs_slice in iter_concept_slices(
            self.concept_value2id, concept_labels, concept_logits
        ):
            if name == "multi_hot":
                slice_loss = calculate_multi_hot_concept_loss(
                    outputs_slice,
                    labels_slice,
                    per_pixel_loss_weight=self.per_pixel_loss_weight,
                    per_image_loss_weight=self.per_image_loss_weight,
                )
                if self.binary_concept_names is not None:
                    loss_components.update(
                        binary_concept_losses(
                            outputs_slice.detach(),
                            labels_slice,
                            self.binary_concept_names,
                            per_pixel_loss_weight=self.per_pixel_loss_weight,
                            per_image_loss_weight=self.per_image_loss_weight,
                        )
                    )
            else:
                slice_loss = calculate_taxonomic_rank_loss(
                    outputs_slice,
                    labels_slice,
                    from_logits=True,
                    per_pixel_loss_weight=self.per_pixel_loss_weight,
                    per_image_loss_weight=self.per_image_loss_weight,
                    label_smoothing=self.taxonomic_label_smoothing,
                ) / len(TAXONOMIC_CONCEPTS)
            loss_components[name] = slice_loss.detach()
            concept_loss_value = concept_loss_value + slice_loss

        loss_components["concepts"] = concept_loss_value.detach()
        total_loss = class_loss_value + self.lambda_weight * concept_loss_value
        return total_loss, loss_components


LOSS_CLASSES: dict[str, type[ConceptBottleneckLoss]] = {
    "ConceptBottleneckLoss": ConceptBottleneckLoss,
}
