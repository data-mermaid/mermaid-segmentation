"""Shared helpers for taxonomic and multi-hot concept loss and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def map_taxonomy_to_dense(binary_taxonomy: torch.Tensor) -> torch.Tensor:
    """Map taxonomic one-hot/binary labels to dense IDs.

    Maps binary/one-hot taxonomy labels to dense IDs where:
    - 0 represents not_given (all zeros)
    - 1+ represents the argmax index + 1 (concrete category)
    """
    sum_ = binary_taxonomy.sum(dim=1)
    is_not_given = sum_.eq(0)

    taxonomy_id = binary_taxonomy.argmax(dim=1) + 1
    return torch.where(
        is_not_given,
        torch.zeros_like(taxonomy_id),
        taxonomy_id,
    )


def map_taxonomy_predictions_to_dense(pred_taxonomy: torch.Tensor) -> torch.Tensor:
    """Map predicted taxonomic scores to dense IDs via argmax + 2."""
    return pred_taxonomy.argmax(dim=1) + 1


def taxonomic_target_and_mask(
    concept_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target class index and valid mask for one taxonomic rank slice.

    Taxonomic channels use 0=not_given, 1=inactive, 2=active (concrete class). Only pixels with at
    least one active (==2) channel contribute to loss/metrics.
    """
    active = concept_labels.eq(2)
    valid_mask = active.any(dim=1)
    target_idx = active.float().argmax(dim=1).long()
    return target_idx, valid_mask


def taxonomic_valid_mask(concept_labels: torch.Tensor) -> torch.Tensor:
    """Return a mask of pixels with a concrete taxonomic label."""
    return taxonomic_target_and_mask(concept_labels)[1]


def reduce_masked_loss(
    per_element: torch.Tensor,
    mask: torch.Tensor,
    *,
    per_pixel_loss_weight: float,
    per_image_loss_weight: float,
) -> torch.Tensor:
    """Blend a batch-wide per-pixel mean with a per-image mean of a masked loss.

    Densely and sparsely annotated images are handled by two complementary reductions that
    are combined additively:

    - The per-pixel term averages every valid element across the whole batch equally, so
      densely annotated images (with many valid pixels) dominate.
    - The per-image term first averages the valid elements within each image, then averages
      those per-image values across images that have at least one valid element, so every
      image contributes equally regardless of how densely it is annotated. Images with no
      valid elements are excluded rather than counted as zero loss.

    Args:
        per_element (torch.Tensor): Unreduced loss with shape ``(B, ...)``.
        mask (torch.Tensor): Boolean/float mask with the same shape as ``per_element``.
        per_pixel_loss_weight (float): Weight applied to the batch-wide per-pixel mean.
        per_image_loss_weight (float): Weight applied to the per-image mean.

    Returns:
        A scalar tensor equal to
        ``per_pixel_loss_weight * per_pixel_mean + per_image_loss_weight * per_image_mean``.
        Callers must ensure ``mask`` has at least one valid element.
    """
    mask = mask.to(per_element.dtype)
    masked = per_element * mask

    total_valid = mask.sum()
    per_pixel_mean = masked.sum() / total_valid

    batch_size = per_element.shape[0]
    per_image_sum = masked.reshape(batch_size, -1).sum(dim=1)
    per_image_count = mask.reshape(batch_size, -1).sum(dim=1)
    image_has_valid = per_image_count > 0
    per_image_mean_values = per_image_sum / per_image_count.clamp(min=1)
    per_image_mean = per_image_mean_values[image_has_valid].mean()

    return per_pixel_loss_weight * per_pixel_mean + per_image_loss_weight * per_image_mean


def calculate_taxonomic_rank_loss(
    concept_outputs: torch.Tensor,
    concept_labels: torch.Tensor,
    *,
    from_logits: bool = False,
    foreground_mask: torch.Tensor | None = None,
    per_pixel_loss_weight: float = 0.0,
    per_image_loss_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy / NLL on the active taxonomic class; masked where not_given or none."""
    num_channels = concept_outputs.size(1)
    if num_channels == 0:
        return torch.tensor(0.0, device=concept_outputs.device, dtype=concept_outputs.dtype)

    if concept_labels.size(1) != num_channels:
        raise ValueError(
            "Taxonomic label/output channel mismatch: "
            f"labels {concept_labels.size(1)} vs outputs {num_channels}"
        )

    target_idx, valid_mask = taxonomic_target_and_mask(concept_labels)
    if foreground_mask is not None:
        valid_mask = valid_mask & foreground_mask

    if not valid_mask.any():
        return torch.tensor(0.0, device=concept_outputs.device, dtype=concept_outputs.dtype)

    target_idx = target_idx.clamp(0, num_channels - 1)
    flat_target = target_idx.reshape(-1)

    if from_logits:
        flat_logits = concept_outputs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        per_pixel = F.cross_entropy(
            flat_logits, flat_target, reduction="none", label_smoothing=label_smoothing
        )
    else:
        log_probs = torch.log(concept_outputs.clamp(min=1e-6))
        flat_log_probs = log_probs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        per_pixel = F.nll_loss(flat_log_probs, flat_target, reduction="none")

    per_pixel = per_pixel.reshape(valid_mask.shape)
    return reduce_masked_loss(
        per_pixel,
        valid_mask,
        per_pixel_loss_weight=per_pixel_loss_weight,
        per_image_loss_weight=per_image_loss_weight,
    )


def calculate_multi_hot_concept_loss(
    concept_outputs: torch.Tensor,
    concept_labels: torch.Tensor,
    concept_loss: torch.nn.Module | None = None,
    *,
    from_logits: bool = False,
    foreground_mask: torch.Tensor | None = None,
    per_pixel_loss_weight: float = 0.0,
    per_image_loss_weight: float = 1.0,
) -> torch.Tensor:
    """BCE loss for multi-hot concepts with 0=invalid, 1=False, 2=True encoding.

    When ``from_logits`` is True, ``concept_outputs`` are raw logits scored with
    ``binary_cross_entropy_with_logits`` and ``concept_loss`` is unused. Otherwise
    ``concept_outputs`` are probabilities in ``[0, 1]`` and ``concept_loss`` (e.g. ``BCELoss``)
    is required.
    """
    concept_mask = concept_labels.gt(0)
    if foreground_mask is not None:
        concept_mask = concept_mask & foreground_mask.unsqueeze(1)

    if concept_mask.sum() == 0:
        return torch.tensor(0.0, device=concept_outputs.device, dtype=concept_outputs.dtype)

    targets = torch.clamp(concept_labels - 1, 0.01, 0.99)
    if from_logits:
        per_element_loss = F.binary_cross_entropy_with_logits(
            concept_outputs, targets, reduction="none"
        )
    else:
        if concept_loss is None:
            raise ValueError("concept_loss is required when from_logits is False")
        outputs = concept_outputs.clamp(1e-6, 1.0 - 1e-6)
        per_element_loss = concept_loss(outputs, targets)
    return reduce_masked_loss(
        per_element_loss,
        concept_mask.detach(),
        per_pixel_loss_weight=per_pixel_loss_weight,
        per_image_loss_weight=per_image_loss_weight,
    )
