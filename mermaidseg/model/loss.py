from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from mermaidseg.dataset_reconciliation.concepts import (
    TAXONOMIC_CONCEPTS,
    initialize_benthic_hierarchy,
)
from mermaidseg.model.concept_metrics import (
    calculate_multi_hot_concept_loss,
    calculate_taxonomic_rank_loss,
)
from mermaidseg.model.hierarchy_loss import build_distance_matrix, build_level_remaps


def iter_concept_slices(
    concept_value2id: dict[str, dict[str, int]],
    concept_labels: torch.Tensor,
    concept_outputs: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (name, labels_slice, outputs_slice) for each taxonomic rank and the binary
    tail."""
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


def _masked_cross_entropy(
    outputs: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    ignore_index: int,
    label_smoothing: float,
    damping_denominator: float,
) -> torch.Tensor:
    valid_mask = target_labels != ignore_index
    if not valid_mask.any():
        return outputs.sum() * 0.0

    per_pixel = F.cross_entropy(
        outputs,
        target_labels,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    valid = per_pixel[valid_mask]
    return valid.sum() / (valid.numel() + damping_denominator)


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """CrossEntropyLoss is a wrapper of `torch.nn.CrossEntropyLoss` that allows for
    additional customization.

    Attributes:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            This is useful for masking certain values in the target tensor. Defaults to -1.
        kwargs: Additional keyword arguments that are passed to the base `torch.nn.CrossEntropyLoss` class.
    """

    def __init__(
        self, ignore_index: int = 0, damping_denominator: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.damping_denominator = damping_denominator

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = _masked_cross_entropy(
            outputs,
            target_labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            damping_denominator=self.damping_denominator,
        )
        return loss, {"classification": loss.item()}


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """BCE loss for concept prediction that masks background pixels before averaging.

    Wraps `torch.nn.BCEWithLogitsLoss` with `reduction="none"` and applies a foreground
    mask derived from `labels` so background pixels (label == 0) do not contribute to
    the mean.
    """

    def __init__(
        self,
        reduction: str = "none",
        concept_value2id: dict[str, dict[str, int]] | None = None,
        damping_denominator: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.concept_value2id = concept_value2id
        self.damping_denominator = damping_denominator

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
                damping_denominator=self.damping_denominator,
            )
        return calculate_taxonomic_rank_loss(
            concept_outputs,
            concept_labels,
            from_logits=True,
            foreground_mask=foreground_mask,
            damping_denominator=self.damping_denominator,
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
                damping_denominator=self.damping_denominator,
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
    """ConceptBottleneckLoss combines a classification loss with a concept prediction
    loss.

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
        damping_denominator: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.class_loss = class_loss(ignore_index=ignore_index, **kwargs)
        self.concept_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight
        self.concept_value2id = concept_value2id
        self.damping_denominator = damping_denominator

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the total loss as the sum of the classification loss and a weighted
        concept loss.

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
        class_loss_value = _masked_cross_entropy(
            outputs,
            target_labels,
            weight=getattr(self.class_loss, "weight", None),
            ignore_index=self.ignore_index,
            label_smoothing=getattr(self.class_loss, "label_smoothing", 0.0),
            damping_denominator=self.damping_denominator,
        )
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
                    outputs_slice,
                    labels_slice,
                    self.concept_loss,
                    from_logits=True,
                    damping_denominator=self.damping_denominator,
                )
            else:
                slice_loss = calculate_taxonomic_rank_loss(
                    outputs_slice,
                    labels_slice,
                    from_logits=True,
                    damping_denominator=self.damping_denominator,
                ) / len(TAXONOMIC_CONCEPTS)
            loss_components[name] = slice_loss.item()
            concept_loss_value = concept_loss_value + slice_loss

        loss_components["concepts"] = concept_loss_value.item()
        total_loss = class_loss_value + self.lambda_weight * concept_loss_value
        return total_loss, loss_components


def _tree_distance_loss(
    outputs: torch.Tensor,
    target_labels: torch.Tensor,
    distance_matrix: torch.Tensor,
    *,
    ignore_index: int,
    damping_denominator: float,
) -> torch.Tensor:
    """Expected tree distance under softmax: mean_valid Σ_c p_c d(y, c)."""
    valid_mask = target_labels != ignore_index
    if not valid_mask.any():
        return outputs.sum() * 0.0

    num_classes = outputs.size(1)
    if distance_matrix.size(0) != num_classes or distance_matrix.size(1) != num_classes:
        raise ValueError(
            "distance_matrix shape mismatch: "
            f"{tuple(distance_matrix.shape)} vs num_classes={num_classes}"
        )

    probs = F.softmax(outputs, dim=1)
    # (B, H, W, C) distances for each GT class
    flat_targets = target_labels.clamp(0, num_classes - 1)
    dist_rows = distance_matrix[flat_targets]  # (B, H, W, C)
    expected = (probs.permute(0, 2, 3, 1) * dist_rows).sum(dim=-1)
    valid = expected[valid_mask]
    return valid.sum() / (valid.numel() + damping_denominator)


def _level_cross_entropy(
    outputs: torch.Tensor,
    target_labels: torch.Tensor,
    level_remap: torch.Tensor,
    *,
    ignore_index: int,
    damping_denominator: float,
) -> torch.Tensor:
    """Binary in-subtree vs out-of-subtree CE for one hierarchy level.

    ``level_remap`` maps a class id to the level's ancestor id when the class rolls up
    to that level, else ``ignore_index``; the level's *members* are therefore its non-
    ignore entries. We frame a 2-way problem — an "in" logit (``logsumexp`` over member
    channels) against an "out" logit (``logsumexp`` over the remaining foreground
    channels) — and classify each valid pixel as in-subtree (1) or not (0).

    The explicit "out" bucket is what makes this non-degenerate. A previous version
    filled a single coarse channel and left the rest at ``-inf``, so every valid pixel
    mapped to one class and the cross-entropy — and its gradient — was identically zero
    (the ``beta`` term did nothing).
    """
    valid_mask = target_labels != ignore_index
    members = (level_remap != ignore_index).nonzero(as_tuple=False).view(-1)
    if members.numel() == 0 or not valid_mask.any():
        return outputs.sum() * 0.0

    num_classes = outputs.size(1)
    member_mask = torch.zeros(num_classes, dtype=torch.bool, device=outputs.device)
    member_mask[members] = True
    member_mask[ignore_index] = False
    out_mask = ~member_mask
    out_mask[ignore_index] = False
    if not out_mask.any():
        # Level spans every foreground class — no negative bucket, nothing to separate.
        return outputs.sum() * 0.0

    in_logit = torch.logsumexp(outputs[:, member_mask], dim=1)  # (B, H, W)
    out_logit = torch.logsumexp(outputs[:, out_mask], dim=1)  # (B, H, W)
    binary_logits = torch.stack([out_logit, in_logit], dim=1)  # (B, 2, H, W)
    binary_target = member_mask.long()[target_labels.clamp(0, num_classes - 1)]  # (B, H, W)

    per_pixel = F.cross_entropy(binary_logits, binary_target, reduction="none")
    valid = per_pixel[valid_mask]
    return valid.sum() / (valid.numel() + damping_denominator)


class TaxonomicalLoss(torch.nn.Module):
    """Hierarchy-aware class loss for standard-mode linear / LoRA baselines.

    ``L = L_CE + alpha * L_tree + beta * L_levels`` where ``L_tree`` is the expected
    tree distance under the softmax and ``L_levels`` is multi-level CE on rolled-up
    ancestors present in ``id2label``. Growth forms are not part of this tree.
    """

    def __init__(
        self,
        id2label: Mapping[int, str],
        hierarchy: Mapping[str, str | None] | None = None,
        *,
        ignore_index: int = 0,
        damping_denominator: float = 0.0,
        alpha: float = 0.1,
        beta: float = 0.1,
        level_names: Sequence[str] | None = None,
        label_smoothing: float = 0.0,
        weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs  # YAML may forward unused CrossEntropyLoss keys
        self.ignore_index = ignore_index
        self.damping_denominator = damping_denominator
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("_class_weight", weight)
        else:
            self._class_weight = None

        resolved_hierarchy = (
            dict(hierarchy) if hierarchy is not None else initialize_benthic_hierarchy()
        )
        self.id2label = {int(k): str(v) for k, v in id2label.items()}
        num_classes = max(self.id2label.keys(), default=-1) + 1
        distance = build_distance_matrix(
            self.id2label,
            resolved_hierarchy,
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        self.register_buffer("distance_matrix", distance)

        remaps = build_level_remaps(
            self.id2label,
            resolved_hierarchy,
            level_names,
            ignore_index=ignore_index,
            num_classes=num_classes,
        )
        self._level_names = list(remaps.keys())
        for name, remap in remaps.items():
            # Buffer names must be identifiers; keep display name separately.
            buffer_key = f"level_remap__{name.replace(' ', '_').replace('-', '_')}"
            self.register_buffer(buffer_key, remap)
        self._level_buffer_keys = {
            name: f"level_remap__{name.replace(' ', '_').replace('-', '_')}"
            for name in self._level_names
        }

    def _level_remap(self, level_name: str) -> torch.Tensor:
        return getattr(self, self._level_buffer_keys[level_name])

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = outputs.float()
        class_loss = _masked_cross_entropy(
            outputs,
            target_labels,
            weight=self._class_weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            damping_denominator=self.damping_denominator,
        )
        components: dict[str, float] = {"classification": class_loss.item()}
        total = class_loss

        if self.alpha != 0.0:
            tree_loss = _tree_distance_loss(
                outputs,
                target_labels,
                # ``.to`` is a no-op when already on-device; guards the case where the loss
                # module was not moved to the training device (buffers default to CPU).
                self.distance_matrix.to(outputs.device),
                ignore_index=self.ignore_index,
                damping_denominator=self.damping_denominator,
            )
            components["tree_distance"] = tree_loss.item()
            total = total + self.alpha * tree_loss
        else:
            components["tree_distance"] = 0.0

        level_total = outputs.new_zeros(())
        if self.beta != 0.0 and self._level_names:
            for level_name in self._level_names:
                level_loss = _level_cross_entropy(
                    outputs,
                    target_labels,
                    self._level_remap(level_name).to(outputs.device),
                    ignore_index=self.ignore_index,
                    damping_denominator=self.damping_denominator,
                )
                components[f"level/{level_name}"] = level_loss.item()
                level_total = level_total + level_loss
            level_total = level_total / len(self._level_names)
            total = total + self.beta * level_total
        components["levels"] = float(level_total.item()) if torch.is_tensor(level_total) else 0.0
        components["taxonomical_total"] = total.item()
        return total, components


class DualTaxonomicalLoss(torch.nn.Module):
    """Taxonomical class loss plus masked morphology BCE for the dual-head model."""

    def __init__(
        self,
        id2label: Mapping[int, str],
        hierarchy: Mapping[str, str | None] | None = None,
        *,
        ignore_index: int = 0,
        damping_denominator: float = 0.0,
        alpha: float = 0.1,
        beta: float = 0.1,
        gamma: float = 0.1,
        level_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.damping_denominator = damping_denominator
        self.class_loss = TaxonomicalLoss(
            id2label,
            hierarchy,
            ignore_index=ignore_index,
            damping_denominator=damping_denominator,
            alpha=alpha,
            beta=beta,
            level_names=level_names,
            **kwargs,
        )
        self._morph_bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        outputs: torch.Tensor,
        target_labels: torch.Tensor,
        morphology_logits: torch.Tensor | None = None,
        morphology_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, components = self.class_loss(outputs, target_labels)
        if morphology_logits is None or morphology_targets is None or self.gamma == 0.0:
            components["morphology"] = 0.0
            components["taxonomical_total"] = total.item()
            return total, components

        morph_loss = calculate_multi_hot_concept_loss(
            morphology_logits.float(),
            morphology_targets,
            self._morph_bce,
            from_logits=True,
            foreground_mask=target_labels != self.ignore_index,
            damping_denominator=self.damping_denominator,
        )
        components["morphology"] = morph_loss.item()
        total = total + self.gamma * morph_loss
        components["taxonomical_total"] = total.item()
        return total, components
