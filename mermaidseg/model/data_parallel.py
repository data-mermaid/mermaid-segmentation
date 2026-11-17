"""Per-rank label lookup, forward, and concept-bottleneck loss for DDP training.

Each rank runs the label lookup, the forward, and the concept/taxonomy loss on
its own shard (:class:`ShardStep`), wrapped for DDP by :class:`DDPLossStep`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from mermaidseg.dataset_reconciliation.concept_source_mask import apply_concept_source_mask
from mermaidseg.dataset_reconciliation.concepts import source_labels_to_concepts
from mermaidseg.dataset_reconciliation.label_mapping import source_labels_to_target_labels

DATA_PARALLEL_GPUS = 4


def require_divisible_batch(batch_size: int, n_gpus: int = DATA_PARALLEL_GPUS) -> int:
    """Return images per GPU, or raise if ``batch_size`` does not split evenly."""
    if int(batch_size) % int(n_gpus) != 0:
        raise RuntimeError(
            f"data_parallel batch size {batch_size} is not divisible by {n_gpus} GPUs."
        )
    return int(batch_size) // int(n_gpus)


def apply_concept_bottleneck_loss(
    segmentation_outputs: Any,
    loss_fn: nn.Module,
    taxonomic_loss: nn.Module | None,
    target_labels: torch.Tensor,
    target_concepts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute concept and taxonomy loss terms for one concept-bottleneck batch.

    This is the loss body of ``MetaModel.batch_predict_loss``. The single-GPU
    path and each distributed shard both call it, outside autocast, on
    fp32 logits.

    Args:
        segmentation_outputs: Model output with ``logits``, ``concept_logits``,
            and ``hidden_states``.
        loss_fn: Loss returning ``(loss, components)``.
        taxonomic_loss: Optional taxonomy loss added to the total.
        target_labels: Target label map.
        target_concepts: Target concept map.

    Returns:
        The total loss, fp32 logits, detached concept hidden states, and the
        loss components for logging.
    """
    concept_outputs = segmentation_outputs.hidden_states.detach()
    concept_logits = segmentation_outputs.concept_logits.float()
    outputs = segmentation_outputs.logits.float()
    loss, loss_components = loss_fn(outputs, target_labels, concept_logits, target_concepts)
    if taxonomic_loss is not None:
        taxonomic_value, taxonomic_components = taxonomic_loss(concept_logits, target_labels)
        loss = loss + taxonomic_value
        loss_components.update(taxonomic_components)
    return loss, outputs, concept_outputs, loss_components


def shard_targets(
    source_labels: torch.Tensor,
    target_lookup: torch.Tensor | None,
    concept_lookup: torch.Tensor,
    concept_keep: torch.Tensor | None = None,
    source_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target labels and concept maps for one shard on that shard's device.

    Same lookups as ``MetaModel._to_target_labels`` and ``_to_concept_labels``.
    A ``None`` target lookup is the identity, as in ``MetaModel``. With ``concept_keep`` and
    ``source_rows``, masked source x concept targets are set to 0 (``training.concept_source_mask``).

    Args:
        source_labels: Source label map.
        target_lookup: Source-to-target label lookup, or ``None`` for identity.
        concept_lookup: Source-to-concept lookup.
        concept_keep: Optional source x concept keep mask.
        source_rows: Per-image source rows, required with ``concept_keep``.

    Returns:
        The target labels and the target concepts.

    Raises:
        RuntimeError: If ``concept_keep`` is set and ``source_rows`` is ``None``.
    """
    if target_lookup is None:
        target_labels = source_labels
    else:
        target_labels = source_labels_to_target_labels(source_labels, target_lookup)
    target_concepts = source_labels_to_concepts(source_labels, concept_lookup)
    if concept_keep is not None:
        if source_rows is None:
            raise RuntimeError("concept_source_mask is set but the batch has no source rows.")
        target_concepts = apply_concept_source_mask(target_concepts, concept_keep, source_rows)
    return target_labels, target_concepts


@dataclass
class MetricShard:
    """Detached predictions for one GPU. Maps stay on that GPU until metrics."""

    outputs: torch.Tensor | None
    concept_outputs: torch.Tensor | None
    target_labels: torch.Tensor | None
    target_concepts: torch.Tensor | None


class ShardStep(nn.Module):
    """Label lookup, forward, and concept-bottleneck loss for one GPU's shard.

    The step's children are ``MetaModel``'s own model and losses, and its
    lookup buffers share storage with ``MetaModel``'s lookups. Registering
    them here does not re-parent ``MetaModel.model``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        taxonomic_loss: nn.Module | None,
        target_lookup: torch.Tensor | None,
        concept_lookup: torch.Tensor,
        use_amp: bool,
        amp_dtype: torch.dtype,
        concept_keep: torch.Tensor | None = None,
    ) -> None:
        """Register the model, losses, and lookup buffers.

        Args:
            model: Segmentation model.
            loss_fn: Concept-bottleneck loss.
            taxonomic_loss: Optional taxonomy loss.
            target_lookup: Source-to-target label lookup, or ``None`` for identity.
            concept_lookup: Source-to-concept lookup.
            use_amp: Whether the forward runs under autocast.
            amp_dtype: Autocast dtype.
            concept_keep: Optional source x concept keep mask.
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.taxonomic_loss = taxonomic_loss
        self.register_buffer("target_lookup", target_lookup, persistent=False)
        self.register_buffer("concept_lookup", concept_lookup, persistent=False)
        self.register_buffer("concept_keep", concept_keep, persistent=False)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def forward(
        self,
        images: torch.Tensor,
        source_labels: torch.Tensor,
        return_maps: bool = True,
        source_rows: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, float],
    ]:
        """Run lookup, forward, and loss on this shard.

        Args:
            images: Image batch for this shard.
            source_labels: Source label map for this shard.
            return_maps: Whether to return the detached prediction and target maps.
            source_rows: Per-image source rows for the concept-source mask.

        Returns:
            The loss, then outputs, concept outputs, target labels, and target
            concepts (all ``None`` without ``return_maps``), then the loss components.
        """
        target_labels, target_concepts = shard_targets(
            source_labels,
            cast("torch.Tensor | None", self.target_lookup),
            cast("torch.Tensor", self.concept_lookup),
            cast("torch.Tensor | None", self.concept_keep),
            source_rows,
        )
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            segmentation_outputs = self.model(images)
        loss, outputs, concept_outputs, components = apply_concept_bottleneck_loss(
            segmentation_outputs,
            self.loss_fn,
            self.taxonomic_loss,
            target_labels,
            target_concepts,
        )
        if not return_maps:
            return loss, None, None, None, None, components
        # Detach before returning so metric maps do not keep the autograd graph
        # after the scalar loss is dropped. Storage is shared with the graph
        # until that loss is freed.
        return (
            loss,
            outputs.detach(),
            concept_outputs.detach(),
            target_labels,
            target_concepts,
            components,
        )


class DDPLossStep(nn.Module):
    """DDP wrapper that returns only the scalar loss.

    ``ShardStep.forward`` also returns detached maps, label tensors, and a
    logging dict. DDP treats every tensor in that tuple as a graph output, so
    the unused-parameter reducer never finishes and the next iteration raises
    ``Expected to have finished reduction in the prior iteration``.
    """

    def __init__(self, step: ShardStep) -> None:
        """Wrap a shard step.

        Args:
            step: Step whose last result is stored on ``last``.
        """
        super().__init__()
        self.step = step
        self.last: (
            tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                dict[str, float],
            ]
            | None
        ) = None

    def forward(
        self,
        images: torch.Tensor,
        source_labels: torch.Tensor,
        return_maps: bool = True,
        source_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the shard step, keep its full result on ``last``, and return the loss.

        Args:
            images: Image batch for this shard.
            source_labels: Source label map for this shard.
            return_maps: Whether the step returns detached maps.
            source_rows: Per-image source rows for the concept-source mask.

        Returns:
            The scalar loss.
        """
        self.last = self.step(
            images, source_labels, return_maps=bool(return_maps), source_rows=source_rows
        )
        return self.last[0]
