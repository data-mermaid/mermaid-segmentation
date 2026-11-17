"""Per-rank model outputs passed from ``Evaluator`` to rank-output consumers.

For each batch, ``Evaluator`` hands every registered :class:`RankOutputConsumer` one
:class:`RankBatch` per taxonomic rank, with the pre-argmax scores alongside the dense
targets and predictions. Consumers accumulate online rather than buffering: keeping the
scores for a split would not fit in memory (460 images at 256x256 with 367 genus channels
is ~110 GB in fp32).

``target`` and ``pred`` use the dense ids of ``concept_metrics.map_taxonomy_to_dense``:
``0`` is not_given and ``d >= 1`` is channel ``d - 1``. ``TaxonomyTree.index_of`` is
0-based, so use :meth:`RankBatch.target_tree_index` and :meth:`RankBatch.pred_tree_index`
to convert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class RankBatch:
    """One taxonomic rank's outputs and targets for a single batch.

    Attributes:
        rank: Rank name, e.g. ``"genus"``. Matches ``TAXONOMIC_CONCEPTS``.
        scores: Pre-argmax model outputs for this rank, shape ``(B, C_rank, H, W)``.
            These are whatever the model emits (probabilities or logits, depending on
            ``training_kwargs``); consumers that need normalised probabilities should
            say so explicitly rather than assume.
        target: Dense ground-truth ids, shape ``(B, H, W)``. ``0`` means not_given.
        pred: Dense predicted ids, shape ``(B, H, W)``. Always ``>= 1`` -- an argmax
            always selects a channel, so the model never predicts "not_given".
        valid: Boolean mask, shape ``(B, H, W)``, true where ``target != 0``. Metrics
            must restrict to these pixels; elsewhere there is no ground truth to score.
    """

    rank: str
    scores: torch.Tensor
    target: torch.Tensor
    pred: torch.Tensor
    valid: torch.Tensor

    @property
    def num_channels(self) -> int:
        """Number of channels (concrete taxa) at this rank."""
        return int(self.scores.shape[1])

    def target_tree_index(self) -> torch.Tensor:
        """Ground-truth ids as 0-based tree/channel indices (invalid pixels give -1)."""
        return torch.where(self.valid, self.target - 1, torch.full_like(self.target, -1))

    def pred_tree_index(self) -> torch.Tensor:
        """Predicted ids as 0-based tree/channel indices."""
        return self.pred - 1


@runtime_checkable
class RankOutputConsumer(Protocol):
    """Something that accumulates per-rank outputs across an evaluation run.

    Implementations are expected to be stateful accumulators: ``update`` is called once
    per batch with every rank present, ``compute`` returns the finished metrics, and
    ``reset`` clears state so the consumer can be reused for another split.
    """

    def update(self, ranks: dict[str, RankBatch]) -> None:
        """Accumulate one batch.

        Args:
            ranks: All taxonomic ranks for this batch, keyed by rank name. Passed together
                so cross-rank metrics such as consistency see the joint assignment.
        """
        ...

    def compute(self) -> dict[str, float]:
        """Return accumulated metrics as ``name -> value``."""
        ...

    def reset(self) -> None:
        """Clear accumulated state."""
        ...
