"""Per-rank model outputs scored by the hierarchical metrics.

For each batch, :mod:`mermaidseg.evaluation.hierarchical` builds one :class:`RankBatch` per
taxonomic rank, with the pre-argmax scores alongside the dense targets and predictions, and
passes them to :class:`~mermaidseg.evaluation.hierarchical_metrics.HierarchicalMetrics`. The
metrics accumulate online rather than buffering: keeping the scores for a split would not fit
in memory (460 images at 256x256 with 367 genus channels is ~110 GB in fp32).

``target`` and ``pred`` use the dense ids of ``concept_metrics.map_taxonomy_to_dense``:
``0`` is not_given and ``d >= 1`` is channel ``d - 1``. ``TaxonomyTree.index_of`` is
0-based, so use :meth:`RankBatch.target_tree_index` and :meth:`RankBatch.pred_tree_index`
to convert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RankBatch:
    """One taxonomic rank's outputs and targets for a single batch.

    Attributes:
        rank: Rank name, e.g. ``"genus"``. Matches ``TAXONOMIC_CONCEPTS``.
        scores: Pre-argmax model outputs for this rank, shape ``(B, C_rank, H, W)``.
            The concept-bottleneck model applies a per-rank softmax, so these are
            probabilities.
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
