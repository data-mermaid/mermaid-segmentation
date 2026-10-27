"""Hierarchical taxonomy metrics for the evaluation suite.

Feeds point or pixel concept outputs to
:class:`~mermaidseg.model.hierarchical_metrics.HierarchicalMetrics` (lineage consistency,
mistake severity, calibration ECE, rare-taxon accuracy). Scores are the model's activated
concepts, which are already a per-rank softmax, so no score transform is applied.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mermaidseg.dataset_reconciliation.taxonomy_tree import build_taxonomy_tree
from mermaidseg.evaluation.metrics import ConceptLayout
from mermaidseg.model.hierarchical_metrics import HierarchicalMetrics
from mermaidseg.model.rank_outputs import RankBatch


class HierarchicalScore:
    """Accumulates hierarchical metrics from ``(K, N)`` concept probabilities.

    Targets are per-rank dense ids ``(R, N)`` in :attr:`ranks` order: ``0`` is not_given and
    ``d >= 1`` is channel ``d - 1``. Build them from 0/1/2 concept rows with
    :meth:`dense_targets`.
    """

    def __init__(self, concept_names: list[str], taxonomy_csv: str | Path):
        layout = ConceptLayout.from_concept_names(concept_names)
        tree = build_taxonomy_tree(taxonomy_csv)
        self.ranks: tuple[str, ...] = tuple(rank for rank, _, _ in layout.taxo)
        self._slices: list[slice] = []
        for rank, indices, _ in layout.taxo:
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"{rank} concept channels are not contiguous in concept_names")
            values = [concept_names[i].split("__", 1)[1] for i in indices]
            if values != tree.channels[rank]:
                raise ValueError(
                    f"{rank} has {len(values)} model channels but {len(tree.channels[rank])} "
                    f"channels in {taxonomy_csv}, or they differ in order. The taxonomy CSV "
                    "must be the one the checkpoint was trained with."
                )
            self._slices.append(slice(indices[0], indices[-1] + 1))
        self.metrics = HierarchicalMetrics(tree, ranks=self.ranks)

    def dense_targets(self, concept_rows: torch.Tensor) -> torch.Tensor:
        """``(M, K)`` concept rows with 0/1/2 codes -> ``(R, M)`` dense ids."""
        targets = []
        for sl in self._slices:
            active = concept_rows[:, sl].eq(2)
            dense = active.int().argmax(dim=1) + 1
            targets.append(torch.where(active.any(dim=1), dense, torch.zeros_like(dense)))
        return torch.stack(targets).long()

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Add ``N`` points: ``probs`` ``(K, N)`` and ``targets`` ``(R, N)`` on one device."""
        n = probs.shape[1]
        batches: dict[str, RankBatch] = {}
        for rank, sl, target in zip(self.ranks, self._slices, targets, strict=True):
            scores = probs[sl].reshape(1, -1, 1, n)
            target = target.reshape(1, 1, n)
            batches[rank] = RankBatch(
                rank=rank,
                scores=scores,
                target=target,
                pred=scores.argmax(dim=1) + 1,
                valid=target.ne(0),
            )
        self.metrics.update(batches)

    def to_dict(self) -> dict[str, float]:
        return self.metrics.compute()
