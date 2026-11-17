"""Hierarchical taxonomy metrics for the evaluation suite.

Feeds point or pixel concept outputs to
:class:`~mermaidseg.model.hierarchical_metrics.HierarchicalMetrics` (lineage consistency,
mistake severity, calibration ECE, rare-taxon accuracy). Scores are the model's activated
concepts, which are already a per-rank softmax, so no score transform is applied.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch

from mermaidseg.dataset_reconciliation.concept_ranks import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.concept_removal import apply_concept_removal, concept_kind
from mermaidseg.dataset_reconciliation.taxonomy_tree import (
    TaxonomyTree,
    build_taxonomy_tree,
    load_taxonomy_frame,
)
from mermaidseg.evaluation.metrics import ConceptLayout
from mermaidseg.model.hierarchical_metrics import HierarchicalMetrics
from mermaidseg.model.rank_outputs import RankBatch

logger = logging.getLogger(__name__)

#: Headline keys of ``HierarchicalMetrics.compute()`` reported per unit and in macro means.
HIER_HEADLINE_KEYS: tuple[str, ...] = (
    "consistency/overall",
    *(
        f"consistency/{c}->{f}"
        for c, f in zip(TAXONOMIC_CONCEPTS, TAXONOMIC_CONCEPTS[1:], strict=False)
    ),
    *(f"severity/{rank}" for rank in TAXONOMIC_CONCEPTS),
    "severity/mean_normalized",
    *(f"ece/{rank}" for rank in TAXONOMIC_CONCEPTS),
    "ece/mean",
)


def _run_taxonomy_tree(taxonomy_csv: str | Path, removed_concepts: Iterable[str]) -> TaxonomyTree:
    """Taxonomy tree of ``taxonomy_csv`` without the run's removed taxonomic channels.

    Training set removed ``rank__value`` channels to ``not_given`` (``apply_concept_removal``);
    removed binary concepts do not affect the tree.
    """
    df = load_taxonomy_frame(taxonomy_csv)
    removed = [c for c in dict.fromkeys(removed_concepts) if concept_kind(c) != "binary"]
    if removed:
        logger.info(
            "hierarchical metrics: %d taxonomic channels removed from %s (concept removal)",
            len(removed),
            taxonomy_csv,
        )
        df = apply_concept_removal(df, removed)
    return build_taxonomy_tree(df=df)


class HierarchicalScore:
    """Accumulates hierarchical metrics from ``(K, N)`` concept probabilities.

    Targets are per-rank dense ids ``(R, N)`` in :attr:`ranks` order: ``0`` is not_given and
    ``d >= 1`` is channel ``d - 1``. Build them from 0/1/2 concept rows with
    :meth:`dense_targets`.
    """

    def __init__(
        self,
        concept_names: list[str],
        taxonomy_csv: str | Path,
        removed_concepts: Iterable[str] = (),
    ):
        """Validate the taxonomy against the model channels and build the metrics.

        Args:
            concept_names: Model concept channel names, in channel order.
            taxonomy_csv: Taxonomy CSV the checkpoint was trained with.
            removed_concepts: Concepts removed from the taxonomy during training.

        Raises:
            ValueError: If a rank's channels are not contiguous or differ from the taxonomy.
        """
        layout = ConceptLayout.from_concept_names(concept_names)
        tree_removed = list(removed_concepts)
        tree = _run_taxonomy_tree(taxonomy_csv, tree_removed)
        self.ranks: tuple[str, ...] = tuple(rank for rank, _, _ in layout.taxo)
        self._slices: list[slice] = []
        for rank, indices, _ in layout.taxo:
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError(f"{rank} concept channels are not contiguous in concept_names")
            values = [concept_names[i].split("__", 1)[1] for i in indices]
            if values != tree.channels[rank]:
                raise ValueError(
                    f"{rank} has {len(values)} model channels but {len(tree.channels[rank])} "
                    f"channels in {taxonomy_csv} after removing {len(tree_removed)} concepts, "
                    "or they differ in order. The taxonomy CSV and the concept removal "
                    "(training.concept_removal of --training-config) must be the ones the "
                    "checkpoint was trained with."
                )
            self._slices.append(slice(indices[0], indices[-1] + 1))
        self.tree = tree
        self.metrics = HierarchicalMetrics(tree, ranks=self.ranks)

    def fresh(self) -> HierarchicalScore:
        """Empty accumulator with this one's validated tree and channel slices."""
        out = object.__new__(HierarchicalScore)
        out.ranks = self.ranks
        out._slices = list(self._slices)
        out.tree = self.tree
        out.metrics = HierarchicalMetrics(self.tree, ranks=self.ranks)
        src = self.metrics
        if src._device is not None:
            # Read-only tree lookups; sharing them skips rebuilding per accumulator.
            out.metrics._device = src._device
            out.metrics._incidence = src._incidence
            out.metrics._distance = src._distance
            out.metrics._taxon_mask = src._taxon_mask
            out.metrics._taxon_or_none_mask = src._taxon_or_none_mask
        return out

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
        """Add ``N`` points: ``probs`` ``(K, N)`` and ``targets`` ``(R, N)``, on one device."""
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
        """Return the accumulated hierarchical metrics."""
        return self.metrics.compute()
