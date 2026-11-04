"""Expected taxonomic-distance loss over the per-rank softmax heads.

For each taxonomic rank ``r`` and every pixel labelled at that rank with class ``y``:

    L^(r) = sum_j p^(r)_j * D_r[y, j] / D_r,max,    p^(r) = softmax(logits_r)

i.e. the expected tree distance between the prediction and the label, normalised to ``[0, 1]``.
``D_r`` is :func:`~mermaidseg.dataset_reconciliation.taxonomy_tree.rank_distance_matrix`, the
matrix behind the hierarchical metrics' mistake severity: 0 for the same taxon, 1 for siblings, up
to ``depth + 1`` when the lineages never meet. ``none`` (not an organism) is at the maximum distance
from every other channel, so confusing an organism with substrate costs the most.

The gradient for logit ``j`` is ``p_j * (D[y, j] - L)``: mass moves away from classes farther than
the current expected distance, so far mistakes are pushed down harder than near ones. The loss is
0 only when all mass is on the label, and it adds no signal on pixels without a label at a rank.
"""

from __future__ import annotations

import torch

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.taxonomy_tree import TaxonomyTree, rank_distance_matrix
from mermaidseg.model.concept_metrics import reduce_masked_loss, taxonomic_target_and_mask


class TaxonomicDistanceLoss(torch.nn.Module):
    """Expected normalised tree distance per rank head, averaged over ranks like the per-rank CE.

        L = weight * sum_r L^(r) / num_ranks

    Args:
        taxonomy: Parent-child structure from ``build_taxonomy_tree`` over ``class_to_concepts.csv``,
            the same tree the hierarchical metrics use.
        concept_value2id: Concept name -> value -> channel mapping.
        weight: Weight of the term. Defaults to 1.
        per_pixel_loss_weight: Reduction, matching the concept loss's setting.
        per_image_loss_weight: Reduction, matching the concept loss's setting.
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        weight: float = 1.0,
        per_pixel_loss_weight: float = 0.0,
        per_image_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.per_pixel_loss_weight = per_pixel_loss_weight
        self.per_image_loss_weight = per_image_loss_weight
        self.num_ranks = len(TAXONOMIC_CONCEPTS)

        # (rank, channel offset, number of channels); taxonomic ranks come first in the logits.
        self.slices: list[tuple[str, int, int]] = []
        offset = 0
        for rank in TAXONOMIC_CONCEPTS:
            if rank not in concept_value2id:
                continue
            names = [v for v in concept_value2id[rank] if v != "not_given"]
            distance = torch.from_numpy(rank_distance_matrix(taxonomy, rank)).float()
            index = _tree_channel_index(taxonomy, rank, names)
            distance = distance[index][:, index]
            max_distance = taxonomy.ranks.index(rank) + 1
            self.register_buffer(f"distance_{rank}", distance / max_distance)
            self.slices.append((rank, offset, len(names)))
            offset += len(names)

    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted sum of the per-rank expected distances, plus per-rank logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            concept_labels: Concept targets ``(B, C, H, W)`` in the taxonomic 0/1/2 encoding.
        """
        loss_components: dict[str, float] = {}
        total = concept_logits.new_zeros(())
        for rank, offset, size in self.slices:
            target, valid = taxonomic_target_and_mask(concept_labels[:, offset : offset + size])
            loss_components[f"distance/{rank}"] = 0.0
            if not valid.any():
                continue
            logits = concept_logits[:, offset : offset + size].permute(0, 2, 3, 1)[valid]  # (P, C)
            distance_rows = getattr(self, f"distance_{rank}")[target[valid]]  # (P, C)
            pixel_loss = (logits.softmax(dim=1) * distance_rows).sum(dim=1)

            per_pixel = concept_logits.new_zeros(valid.shape)
            per_pixel[valid] = pixel_loss
            rank_loss = reduce_masked_loss(
                per_pixel,
                valid,
                per_pixel_loss_weight=self.per_pixel_loss_weight,
                per_image_loss_weight=self.per_image_loss_weight,
            )
            loss_components[f"distance/{rank}"] = rank_loss.item()
            total = total + rank_loss

        total = total / self.num_ranks
        loss_components["distance"] = total.item()
        return self.weight * total, loss_components


def _tree_channel_index(taxonomy: TaxonomyTree, rank: str, names: list[str]) -> list[int]:
    """Tree channel index of each model channel ``names`` at ``rank``, matched by class name."""
    index = taxonomy.index_of(rank)
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(
            f"Model channels at rank {rank!r} are missing from the taxonomy tree: {missing}. "
            "Build the tree from the same class_to_concepts.csv as the concept schema."
        )
    return [index[name] for name in names]
