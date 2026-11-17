"""Taxonomy-aware loss over the per-rank softmax heads.

The loss takes its parent-child structure from the :class:`TaxonomyTree` the hierarchical
metrics use (``build_taxonomy_tree`` over ``class_to_concepts.csv``), so training and evaluation
share one taxonomy.

Label-free coherence loss (:class:`TaxonomicCoherenceLoss`): for each adjacent rank pair, the coarse
head's distribution ``p`` should equal the fine head's distribution summed per parent ``q``:

    L_KL^(r) = KL( stopgrad(p) || q ),    q = A_r @ softmax(logits_{r+1})
"""

from __future__ import annotations

import logging

import torch

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.taxonomy_tree import (
    NONE_TAXON,
    TaxonomyTree,
    build_rank_incidence,
)
from mermaidseg.model.concept_metrics import reduce_masked_loss

logger = logging.getLogger(__name__)


def taxonomic_rank_slices(
    concept_value2id: dict[str, dict[str, list[int]]],
) -> list[tuple[str, int, list[str]]]:
    """Return ``(rank, channel_offset, class_names)`` for each taxonomic rank, in order."""
    slices = []
    offset = 0
    for rank in TAXONOMIC_CONCEPTS:
        if rank not in concept_value2id:
            continue
        class_names = [v for v in concept_value2id[rank] if v != "not_given"]
        slices.append((rank, offset, class_names))
        offset += len(class_names)
    return slices


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


def build_parent_matrices(
    taxonomy: TaxonomyTree,
    concept_value2id: dict[str, dict[str, list[int]]],
) -> dict[tuple[int, int], torch.Tensor]:
    """Build ``A_{r->r+1}`` in ``{0, 1}^{C_r x C_{r+1}}`` for every adjacent pair of rank indices.

    ``A[i, j] = 1`` when class ``j`` at rank ``r + 1`` is a child of class ``i`` at rank ``r`` in
    ``taxonomy``, read with :func:`build_rank_incidence` like the hierarchical metrics. Rows and
    columns follow the model's channel order; classes are matched by name. Multi-parent taxa keep
    every parent under the tree's ``permissive`` policy.

    ``none`` (not an organism) has no tree edges. ``none -> none`` is added so the non-biological
    class stays coherent across ranks; ``none`` is still never a child of a real taxon, otherwise a
    coarse animal label (fish: animalia -> none) could be satisfied by predicting it.
    """
    slices = taxonomic_rank_slices(concept_value2id)
    matrices = {}
    for r in range(len(slices) - 1):
        (coarse, _, names_c), (fine, _, names_f) = slices[r], slices[r + 1]
        incidence = torch.from_numpy(build_rank_incidence(taxonomy, coarse, fine)).T.float()
        rows = _tree_channel_index(taxonomy, coarse, names_c)
        cols = _tree_channel_index(taxonomy, fine, names_f)
        parents = incidence[rows][:, cols].contiguous()
        if NONE_TAXON in names_c and NONE_TAXON in names_f:
            parents[names_c.index(NONE_TAXON), names_f.index(NONE_TAXON)] = 1.0
        matrices[(r, r + 1)] = parents
    return matrices


def grouped_log_softmax(logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Log of the softmax over dim 1 summed per group: ``log(sum_j groups[g, j] * softmax_j)``.

    Computed in log space as ``log(sum_group exp(z - m)) - log(sum_all exp(z - m))``; the shift
    ``m`` cancels, and one ``exp`` serves both sums.

    Args:
        logits: Tensor of shape ``(B, C, ...)``.
        groups: Non-negative group weights of shape ``(G, C)``; every group needs a non-zero weight.
    Returns:
        Tensor of shape ``(B, G, ...)``.
    """
    shifted = (logits - logits.amax(dim=1, keepdim=True).detach()).exp()
    grouped = torch.einsum("bc...,gc->bg...", shifted, groups.to(logits.dtype))
    # A group whose logits all sit ~88 below the max underflows in fp32; floor it instead of -inf.
    grouped = grouped.clamp(min=torch.finfo(grouped.dtype).tiny)
    return grouped.log() - shifted.sum(dim=1, keepdim=True).log()


class TaxonomicCoherenceLoss(torch.nn.Module):
    """Label-free KL coherence loss between adjacent rank heads.

    For each adjacent pair (coarse rank ``r``, fine rank ``r+1``) and each selected pixel:

        L_KL^(r) = sum_i p_i * (log p_i - log q_i),   p = softmax(coarse),  q = A_r @ softmax(fine)

    With ``stop_grad_coarse`` the coarse distribution is detached in its coarse role only, so a
    middle head still receives gradients as the fine side of the pair above it.

    The data taxonomy is not a clean tree, so each pair is restricted to coarse classes with at
    least one child and fine classes with at least one parent, and both softmaxes are renormalised
    over those classes. A fine class with several parents splits its mass equally between them.
    Every pixel is selected.
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        stop_grad_coarse: bool = True,
    ) -> None:
        super().__init__()
        self.stop_grad_coarse = stop_grad_coarse
        self.slices = taxonomic_rank_slices(concept_value2id)

        matrices = build_parent_matrices(taxonomy, concept_value2id)
        for r in range(len(self.slices) - 1):
            parents = matrices[(r, r + 1)]
            coarse_keep = parents.any(dim=1)
            fine_keep = parents.any(dim=0)
            self._warn_defects(r, parents, coarse_keep, fine_keep)
            groups = parents[coarse_keep][:, fine_keep]
            self.register_buffer(f"coarse_keep_{r}", coarse_keep.nonzero().flatten())
            self.register_buffer(f"fine_keep_{r}", fine_keep.nonzero().flatten())
            self.register_buffer(f"groups_{r}", groups / groups.sum(dim=0, keepdim=True))

    def _pair_name(self, r: int) -> str:
        return f"{self.slices[r][0]}->{self.slices[r + 1][0]}"

    def _warn_defects(
        self,
        r: int,
        parents: torch.Tensor,
        coarse_keep: torch.Tensor,
        fine_keep: torch.Tensor,
    ) -> None:
        coarse_names, fine_names = self.slices[r][2], self.slices[r + 1][2]
        defects = {
            "coarse classes without children (excluded)": [
                coarse_names[i] for i in (~coarse_keep).nonzero().flatten().tolist()
            ],
            "fine classes without parent (excluded)": [
                fine_names[j] for j in (~fine_keep).nonzero().flatten().tolist()
            ],
            "fine classes with several parents (mass split)": [
                fine_names[j] for j in (parents.sum(dim=0) > 1).nonzero().flatten().tolist()
            ],
        }
        for description, names in defects.items():
            if names:
                logger.warning("KL %s: %s: %s", self._pair_name(r), description, names)

    def _rank_logits(
        self,
        concept_logits: torch.Tensor,
        r: int,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits of rank ``r``, restricted to the ``keep`` classes (a copy only when needed)."""
        _, offset, names = self.slices[r]
        logits = concept_logits[:, offset : offset + len(names)]
        if keep is None or keep.numel() == len(names):
            return logits
        return logits[:, keep]

    def forward(
        self,
        concept_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of the adjacent-pair KL terms, plus per-pair logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            target_labels: Target-space labels ``(B, H, W)``; gives the pixel mask's shape.
        """
        mask = torch.ones_like(target_labels, dtype=torch.bool)

        loss_components: dict[str, float] = {}
        total = concept_logits.new_zeros(())
        for r in range(len(self.slices) - 1):
            if not mask.any():
                loss_components[f"kl/{self._pair_name(r)}"] = 0.0
                continue

            coarse = self._rank_logits(concept_logits, r, getattr(self, f"coarse_keep_{r}"))
            fine = self._rank_logits(concept_logits, r + 1, getattr(self, f"fine_keep_{r}"))
            log_p = coarse.log_softmax(dim=1)
            if self.stop_grad_coarse:
                log_p = log_p.detach()
            log_q = grouped_log_softmax(fine, getattr(self, f"groups_{r}"))
            kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)  # KL(p || q), shape (B, H, W)
            pair_loss = reduce_masked_loss(
                kl, mask, per_pixel_loss_weight=0.0, per_image_loss_weight=1.0
            )
            loss_components[f"kl/{self._pair_name(r)}"] = pair_loss.item()
            total = total + pair_loss
        return total, loss_components


class TaxonomicLoss(torch.nn.Module):
    """Taxonomy-aware term added on top of the per-rank cross-entropy of the concept loss.

        L = lambda * sum_r L_KL^(r) / num_ranks

    The term is divided by the number of taxonomic ranks like the per-rank cross-entropy in
    :class:`~mermaidseg.model.loss.ConceptBottleneckLoss`, so its weight is on the same scale.
    The weight is constant from the first update; the LR warmup and decay are the only schedule.
    The term is only built when its weight is positive.

    Args:
        taxonomy: Parent-child structure from ``build_taxonomy_tree`` over ``class_to_concepts.csv``,
            the same tree the hierarchical metrics use.
        concept_value2id: Concept name -> value -> channel mapping.
        kl_weight: Weight ``lambda`` of :class:`TaxonomicCoherenceLoss`. Defaults to 0 (off).
        kl_stop_grad_coarse: Detach the coarse head's distribution in each KL pair.
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        kl_weight: float = 0.0,
        kl_stop_grad_coarse: bool = True,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.num_ranks = len(TAXONOMIC_CONCEPTS)

        self.coherence_loss = None
        if kl_weight > 0:
            self.coherence_loss = TaxonomicCoherenceLoss(
                taxonomy,
                concept_value2id,
                stop_grad_coarse=kl_stop_grad_coarse,
            )

    def forward(
        self,
        concept_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted taxonomic term, plus its logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            target_labels: Target-space labels ``(B, H, W)``; give the KL pixel mask's shape.
        """
        loss_components: dict[str, float] = {}
        total = concept_logits.new_zeros(())

        if self.coherence_loss is not None:
            kl_value, kl_components = self.coherence_loss(concept_logits, target_labels)
            kl_value = kl_value / self.num_ranks
            loss_components.update(kl_components)
            loss_components["kl"] = kl_value.item()
            total = total + self.kl_weight * kl_value

        return total, loss_components
