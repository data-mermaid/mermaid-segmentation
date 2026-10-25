"""Taxonomy-aware losses over the per-rank softmax heads.

Marginal partial-label loss (:class:`TaxonomicPartialLoss`): for each adjacent rank pair
``d -> r = d + 1``, every pixel labelled at rank ``d`` with class ``c`` gives head ``r`` the constraint
that its probability mass lies inside the children of ``c``:

    L^(d->r) = -log( sum_{j in desc_r(c)} p^(r)_j )
             = logsumexp(logits_r) - logsumexp(logits_r[desc_r(c)])

i.e. cross-entropy between head ``r``'s predictions summed up to rank ``d`` and the rank-``d`` label.
A pixel labelled down to family gets kingdom->phylum, ..., order->family and family->genus terms.

Both losses take their parent-child structure from the :class:`TaxonomyTree` the hierarchical
metrics use (``build_taxonomy_tree`` over ``class_to_concepts.csv``), so training and evaluation
share one taxonomy.

Label-free coherence loss (:class:`TaxonomicCoherenceLoss`): for each adjacent rank pair, the coarse
head's distribution ``p`` should equal the fine head's distribution summed per parent ``q``:

    L_KL^(r) = KL( stopgrad(p) || q ),    q = A_r @ softmax(logits_{r+1})
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.taxonomy_tree import (
    NONE_TAXON,
    TaxonomyTree,
    build_rank_incidence,
)
from mermaidseg.model.concept_metrics import reduce_masked_loss, taxonomic_target_and_mask

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


def rank_targets(
    concept_labels: torch.Tensor,
    slices: list[tuple[str, int, list[str]]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Per-rank class targets and valid masks, and the deepest labelled rank per pixel.

    ``concept_labels`` has shape ``(B, C, ...)``. The returned depth is the index into ``slices``
    of the deepest labelled rank, or ``-1`` for void pixels with no taxonomic label at all.
    """
    targets, valid = [], []
    for _, offset, names in slices:
        target_idx, is_valid = taxonomic_target_and_mask(
            concept_labels[:, offset : offset + len(names)]
        )
        targets.append(target_idx)
        valid.append(is_valid)
    stacked = torch.stack(valid)
    rank_ids = torch.arange(len(slices), device=stacked.device)
    rank_ids = rank_ids.view(-1, *[1] * (stacked.ndim - 1))
    depth = torch.where(stacked, rank_ids, -1).amax(dim=0)
    return targets, valid, depth


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


def boundary_band(target_labels: torch.Tensor, width: int) -> torch.Tensor:
    """Pixels within ``width`` px of two different annotated (non-zero) labels."""
    annotated = target_labels > 0
    labels = target_labels.float()
    kernel, pad = 2 * width + 1, width
    highest = F.max_pool2d(
        torch.where(annotated, labels, -1.0)[:, None], kernel, stride=1, padding=pad
    )
    lowest = -F.max_pool2d(
        torch.where(annotated, -labels, -float("inf"))[:, None], kernel, stride=1, padding=pad
    )
    return (highest[:, 0] >= 0) & (highest[:, 0] != lowest[:, 0])


class TaxonomicPartialLoss(torch.nn.Module):
    """Marginal partial-label cross-entropy over adjacent ``(r - 1) -> r`` rank pairs.

    Every pixel labelled at rank ``r - 1`` selects its children at rank ``r`` with a row lookup in
    ``A_{r-1->r}``, independent of whether rank ``r`` is labelled. Each pair is reduced and logged
    separately.

    Optionally mixes in a uniform-target variant ``-(1/|S|) sum_{j in S} log p_j`` with weight
    ``alpha``, annealed linearly from 1 to 0 over ``uniform_anneal_iters`` training steps. The
    uniform target counteracts the marginal loss reinforcing whichever fine class the model already
    prefers inside the labelled subtree. ``uniform_anneal_iters=0`` disables it.
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        per_pixel_loss_weight: float = 0.0,
        per_image_loss_weight: float = 1.0,
        uniform_anneal_iters: int = 0,
    ) -> None:
        super().__init__()
        self.per_pixel_loss_weight = per_pixel_loss_weight
        self.per_image_loss_weight = per_image_loss_weight
        self.uniform_anneal_iters = uniform_anneal_iters
        self.slices = taxonomic_rank_slices(concept_value2id)

        matrices = build_parent_matrices(taxonomy, concept_value2id)
        self._warn_classes_without_descendants(matrices)
        for (d, r), ancestor in matrices.items():
            self.register_buffer(f"ancestor_{d}_{r}", ancestor.bool())

    def _pair_name(self, d: int, r: int) -> str:
        return f"partial/{self.slices[d][0]}->{self.slices[r][0]}"

    def _warn_classes_without_descendants(
        self,
        matrices: dict[tuple[int, int], torch.Tensor],
    ) -> None:
        """Warn about coarse classes whose pixels are masked for lack of children."""
        for (d, r), ancestor in matrices.items():
            has_desc = ancestor.sum(dim=1) > 0
            missing = [name for k, name in enumerate(self.slices[d][2]) if not has_desc[k]]
            if missing:
                logger.warning(
                    "%s: no descendants for %s; those pixels are masked.",
                    self._pair_name(d, r),
                    missing,
                )

    def uniform_alpha(self, step: int) -> float:
        if self.uniform_anneal_iters <= 0:
            return 0.0
        return max(0.0, 1.0 - step / self.uniform_anneal_iters)

    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of the ``(d, r)`` partial terms, plus per-term and per-depth logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            concept_labels: Concept targets ``(B, C, H, W)`` in the taxonomic 0/1/2 encoding.
            step: Training step, used only to anneal the uniform-target mix.
        """
        targets, valid, depth = rank_targets(concept_labels, self.slices)

        loss_components: dict[str, float] = {}
        num_labelled = (depth >= 0).sum().clamp(min=1)
        for d, (rank, _, _) in enumerate(self.slices):
            loss_components[f"depth_frac/{rank}"] = ((depth == d).sum() / num_labelled).item()

        alpha = self.uniform_alpha(step)
        total = concept_logits.new_zeros(())
        for r in range(1, len(self.slices)):
            d = r - 1
            ancestor = getattr(self, f"ancestor_{d}_{r}")
            rows = targets[d]
            mask = valid[d] & ancestor.any(dim=1)[rows]
            loss_components[self._pair_name(d, r)] = 0.0
            if not mask.any():
                continue

            _, offset_r, names_r = self.slices[r]
            logits_r = concept_logits[:, offset_r : offset_r + len(names_r)]
            pixel_logits = logits_r.permute(0, 2, 3, 1)[mask]  # (P, C_r)
            in_subtree = ancestor[rows[mask]]  # (P, C_r)
            lse_all = pixel_logits.logsumexp(dim=1)
            lse_subtree = pixel_logits.masked_fill(~in_subtree, float("-inf")).logsumexp(dim=1)
            pixel_loss = lse_all - lse_subtree
            if alpha > 0:
                log_probs = pixel_logits - lse_all.unsqueeze(1)
                uniform = -(log_probs * in_subtree).sum(dim=1) / in_subtree.sum(dim=1)
                pixel_loss = (1 - alpha) * pixel_loss + alpha * uniform

            per_pixel = concept_logits.new_zeros(mask.shape)
            per_pixel[mask] = pixel_loss
            pair_loss = reduce_masked_loss(
                per_pixel,
                mask,
                per_pixel_loss_weight=self.per_pixel_loss_weight,
                per_image_loss_weight=self.per_image_loss_weight,
            )
            loss_components[self._pair_name(d, r)] = pair_loss.item()
            total = total + pair_loss
        return total, loss_components


class TaxonomicCoherenceLoss(torch.nn.Module):
    """Label-free KL coherence loss between adjacent rank heads.

    For each adjacent pair (coarse rank ``r``, fine rank ``r+1``) and each selected pixel:

        L_KL^(r) = sum_i p_i * (log p_i - log q_i),   p = softmax(coarse),  q = A_r @ softmax(fine)

    With ``stop_grad_coarse`` the coarse distribution is detached in its coarse role only, so a
    middle head still receives gradients as the fine side of the pair above it.

    The data taxonomy is not a clean tree, so each pair is restricted to coarse classes with at
    least one child and fine classes with at least one parent, and both softmaxes are renormalised
    over those classes. A fine class with several parents splits its mass equally between them.

    Args:
        pixels: ``"all_valid"`` (every non-void pixel) or ``"not_fine_labelled"`` (only pixels
            without a label at the pair's fine rank).
        boundary_px: Exclude pixels within this many px of two different annotated labels.
    """

    PIXEL_MODES = ("all_valid", "not_fine_labelled")

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        stop_grad_coarse: bool = True,
        pixels: str = "all_valid",
        boundary_px: int = 0,
    ) -> None:
        super().__init__()
        if pixels not in self.PIXEL_MODES:
            raise ValueError(f"kl_pixels must be one of {self.PIXEL_MODES}, got {pixels!r}")
        self.stop_grad_coarse = stop_grad_coarse
        self.pixels = pixels
        self.boundary_px = boundary_px
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
        concept_labels: torch.Tensor,
        target_labels: torch.Tensor,
        void_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Sum of the adjacent-pair KL terms, plus per-pair logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            concept_labels: Concept targets ``(B, C, H, W)``; used only for ``not_fine_labelled``.
            target_labels: Target-space labels ``(B, H, W)``; used only for the boundary band.
            void_mask: Optional ``(B, H, W)`` mask of void pixels (padding, out of frame), which are
                always excluded. ``None`` treats every pixel as valid.
        """
        valid = torch.ones_like(target_labels, dtype=torch.bool)
        if void_mask is not None:
            valid = valid & ~void_mask
        if self.boundary_px > 0:
            valid = valid & ~boundary_band(target_labels, self.boundary_px)
        if self.pixels == "not_fine_labelled":
            _, labelled, _ = rank_targets(concept_labels, self.slices)

        loss_components: dict[str, float] = {}
        total = concept_logits.new_zeros(())
        for r in range(len(self.slices) - 1):
            mask = valid if self.pixels == "all_valid" else valid & ~labelled[r + 1]
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
    """Taxonomy-aware terms added on top of the per-rank cross-entropy of the concept loss.

        L = mu * sum_r L_partial^(r-1->r) / num_ranks + lambda(t) * sum_r L_KL^(r) / num_ranks

    Both terms are divided by the number of taxonomic ranks like the per-rank cross-entropy in
    :class:`~mermaidseg.model.loss.ConceptBottleneckLoss`, so their weights are on the same scale. ``lambda(t)`` is 0 before
    ``kl_start_iters``, then ramps linearly to ``kl_weight`` over ``kl_ramp_iters``. ``t`` counts
    optimizer updates (``MetaModel.global_step``), the unit of the LR warmup and decay, so the
    schedule means the same in every config and survives a mid-epoch resume. Each term is only
    built when its weight is positive.

    Args:
        taxonomy: Parent-child structure from ``build_taxonomy_tree`` over ``class_to_concepts.csv``,
            the same tree the hierarchical metrics use.
        concept_value2id: Concept name -> value -> channel mapping.
        partial_weight: Weight ``mu`` of :class:`TaxonomicPartialLoss`. Defaults to 0 (off).
        partial_uniform_anneal_iters: Optimizer updates over which the partial loss's uniform-target
            mix anneals from 1 to 0. Defaults to 0 (no uniform mix).
        per_pixel_loss_weight: Partial-loss reduction, matching the concept loss's setting.
        per_image_loss_weight: Partial-loss reduction, matching the concept loss's setting.
        kl_weight: Peak weight ``lambda_kl`` of :class:`TaxonomicCoherenceLoss`. Defaults to 0 (off).
        kl_start_iters: The KL weight is 0 before this many optimizer updates.
        kl_ramp_iters: The KL weight then ramps linearly to ``kl_weight`` over this many updates
            (0 = jump straight to ``kl_weight``).
        kl_stop_grad_coarse: Detach the coarse head's distribution in each KL pair.
        kl_pixels: ``"all_valid"`` or ``"not_fine_labelled"``.
        kl_boundary_px: Exclude pixels within this many px of an annotated label boundary.
    """

    def __init__(
        self,
        taxonomy: TaxonomyTree,
        concept_value2id: dict[str, dict[str, list[int]]],
        partial_weight: float = 0.0,
        partial_uniform_anneal_iters: int = 0,
        per_pixel_loss_weight: float = 0.0,
        per_image_loss_weight: float = 1.0,
        kl_weight: float = 0.0,
        kl_start_iters: int = 0,
        kl_ramp_iters: int = 0,
        kl_stop_grad_coarse: bool = True,
        kl_pixels: str = "all_valid",
        kl_boundary_px: int = 0,
    ) -> None:
        super().__init__()
        self.partial_weight = partial_weight
        self.kl_peak_weight = kl_weight
        self.kl_start_iters = kl_start_iters
        self.kl_ramp_iters = kl_ramp_iters
        self.num_ranks = len(TAXONOMIC_CONCEPTS)

        self.partial_loss = None
        if partial_weight > 0:
            self.partial_loss = TaxonomicPartialLoss(
                taxonomy,
                concept_value2id,
                per_pixel_loss_weight=per_pixel_loss_weight,
                per_image_loss_weight=per_image_loss_weight,
                uniform_anneal_iters=partial_uniform_anneal_iters,
            )
        self.coherence_loss = None
        if kl_weight > 0:
            self.coherence_loss = TaxonomicCoherenceLoss(
                taxonomy,
                concept_value2id,
                stop_grad_coarse=kl_stop_grad_coarse,
                pixels=kl_pixels,
                boundary_px=kl_boundary_px,
            )

    def kl_weight(self, step: int) -> float:
        """``lambda(t)``: 0 before ``kl_start_iters``, then a linear ramp to the peak weight."""
        if self.kl_peak_weight <= 0 or step < self.kl_start_iters:
            return 0.0
        if self.kl_ramp_iters <= 0:
            return self.kl_peak_weight
        return self.kl_peak_weight * min(1.0, (step - self.kl_start_iters) / self.kl_ramp_iters)

    def forward(
        self,
        concept_logits: torch.Tensor,
        concept_labels: torch.Tensor,
        target_labels: torch.Tensor,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted sum of the enabled taxonomic terms, plus their logging values.

        Args:
            concept_logits: Raw concept logits ``(B, C, H, W)``; taxonomic ranks come first.
            concept_labels: Concept targets ``(B, C, H, W)`` in the taxonomic 0/1/2 encoding.
            target_labels: Target-space labels ``(B, H, W)``; used only for the KL boundary band.
            step: Optimizer updates so far (``MetaModel.global_step``). Drives the KL weight
                schedule and the partial loss's uniform-target anneal.
        """
        loss_components: dict[str, float] = {}
        total = concept_logits.new_zeros(())

        if self.partial_loss is not None:
            partial_value, partial_components = self.partial_loss(
                concept_logits, concept_labels, step=step
            )
            partial_value = partial_value / self.num_ranks
            loss_components.update(partial_components)
            loss_components["partial"] = partial_value.item()
            total = total + self.partial_weight * partial_value

        kl_weight = self.kl_weight(step)
        if self.coherence_loss is not None and kl_weight > 0:
            kl_value, kl_components = self.coherence_loss(
                concept_logits, concept_labels, target_labels
            )
            kl_value = kl_value / self.num_ranks
            loss_components.update(kl_components)
            loss_components["kl"] = kl_value.item()
            loss_components["kl_weight"] = kl_weight
            total = total + kl_weight * kl_value

        return total, loss_components
