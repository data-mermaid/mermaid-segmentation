"""Hierarchical and taxonomy-loss metrics for the point evaluations (``evaluate.py --hierarchical``).

Each annotated point contributes its per-rank softmax probabilities (sampled exactly as for the
standard metrics) and its ``0/1/2`` concept row. Two accumulators consume them:

* :class:`~mermaidseg.model.hierarchical_metrics.HierarchicalMetrics`: lineage consistency,
  mistake severity, calibration (ECE) and rare-taxon accuracy, keyed as in that module.
* :class:`LossTargetMetrics`: the quantities the taxonomy losses optimise, on held-out points, built
  from the losses' own matrices so they measure exactly the training objectives:

  - ``partial/<coarse>-><fine>/subtree_mass``: for points whose deepest labelled rank is
    ``coarse``, the mass of the ``fine`` head inside the labelled taxon's children
    (``exp(-L_partial)``); ``.../argmax_in_subtree`` and ``.../n`` alongside.
  - ``expected_distance/<rank>``: ``sum_j p_j * D[y, j] / D_max`` over points labelled at ``rank``
    (the distance loss's objective, in ``[0, 1]``); ``.../n`` alongside.
  - ``kl/<coarse>-><fine>``: mean ``KL(p_coarse || aggregated p_fine)`` over all points, with the
    KL loss's renormalisation and multi-parent splits (label-free); ``.../n`` alongside.

Model concept channels must match the tree's channel order rank by rank; :class:`HierarchicalSpec`
checks this once, so a checkpoint with a different concept layout fails instead of misreporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS
from mermaidseg.dataset_reconciliation.taxonomy_tree import (
    AmbiguityPolicy,
    TaxonomyTree,
    build_taxonomy_tree,
)
from mermaidseg.model.distance_loss import TaxonomicDistanceLoss
from mermaidseg.model.hierarchical_metrics import HierarchicalMetrics
from mermaidseg.model.rank_outputs import RankBatch
from mermaidseg.model.taxonomy_loss import TaxonomicCoherenceLoss, build_parent_matrices

logger = logging.getLogger(__name__)


def concept_value2id_from_names(concept_names: list[str]) -> dict[str, dict[str, list[int]]]:
    """Taxonomic ``concept_value2id`` (rank -> ordered values) from ``rank__value`` channel names.

    Only the key order matters to the taxonomy losses' channel slicing, so the value encodings are
    placeholders. Raises unless the taxonomic channels come first, rank by rank, in
    ``TAXONOMIC_CONCEPTS`` order, which is the layout the losses and metrics index into.
    """
    value2id: dict[str, dict[str, list[int]]] = {}
    expected_index = 0
    for rank in TAXONOMIC_CONCEPTS:
        prefix = f"{rank}__"
        indices = [i for i, name in enumerate(concept_names) if name.startswith(prefix)]
        if not indices:
            continue
        if indices != list(range(expected_index, expected_index + len(indices))):
            raise ValueError(
                f"Concept channels for rank {rank!r} are not contiguous at index {expected_index}; "
                "taxonomic ranks must come first, in TAXONOMIC_CONCEPTS order."
            )
        expected_index += len(indices)
        values = [concept_names[i][len(prefix) :] for i in indices]
        placeholder = [0] * len(values)
        value2id[rank] = {"not_given": placeholder, **{v: placeholder for v in values}}
    return value2id


@dataclass
class HierarchicalSpec:
    """Taxonomy-derived lookups shared by every per-unit accumulator of one evaluation."""

    tree: TaxonomyTree
    ranks: list[str]
    slices: list[tuple[str, int, int]]  # (rank, channel offset, channel count)
    parents: dict[int, torch.Tensor]  # d -> bool (C_d, C_{d+1}), parent -> child
    distance: dict[str, torch.Tensor]  # rank -> (C, C) tree distance / max distance
    kl_coarse_keep: dict[int, torch.Tensor]
    kl_fine_keep: dict[int, torch.Tensor]
    kl_groups: dict[int, torch.Tensor]

    @classmethod
    def build(
        cls,
        concept_names: list[str],
        taxonomy_csv: str | Path,
        ambiguity_policy: AmbiguityPolicy = "permissive",
    ) -> HierarchicalSpec:
        tree = build_taxonomy_tree(taxonomy_csv, ambiguity_policy=ambiguity_policy)
        value2id = concept_value2id_from_names(concept_names)
        slices, offset = [], 0
        for rank, values in value2id.items():
            names = [v for v in values if v != "not_given"]
            if names != tree.channels[rank]:
                raise ValueError(
                    f"Model concept channels at rank {rank!r} do not match the taxonomy tree from "
                    f"{taxonomy_csv} ({len(names)} vs {len(tree.channels[rank])} channels, or a "
                    "different order). Use the class_to_concepts.csv the checkpoint was trained with."
                )
            slices.append((rank, offset, len(names)))
            offset += len(names)

        parents = {d: m.bool() for (d, _), m in build_parent_matrices(tree, value2id).items()}
        distance_loss = TaxonomicDistanceLoss(tree, value2id)
        coherence = TaxonomicCoherenceLoss(tree, value2id)  # logs the taxonomy's defects once
        pairs = range(len(slices) - 1)
        return cls(
            tree=tree,
            ranks=[s[0] for s in slices],
            slices=slices,
            parents=parents,
            distance={r: getattr(distance_loss, f"distance_{r}") for r, _, _ in slices},
            kl_coarse_keep={r: getattr(coherence, f"coarse_keep_{r}") for r in pairs},
            kl_fine_keep={r: getattr(coherence, f"fine_keep_{r}") for r in pairs},
            kl_groups={r: getattr(coherence, f"groups_{r}") for r in pairs},
        )

    def pair_name(self, d: int) -> str:
        return f"{self.ranks[d]}->{self.ranks[d + 1]}"


def rank_targets_from_rows(
    concept_rows: torch.Tensor, spec: HierarchicalSpec
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Per-rank 0-based targets, valid masks, and the deepest labelled rank (-1 if none) per point."""
    targets, valid = [], []
    for _, offset, size in spec.slices:
        active = concept_rows[:, offset : offset + size].eq(2)
        targets.append(active.float().argmax(dim=1))
        valid.append(active.any(dim=1))
    rank_ids = torch.arange(len(spec.slices), device=concept_rows.device)[:, None]
    depth = torch.where(torch.stack(valid), rank_ids, -1).amax(dim=0)
    return targets, valid, depth


def point_rank_batches(
    concept_rows: torch.Tensor, probs: torch.Tensor, spec: HierarchicalSpec
) -> dict[str, RankBatch]:
    """``RankBatch`` per rank for ``N`` points, shaped as one ``1 x N`` image strip."""
    targets, valid, _ = rank_targets_from_rows(concept_rows, spec)
    batches = {}
    for d, (rank, offset, size) in enumerate(spec.slices):
        scores = probs[:, offset : offset + size]
        target = torch.where(valid[d], targets[d] + 1, torch.zeros_like(targets[d]))
        batches[rank] = RankBatch(
            rank=rank,
            scores=scores.T[None, :, None, :],
            target=target[None, None, :],
            pred=(scores.argmax(dim=1) + 1)[None, None, :],
            valid=valid[d][None, None, :],
        )
    return batches


class LossTargetMetrics:
    """Held-out partial-label subtree mass, expected tree distance and inter-head KL."""

    def __init__(self, spec: HierarchicalSpec):
        self.spec = spec
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _add(self, key: str, values: torch.Tensor) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(values.double().sum().item())
        self._counts[key] = self._counts.get(key, 0) + int(values.numel())

    @torch.no_grad()
    def update(self, concept_rows: torch.Tensor, probs: torch.Tensor) -> None:
        spec = self.spec
        device = probs.device
        targets, valid, depth = rank_targets_from_rows(concept_rows, spec)
        rank_probs = [probs[:, o : o + n].double() for _, o, n in spec.slices]
        tiny = torch.finfo(torch.float64).tiny

        for d in range(len(spec.slices) - 1):
            pair = spec.pair_name(d)
            parents = spec.parents[d].to(device)
            # Partial CE target population: d is the deepest labelled rank and has children.
            mask = depth.eq(d) & parents.any(dim=1)[targets[d]]
            if mask.any():
                children = parents[targets[d][mask]]  # (P, C_fine)
                fine = rank_probs[d + 1][mask]
                self._add(f"partial/{pair}/subtree_mass", (fine * children).sum(dim=1))
                in_subtree = children.gather(1, fine.argmax(dim=1, keepdim=True)).squeeze(1)
                self._add(f"partial/{pair}/argmax_in_subtree", in_subtree.double())

            # KL coherence, label-free, with the KL loss's kept classes and parent splits.
            p = rank_probs[d][:, spec.kl_coarse_keep[d].to(device)]
            q_fine = rank_probs[d + 1][:, spec.kl_fine_keep[d].to(device)]
            p = p / p.sum(dim=1, keepdim=True).clamp(min=tiny)
            q_fine = q_fine / q_fine.sum(dim=1, keepdim=True).clamp(min=tiny)
            q = q_fine @ spec.kl_groups[d].to(device).double().T
            kl = (p * (p.clamp(min=tiny).log() - q.clamp(min=tiny).log())).sum(dim=1)
            self._add(f"kl/{pair}", kl)

        for d, (rank, _, _) in enumerate(spec.slices):
            if valid[d].any():
                distance_rows = spec.distance[rank].to(device).double()[targets[d][valid[d]]]
                self._add(
                    f"expected_distance/{rank}", (rank_probs[d][valid[d]] * distance_rows).sum(dim=1)
                )

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, total in self._sums.items():
            n = self._counts[key]
            out[key] = total / n if n else float("nan")
            stem = key.rsplit("/", 1)[0] if key.startswith("partial/") else key
            out[f"{stem}/n"] = float(n)
        return out


class HierarchicalPointEval:
    """``HierarchicalMetrics`` and :class:`LossTargetMetrics` over one unit's annotated points.

    Points are buffered and handed to the accumulators in chunks of ``flush_points``: each update
    does a few hundred small device syncs, so per-image updates would dominate the evaluation.
    Every metric is a sum or count over points, so the chunking does not change the results.
    """

    def __init__(self, spec: HierarchicalSpec, flush_points: int = 20_000):
        self.spec = spec
        self.hierarchical = HierarchicalMetrics(spec.tree, ranks=tuple(spec.ranks), score_transform="none")
        self.loss_targets = LossTargetMetrics(spec)
        self.flush_points = int(flush_points)
        self.num_points = 0
        self._rows: list[torch.Tensor] = []
        self._probs: list[torch.Tensor] = []
        self._buffered = 0

    def update(self, concept_rows: NDArray[np.integer] | torch.Tensor, probs: torch.Tensor) -> None:
        """Accumulate ``N`` points: concept rows ``(N, K)`` (0/1/2) and probabilities ``(N, K)``."""
        if probs.shape[0] == 0:
            return
        self._rows.append(torch.as_tensor(np.asarray(concept_rows), dtype=torch.long, device=probs.device))
        self._probs.append(probs.float())
        self._buffered += int(probs.shape[0])
        self.num_points += int(probs.shape[0])
        if self._buffered >= self.flush_points:
            self._flush()

    def _flush(self) -> None:
        if not self._buffered:
            return
        rows, probs = torch.cat(self._rows), torch.cat(self._probs)
        self._rows, self._probs, self._buffered = [], [], 0
        self.hierarchical.update(point_rank_batches(rows, probs, self.spec))
        self.loss_targets.update(rows, probs)

    def compute(self) -> dict[str, float]:
        self._flush()
        return {
            "num_points": float(self.num_points),
            **self.hierarchical.compute(),
            **self.loss_targets.compute(),
        }
