"""Hierarchical evaluation metrics over the taxonomic ranks.

* consistency: whether the predicted lineage is coherent, e.g. the predicted genus sits
  inside the predicted family.
* mistake severity: mean tree distance between predicted and true taxon, over mistakes.
* calibration: ECE of the top-1 confidence, per rank.
* rare taxon: macro accuracy per rank, split into head and tail taxa by support.

:class:`HierarchicalMetrics` accumulates counts per batch of
:class:`~mermaidseg.evaluation.rank_outputs.RankBatch`, so memory does not grow with the
dataset (see :mod:`mermaidseg.evaluation.hierarchical`).

The concept-bottleneck model applies a per-rank softmax, so ``RankBatch.scores`` are
probabilities and ECE reads their per-pixel maximum as the confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import torch

from mermaidseg.dataset_reconciliation.taxonomy_tree import (
    NONE_TAXON,
    TaxonomyTree,
    build_rank_incidence,
    rank_distance_matrix,
    taxon_channel_mask,
)
from mermaidseg.evaluation.rank_outputs import RankBatch

#: Four-way split of labelled pixels by whether GT and prediction are a taxon or 'none'
#: (taxon GT is further split into correct and wrong-taxon).
#: Calibration is reported over all labelled pixels and split by GT type.
CALIBRATION_SUBSETS = ("all", "taxon_gt", "none_gt")

ERROR_SPLIT = (
    "taxon_correct",
    "taxon_as_other_taxon",
    "taxon_as_none",
    "none_as_taxon",
    "none_as_none",
)


def _accumulator_dtype(device: torch.device) -> torch.dtype:
    """Return the float dtype for calibration accumulators on ``device``.

    float64 where supported; MPS has no float64 support, so it gets float32.
    """
    return torch.float32 if device.type == "mps" else torch.float64


class HierarchicalMetrics:
    """Accumulate the hierarchical metrics over an evaluation run.

    Args:
        tree: Taxonomy to score against. Under ``policy="permissive"`` a multi-parent taxon
            is consistent with any of its parents and takes the minimum tree distance; build
            the tree with ``"exclude"`` to drop such taxa instead.
        ranks: Ranks to score. Defaults to every rank in ``tree``.
        n_bins: Number of equal-width confidence bins for ECE.
        tail_quantile: Taxa below this quantile of support form the "tail" stratum.
        head_quantile: Taxa above this quantile of support form the "head" stratum.
        min_support: Taxa with fewer than this many ground-truth pixels are ignored
            when computing per-taxon accuracy, to avoid 0/1-pixel taxa dominating.
    """

    def __init__(
        self,
        tree: TaxonomyTree,
        ranks: tuple[str, ...] | None = None,
        n_bins: int = 15,
        tail_quantile: float = 0.5,
        head_quantile: float = 0.8,
        min_support: int = 1,
    ) -> None:
        """Set up empty accumulators; see the class docstring for the arguments."""
        self.tree = tree
        self.ranks = tuple(ranks) if ranks is not None else tuple(tree.ranks)
        self.n_bins = int(n_bins)
        self.tail_quantile = float(tail_quantile)
        self.head_quantile = float(head_quantile)
        self.min_support = int(min_support)

        self._pairs = [
            (coarse, fine) for coarse, fine in pairwise(self.ranks) if (coarse, fine) in tree.edges
        ]

        self._none_index = {r: tree.index_of(r).get(NONE_TAXON) for r in self.ranks}

        # Lazily moved onto the batch device on first update.
        self._incidence: dict[tuple[str, str], torch.Tensor] = {}
        self._distance: dict[str, torch.Tensor] = {}
        self._taxon_mask: dict[str, torch.Tensor] = {}
        self._taxon_or_none_mask: dict[str, torch.Tensor] = {}
        self._device: torch.device | None = None

        self.reset()

    # ---------------------------------------------------------------- state

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._consistent: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._consistent_total: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._consistent_labeled: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._consistent_labeled_total: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._truncated: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._truncated_labeled: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._none_none: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)
        self._no_parent: dict[tuple[str, str], int] = dict.fromkeys(self._pairs, 0)

        self._severity_sum: dict[str, float] = dict.fromkeys(self.ranks, 0.0)
        self._n_wrong: dict[str, int] = dict.fromkeys(self.ranks, 0)
        self._n_scored: dict[str, int] = dict.fromkeys(self.ranks, 0)
        self._severity_sum_with_none: dict[str, float] = dict.fromkeys(self.ranks, 0.0)
        self._n_wrong_with_none: dict[str, int] = dict.fromkeys(self.ranks, 0)
        self._error_split: dict[str, dict[str, int]] = {
            r: dict.fromkeys(ERROR_SPLIT, 0) for r in self.ranks
        }

        # Calibration bins, shape (len(CALIBRATION_SUBSETS), n_bins).
        self._bin_count: dict[str, torch.Tensor] = {}
        self._bin_conf: dict[str, torch.Tensor] = {}
        self._bin_correct: dict[str, torch.Tensor] = {}

        self._support: dict[str, torch.Tensor] = {}
        self._correct: dict[str, torch.Tensor] = {}

    def _ensure_device(self, device: torch.device) -> None:
        """Materialise the tree-derived lookup tensors on ``device`` (once)."""
        if self._device == device:
            return
        self._device = device
        self._incidence = {
            (c, f): torch.from_numpy(build_rank_incidence(self.tree, c, f).astype(np.int64)).to(
                device
            )
            for c, f in self._pairs
        }
        self._distance = {
            r: torch.from_numpy(rank_distance_matrix(self.tree, r).astype(np.int64)).to(device)
            for r in self.ranks
        }
        self._taxon_mask = {
            r: torch.from_numpy(taxon_channel_mask(self.tree, r)).to(device) for r in self.ranks
        }
        self._taxon_or_none_mask = {}
        for r, mask in self._taxon_mask.items():
            mask = mask.clone()
            if self._none_index[r] is not None:
                mask[self._none_index[r]] = True
            self._taxon_or_none_mask[r] = mask

    def _ensure_rank_state(self, rank: str, num_channels: int, device: torch.device) -> None:
        if rank in self._bin_count:
            return
        dtype = _accumulator_dtype(device)
        z = torch.zeros(len(CALIBRATION_SUBSETS), self.n_bins, dtype=dtype, device=device)
        self._bin_count[rank] = z.clone()
        self._bin_conf[rank] = z.clone()
        self._bin_correct[rank] = z.clone()
        c = torch.zeros(num_channels, dtype=torch.int64, device=device)
        self._support[rank] = c.clone()
        self._correct[rank] = c.clone()

    # ---------------------------------------------------------------- update

    @torch.no_grad()
    def update(self, ranks: dict[str, RankBatch]) -> None:
        """Accumulate one batch.

        Args:
            ranks: All taxonomic ranks for this batch, keyed by rank name. Passed together
                so the consistency metric sees the joint assignment.
        """
        present = {r: b for r, b in ranks.items() if r in self.ranks}
        if not present:
            return
        device = next(iter(present.values())).target.device
        self._ensure_device(device)

        for rank, batch in present.items():
            self._ensure_rank_state(rank, batch.num_channels, device)
            self._update_rank(rank, batch)

        for coarse, fine in self._pairs:
            if coarse in present and fine in present:
                self._update_consistency(coarse, fine, present[coarse], present[fine])

    def _update_rank(self, rank: str, batch: RankBatch) -> None:
        valid = batch.valid.reshape(-1)
        if not bool(valid.any()):
            return
        target_idx = batch.target_tree_index().reshape(-1)[valid]
        pred_idx = batch.pred_tree_index().reshape(-1)[valid]
        correct = target_idx.eq(pred_idx)

        # --- rare taxon: per-taxon support and hits -------------------------------
        n_channels = batch.num_channels
        self._support[rank] += torch.bincount(target_idx, minlength=n_channels)
        if bool(correct.any()):
            self._correct[rank] += torch.bincount(target_idx[correct], minlength=n_channels)

        # --- mistake severity: mean tree distance over MISTAKES --------------------
        # Main score: taxon-vs-taxon only. The 'with none' variant also scores 'none'
        # pixels; the distance matrix already puts 'none' at the maximum distance from
        # every taxon (lineages never meet) and 0 from itself.
        mask = self._taxon_mask[rank]
        scored = mask[target_idx] & mask[pred_idx]
        if bool(scored.any()):
            wrong = scored & ~correct
            self._n_scored[rank] += int(scored.sum().item())
            if bool(wrong.any()):
                distances = self._distance[rank][pred_idx[wrong], target_idx[wrong]]
                self._severity_sum[rank] += float(distances.sum().item())
                self._n_wrong[rank] += int(wrong.sum().item())

        mask = self._taxon_or_none_mask[rank]
        scored = mask[target_idx] & mask[pred_idx]
        if bool(scored.any()):
            wrong = scored & ~correct
            if bool(wrong.any()):
                distances = self._distance[rank][pred_idx[wrong], target_idx[wrong]]
                self._severity_sum_with_none[rank] += float(distances.sum().item())
                self._n_wrong_with_none[rank] += int(wrong.sum().item())
            target_none = self._is_none(rank, target_idx)
            pred_none = self._is_none(rank, pred_idx)
            split = self._error_split[rank]
            split["taxon_correct"] += int((scored & ~target_none & correct).sum().item())
            split["taxon_as_other_taxon"] += int(
                (scored & ~target_none & ~pred_none & ~correct).sum().item()
            )
            split["taxon_as_none"] += int((scored & ~target_none & pred_none).sum().item())
            split["none_as_taxon"] += int((scored & target_none & ~pred_none).sum().item())
            split["none_as_none"] += int((scored & target_none & pred_none).sum().item())

        # --- calibration: confidence-binned accuracy ------------------------------
        probs = batch.scores
        dtype = _accumulator_dtype(probs.device)
        confidence = probs.amax(dim=1).reshape(-1)[valid].to(dtype)
        bin_idx = torch.clamp((confidence * self.n_bins).long(), 0, self.n_bins - 1)
        # Subsets are chosen by GT only: selecting on the prediction (e.g. dropping
        # pixels predicted 'none') would remove exactly the confident misses.
        target_none = self._is_none(rank, target_idx)
        subsets = (torch.ones_like(target_none), ~target_none, target_none)
        for i, subset in enumerate(subsets):
            idx = bin_idx[subset]
            self._bin_count[rank][i].scatter_add_(0, idx, torch.ones_like(confidence[subset]))
            self._bin_conf[rank][i].scatter_add_(0, idx, confidence[subset])
            self._bin_correct[rank][i].scatter_add_(0, idx, correct[subset].to(dtype))

    def _update_consistency(
        self, coarse: str, fine: str, coarse_batch: RankBatch, fine_batch: RankBatch
    ) -> None:
        pair = (coarse, fine)
        incidence = self._incidence[pair]
        fine_pred = fine_batch.pred_tree_index().reshape(-1)
        coarse_pred = coarse_batch.pred_tree_index().reshape(-1)
        fine_labeled = fine_batch.valid.reshape(-1)

        # A coarse taxon over a fine 'none' ("Acroporidae, but not an organism") is a
        # truncated lineage. It is excluded from the main score and counted here, so the
        # score can also be reported with truncations treated as inconsistent.
        fine_is_none = self._is_none(fine, fine_pred)
        coarse_is_none = self._is_none(coarse, coarse_pred)
        truncated = fine_is_none & ~coarse_is_none
        self._truncated[pair] += int(truncated.sum().item())
        self._truncated_labeled[pair] += int((truncated & fine_labeled).sum().item())
        self._none_none[pair] += int((fine_is_none & coarse_is_none).sum().item())

        # A fine taxon with no recorded parent (orphan, or excluded) cannot be judged:
        # scoring it as inconsistent would penalise the model for a gap in the label space
        # rather than a modelling error.
        has_parent = incidence.sum(dim=1) > 0
        scorable = has_parent[fine_pred]
        self._no_parent[pair] += int((~scorable & ~fine_is_none).sum().item())
        if not bool(scorable.any()):
            return
        consistent = incidence[fine_pred, coarse_pred].eq(1) & scorable

        self._consistent[pair] += int(consistent.sum().item())
        self._consistent_total[pair] += int(scorable.sum().item())

        labeled = fine_labeled & scorable
        if bool(labeled.any()):
            self._consistent_labeled[pair] += int((consistent & labeled).sum().item())
            self._consistent_labeled_total[pair] += int(labeled.sum().item())

    def _is_none(self, rank: str, pred_idx: torch.Tensor) -> torch.Tensor:
        index = self._none_index[rank]
        if index is None:
            return torch.zeros_like(pred_idx, dtype=torch.bool)
        return pred_idx.eq(index)

    # ---------------------------------------------------------------- compute

    def compute(self) -> dict[str, float]:
        """Return all accumulated metrics as ``name -> value``."""
        out: dict[str, float] = {}
        out.update(self._compute_consistency())
        out.update(self._compute_severity())
        out.update(self._compute_calibration())
        out.update(self._compute_rare_taxon())
        return out

    def _compute_consistency(self) -> dict[str, float]:
        """Compute lineage consistency, with and without truncated lineages.

        The main score drops truncated lineages. Under ``.../truncated_as_inconsistent``
        they count as inconsistent. The pixel counts behind both are included.
        """
        out: dict[str, float] = {}
        total = dict.fromkeys(("ok", "n", "trunc", "none_none", "no_parent"), 0)
        labeled = dict.fromkeys(("ok", "n", "trunc"), 0)
        for pair in self._pairs:
            counts = {
                "ok": self._consistent[pair],
                "n": self._consistent_total[pair],
                "trunc": self._truncated[pair],
                "none_none": self._none_none[pair],
                "no_parent": self._no_parent[pair],
            }
            for key, value in counts.items():
                total[key] += value
            labeled["ok"] += self._consistent_labeled[pair]
            labeled["n"] += self._consistent_labeled_total[pair]
            labeled["trunc"] += self._truncated_labeled[pair]
            _emit_consistency(out, f"consistency/{pair[0]}->{pair[1]}", counts)
        _emit_consistency(out, "consistency/overall", total)
        _emit_consistency(out, "consistency/overall_labeled", labeled)
        return out

    def _compute_severity(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for rank in self.ranks:
            # Normalised so ranks are comparable: 1.0 means the average mistake is
            # maximally distant (lineages never meet).
            max_distance = self.tree.ranks.index(rank) + 1
            wrong = self._n_wrong[rank]
            if wrong:
                mean_distance = self._severity_sum[rank] / wrong
                out[f"severity/{rank}"] = mean_distance
                out[f"severity/{rank}/normalized"] = mean_distance / max_distance
                if self._n_scored[rank]:
                    out[f"severity/{rank}/error_rate"] = wrong / self._n_scored[rank]

            split = self._error_split[rank]
            n_scored = sum(split.values())
            if not n_scored:
                continue
            for name, count in split.items():
                out[f"severity/{rank}/with_none/n_{name}"] = float(count)
            wrong = self._n_wrong_with_none[rank]
            out[f"severity/{rank}/with_none/error_rate"] = wrong / n_scored
            if wrong:
                mean_distance = self._severity_sum_with_none[rank] / wrong
                out[f"severity/{rank}/with_none"] = mean_distance
                out[f"severity/{rank}/with_none/normalized"] = mean_distance / max_distance

        for suffix in ("normalized", "with_none/normalized"):
            per_rank = [out[k] for r in self.ranks if (k := f"severity/{r}/{suffix}") in out]
            if per_rank:
                key = "severity/mean_normalized"
                out[key if suffix == "normalized" else f"{key}/with_none"] = float(
                    np.mean(per_rank)
                )
        return out

    def _compute_calibration(self) -> dict[str, float]:
        """Compute ECE over all labelled pixels and split by GT type.

        Keys are ``ece/<rank>``, ``ece/<rank>/taxon_gt``, and ``ece/<rank>/none_gt``,
        each with its mean confidence, accuracy and pixel count.
        """
        out: dict[str, float] = {}
        eces: dict[str, list[float]] = {s: [] for s in CALIBRATION_SUBSETS}
        for rank in self.ranks:
            if rank not in self._bin_count:
                continue
            for i, subset in enumerate(CALIBRATION_SUBSETS):
                count = self._bin_count[rank][i]
                total = float(count.sum().item())
                if total == 0:
                    continue
                correct = self._bin_correct[rank][i]
                conf = self._bin_conf[rank][i]
                zeros = torch.zeros_like(count)
                accuracy = torch.where(count > 0, correct / count.clamp(min=1), zeros)
                confidence = torch.where(count > 0, conf / count.clamp(min=1), zeros)
                ece = float(((count / total) * (accuracy - confidence).abs()).sum().item())
                key = f"ece/{rank}" if subset == "all" else f"ece/{rank}/{subset}"
                out[key] = ece
                mean_conf = float((conf.sum() / total).item())
                if subset == "all":
                    out[f"confidence/{rank}"] = mean_conf
                else:
                    out[f"{key}/confidence"] = mean_conf
                out[f"{key}/accuracy"] = float((correct.sum() / total).item())
                out[f"{key}/n"] = total
                eces[subset].append(ece)
        for subset, values in eces.items():
            if values:
                out["ece/mean" if subset == "all" else f"ece/mean/{subset}"] = float(
                    np.mean(values)
                )
        return out

    def _compute_rare_taxon(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for rank in self.ranks:
            if rank not in self._support:
                continue
            support = self._support[rank].cpu().double().numpy()
            correct = self._correct[rank].cpu().double().numpy()
            mask = taxon_channel_mask(self.tree, rank) & (support >= self.min_support)
            if not mask.any():
                continue
            accuracy = np.divide(correct, support, out=np.zeros_like(support), where=support > 0)[
                mask
            ]
            seen_support = support[mask]
            out[f"rare/{rank}/n_taxa_seen"] = float(mask.sum())
            out[f"rare/{rank}/macro_acc"] = float(accuracy.mean())

            tail_cut = np.quantile(seen_support, self.tail_quantile)
            head_cut = np.quantile(seen_support, self.head_quantile)
            strata = {
                "tail": seen_support <= tail_cut,
                "head": seen_support >= head_cut,
                "mid": (seen_support > tail_cut) & (seen_support < head_cut),
            }
            for name, selector in strata.items():
                if selector.any():
                    out[f"rare/{rank}/{name}_macro_acc"] = float(accuracy[selector].mean())
                    out[f"rare/{rank}/{name}_n_taxa"] = float(selector.sum())
            if strata["head"].any() and strata["tail"].any():
                out[f"rare/{rank}/head_tail_gap"] = float(
                    accuracy[strata["head"]].mean() - accuracy[strata["tail"]].mean()
                )
        return out


def _emit_consistency(out: dict[str, float], key: str, counts: Mapping[str, int]) -> None:
    if not any(counts.values()):
        return
    n, trunc = counts["n"], counts["trunc"]
    if n:
        out[key] = counts["ok"] / n
    if n + trunc:
        out[f"{key}/truncated_as_inconsistent"] = counts["ok"] / (n + trunc)
        out[f"{key}/truncated_frac"] = trunc / (n + trunc)
    out[f"{key}/n_scored"] = float(n)
    out[f"{key}/n_truncated"] = float(trunc)
    if "none_none" in counts:
        out[f"{key}/n_none_none"] = float(counts["none_none"])
        out[f"{key}/n_no_parent"] = float(counts["no_parent"])
