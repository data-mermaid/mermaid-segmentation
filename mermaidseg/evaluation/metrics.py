"""Additive metric accumulators for the evaluation suite.

All accumulators store plain integer count tensors (confusion matrices,
per-concept TP/FP/FN/TN, per-rank correct/total counts) so that:

- a *pooled* (micro) aggregate is the element-wise sum of the per-unit
  accumulators, and
- a *macro* aggregate is the mean of the per-unit scalar metrics.

The design keeps everything picklable (numpy only) and lets the drivers finalize
and print a unit as soon as its last image has been consumed, then add it into
the pooled accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mermaidseg.dataset_reconciliation.concepts import TAXONOMIC_CONCEPTS


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


class ClassConfusion:
    """Confusion-matrix accumulator for a discrete class label space.

    Rows are ground truth, columns are prediction. ``ignore_index`` (if not
    ``None``) is excluded from *ground-truth* accumulation (pixels/points whose
    GT equals ``ignore_index`` are dropped), but predictions may still take that
    value.
    """

    def __init__(self, num_classes: int, ignore_index: int | None = 0):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, gt: NDArray[np.integer], pred: NDArray[np.integer]) -> None:
        gt = np.asarray(gt).reshape(-1).astype(np.int64)
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        if gt.shape != pred.shape:
            raise ValueError(f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}")
        valid = (gt >= 0) & (gt < self.num_classes)
        if self.ignore_index is not None:
            valid &= gt != self.ignore_index
        # Clip predictions defensively into range (out-of-range preds are rare
        # but should not crash bincount).
        g = gt[valid]
        p = np.clip(pred[valid], 0, self.num_classes - 1)
        if g.size == 0:
            return
        idx = g * self.num_classes + p
        self.cm += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def merge(self, other: "ClassConfusion") -> None:
        if other.num_classes != self.num_classes:
            raise ValueError("Cannot merge ClassConfusion with different num_classes")
        self.cm += other.cm

    def clone(self) -> "ClassConfusion":
        out = ClassConfusion(self.num_classes, self.ignore_index)
        out.cm = self.cm.copy()
        return out

    @property
    def total(self) -> int:
        return int(self.cm.sum())

    def accuracy(self) -> float:
        return _safe_div(np.trace(self.cm), self.cm.sum())

    def per_class_iou(self) -> NDArray[np.float64]:
        tp = np.diag(self.cm).astype(np.float64)
        gt = self.cm.sum(axis=1).astype(np.float64)
        pred = self.cm.sum(axis=0).astype(np.float64)
        union = gt + pred - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        return iou

    def _eval_class_ids(self, class_ids: list[int] | None) -> list[int]:
        if class_ids is not None:
            return list(class_ids)
        ids = list(range(self.num_classes))
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            ids = [c for c in ids if c != self.ignore_index]
        return ids

    def miou(self, class_ids: list[int] | None = None) -> float:
        iou = self.per_class_iou()
        ids = self._eval_class_ids(class_ids)
        vals = np.array([iou[c] for c in ids], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    def to_dict(self, id2name: dict[int, str] | None = None, class_ids: list[int] | None = None) -> dict:
        iou = self.per_class_iou()
        ids = self._eval_class_ids(class_ids)
        per_class = {}
        for c in ids:
            name = id2name.get(c, str(c)) if id2name else str(c)
            per_class[name] = {
                "iou": (None if np.isnan(iou[c]) else float(iou[c])),
                "gt_count": int(self.cm[c].sum()),
                "pred_count": int(self.cm[:, c].sum()),
                "tp": int(self.cm[c, c]),
            }
        return {
            "accuracy": self.accuracy(),
            "miou": self.miou(class_ids=ids),
            "total": self.total,
            "per_class": per_class,
        }


@dataclass
class TaxonomicRankAccuracy:
    """Per-rank taxonomic accuracy including / excluding ``None``.

    Concept channels use ``0=not_given, 1=inactive, 2=active``. A pixel is
    *given* at this rank if any channel in the rank slice equals 2. The GT class
    is the channel holding the 2; the prediction is the argmax over the same
    channels of the model's (per-rank softmax) probabilities.

    - ``acc_all``: over all *given* pixels (including the ``rank__none`` value).
    - ``acc_living``: over given pixels whose GT value is not ``rank__none``.
    """

    rank: str
    channel_indices: list[int]
    none_local_index: int | None = None
    correct_all: int = 0
    total_all: int = 0
    correct_living: int = 0
    total_living: int = 0

    def update(self, gt_rows: NDArray[np.integer], pred_probs: NDArray[np.floating]) -> None:
        if not self.channel_indices:
            return
        g = np.asarray(gt_rows)[:, self.channel_indices]
        p = np.asarray(pred_probs)[:, self.channel_indices]
        active = g == 2
        valid = active.any(axis=1)
        if not valid.any():
            return
        gt_idx = active[valid].argmax(axis=1)
        pred_idx = p[valid].argmax(axis=1)
        correct = gt_idx == pred_idx

        self.correct_all += int(correct.sum())
        self.total_all += int(valid.sum())

        if self.none_local_index is not None:
            living = gt_idx != self.none_local_index
        else:
            living = np.ones_like(gt_idx, dtype=bool)
        self.correct_living += int(correct[living].sum())
        self.total_living += int(living.sum())

    def merge(self, other: "TaxonomicRankAccuracy") -> None:
        self.correct_all += other.correct_all
        self.total_all += other.total_all
        self.correct_living += other.correct_living
        self.total_living += other.total_living

    def clone(self) -> "TaxonomicRankAccuracy":
        return TaxonomicRankAccuracy(
            rank=self.rank,
            channel_indices=list(self.channel_indices),
            none_local_index=self.none_local_index,
            correct_all=self.correct_all,
            total_all=self.total_all,
            correct_living=self.correct_living,
            total_living=self.total_living,
        )

    @property
    def acc_all(self) -> float:
        return _safe_div(self.correct_all, self.total_all)

    @property
    def acc_living(self) -> float:
        return _safe_div(self.correct_living, self.total_living)

    def to_dict(self) -> dict:
        return {
            "acc_all": self.acc_all,
            "acc_living": self.acc_living,
            "n_all": self.total_all,
            "n_living": self.total_living,
        }


class BinaryConceptStats:
    """Per-concept accuracy / F1 for multi-hot binary concepts.

    Concept channels use ``0=not_given, 1=False, 2=True``. Only pixels with GT in
    ``{1, 2}`` (given) are scored. Prediction is ``prob > threshold``. F1 treats
    ``True`` as the positive class.
    """

    def __init__(self, names: list[str], channel_indices: list[int], threshold: float = 0.5):
        if len(names) != len(channel_indices):
            raise ValueError("names and channel_indices must align")
        self.names = list(names)
        self.channel_indices = list(channel_indices)
        self.threshold = float(threshold)
        n = len(names)
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)
        self.tn = np.zeros(n, dtype=np.int64)

    def update(self, gt_rows: NDArray[np.integer], pred_probs: NDArray[np.floating]) -> None:
        if not self.channel_indices:
            return
        g = np.asarray(gt_rows)[:, self.channel_indices]
        p = np.asarray(pred_probs)[:, self.channel_indices] > self.threshold
        valid = g > 0
        gt_true = g == 2
        gt_false = g == 1
        self.tp += (valid & gt_true & p).sum(axis=0).astype(np.int64)
        self.fp += (valid & gt_false & p).sum(axis=0).astype(np.int64)
        self.fn += (valid & gt_true & ~p).sum(axis=0).astype(np.int64)
        self.tn += (valid & gt_false & ~p).sum(axis=0).astype(np.int64)

    def merge(self, other: "BinaryConceptStats") -> None:
        if other.names != self.names:
            raise ValueError("Cannot merge BinaryConceptStats with different concepts")
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    def clone(self) -> "BinaryConceptStats":
        out = BinaryConceptStats(self.names, self.channel_indices, self.threshold)
        out.tp = self.tp.copy()
        out.fp = self.fp.copy()
        out.fn = self.fn.copy()
        out.tn = self.tn.copy()
        return out

    def per_concept(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i, name in enumerate(self.names):
            tp, fp, fn, tn = (int(self.tp[i]), int(self.fp[i]), int(self.fn[i]), int(self.tn[i]))
            n_valid = tp + fp + fn + tn
            acc = _safe_div(tp + tn, n_valid)
            f1_den = 2 * tp + fp + fn
            f1 = _safe_div(2 * tp, f1_den)
            out[name] = {
                "accuracy": acc,
                "f1": f1,
                "n_valid": n_valid,
                "n_true": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        return out

    def to_dict(self) -> dict:
        per = self.per_concept()
        accs = [v["accuracy"] for v in per.values() if v["n_valid"] > 0]
        f1s = [v["f1"] for v in per.values() if (2 * v["tp"] + v["fp"] + v["fn"]) > 0]
        return {
            "macro_accuracy": float(np.mean(accs)) if accs else float("nan"),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "n_concepts_scored": len(accs),
            "per_concept": per,
        }


@dataclass
class ConceptLayout:
    """Channel layout derived from the model's ordered concept names."""

    concept_names: list[str]
    taxo: list[tuple[str, list[int], int | None]]  # (rank, indices, none_local_index)
    binary_names: list[str]
    binary_indices: list[int]

    @classmethod
    def from_concept_names(cls, concept_names: list[str]) -> "ConceptLayout":
        taxo: list[tuple[str, list[int], int | None]] = []
        for rank in TAXONOMIC_CONCEPTS:
            prefix = f"{rank}__"
            indices = [i for i, n in enumerate(concept_names) if n.startswith(prefix)]
            if not indices:
                continue
            none_local = None
            for local, gi in enumerate(indices):
                if concept_names[gi] == f"{rank}__none":
                    none_local = local
                    break
            taxo.append((rank, indices, none_local))
        binary_indices = [i for i, n in enumerate(concept_names) if "__" not in n]
        binary_names = [concept_names[i] for i in binary_indices]
        return cls(
            concept_names=list(concept_names),
            taxo=taxo,
            binary_names=binary_names,
            binary_indices=binary_indices,
        )


@dataclass
class MetricBundle:
    """Bundle of class + concept accumulators for one evaluation unit."""

    class_confusion: ClassConfusion
    taxo: list[TaxonomicRankAccuracy] = field(default_factory=list)
    binary: BinaryConceptStats | None = None
    num_points: int = 0

    @classmethod
    def create(cls, num_classes: int, layout: ConceptLayout, ignore_index: int = 0) -> "MetricBundle":
        taxo = [
            TaxonomicRankAccuracy(rank=rank, channel_indices=idx, none_local_index=none_local)
            for rank, idx, none_local in layout.taxo
        ]
        binary = BinaryConceptStats(layout.binary_names, layout.binary_indices)
        return cls(
            class_confusion=ClassConfusion(num_classes, ignore_index=ignore_index),
            taxo=taxo,
            binary=binary,
            num_points=0,
        )

    def update(
        self,
        gt_class: NDArray[np.integer],
        pred_class: NDArray[np.integer],
        gt_rows: NDArray[np.integer],
        pred_probs: NDArray[np.floating],
    ) -> None:
        self.class_confusion.update(gt_class, pred_class)
        for t in self.taxo:
            t.update(gt_rows, pred_probs)
        if self.binary is not None:
            self.binary.update(gt_rows, pred_probs)
        self.num_points += int(np.asarray(gt_class).reshape(-1).shape[0])

    def merge(self, other: "MetricBundle") -> None:
        self.class_confusion.merge(other.class_confusion)
        for a, b in zip(self.taxo, other.taxo, strict=True):
            a.merge(b)
        if self.binary is not None and other.binary is not None:
            self.binary.merge(other.binary)
        self.num_points += other.num_points

    def clone(self) -> "MetricBundle":
        return MetricBundle(
            class_confusion=self.class_confusion.clone(),
            taxo=[t.clone() for t in self.taxo],
            binary=self.binary.clone() if self.binary is not None else None,
            num_points=self.num_points,
        )

    def to_dict(
        self, class_id2name: dict[int, str] | None = None, include_class_detail: bool = False
    ) -> dict:
        out = {
            "num_points": self.num_points,
            "class_accuracy": self.class_confusion.accuracy(),
            "class_miou": self.class_confusion.miou(),
            "class_points_scored": self.class_confusion.total,
            "taxonomic": {t.rank: t.to_dict() for t in self.taxo},
            "binary": self.binary.to_dict() if self.binary is not None else {},
        }
        if include_class_detail:
            out["class_detail"] = self.class_confusion.to_dict(id2name=class_id2name)
        return out
