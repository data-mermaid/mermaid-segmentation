"""Final-class-head distribution: GT vs predicted share per target class."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torchmetrics import Metric


class ClassPixelCounts(Metric):
    """Per-class pixel counts of the final class head, accumulated over every batch.

    Rows: 0 = GT on labelled pixels, 1 = prediction on labelled pixels, 2 = prediction on
    all pixels. Labelled means ``target != ignore_index``.
    """

    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state(
            "counts", default=torch.zeros(3, num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.reshape(-1).long()
        target = target.reshape(-1).long()
        labelled = target != self.ignore_index
        self.counts[0] += torch.bincount(target[labelled], minlength=self.num_classes)
        self.counts[1] += torch.bincount(preds[labelled], minlength=self.num_classes)
        self.counts[2] += torch.bincount(preds, minlength=self.num_classes)

    def compute(self) -> torch.Tensor:
        return self.counts


def format_class_distribution(
    counts: np.ndarray, class_names: Mapping[int, str], unit: str = "px"
) -> str:
    """Table of GT and predicted share per class, from :class:`ClassPixelCounts` output.

    ``unit`` labels the counted elements, e.g. ``"px"`` for dense masks or ``"points"`` for sparse
    point annotations.
    """
    gt, pred_labelled, pred_all = np.asarray(counts, dtype=np.int64)
    lines = [
        f"  labelled {unit} {gt.sum():,} of {pred_all.sum():,}",
        f"  'GT' and 'pred' are shares of labelled {unit}; 'pred (all)' includes unlabelled ones.",
        f"  {'id':>3} {'class':<28} {'GT ' + unit:>12} {'GT':>8} {'pred':>8} {'pred (all)':>11}",
    ]
    for c in sorted(np.flatnonzero(gt + pred_all), key=lambda c: (-gt[c], -pred_all[c])):
        lines.append(
            f"  {c:>3} {class_names.get(int(c), str(c)):<28} {gt[c]:>12,} "
            f"{gt[c] / max(gt.sum(), 1):>8.2%} "
            f"{pred_labelled[c] / max(pred_labelled.sum(), 1):>8.2%} "
            f"{pred_all[c] / max(pred_all.sum(), 1):>11.2%}"
        )
    return "\n".join(lines)
