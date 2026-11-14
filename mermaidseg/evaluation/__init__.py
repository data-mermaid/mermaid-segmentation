"""Full-scale evaluation suite for the concept-bottleneck coral model.

This package provides standalone, cluster-ready evaluations that run at *source
resolution* (the model runs at the evaluation resolution and predictions are
sampled/upsampled back to the original image resolution before metrics are
computed):

- CoralNet val / test sources (per source id + aggregate)
- Pacific Labeled Corals validation (per region + aggregate)
- CoralscapesV2 linear probes, 95-class and the published 39-class map
  (dense val / test accuracy / mIoU)
- Validation loss on the val splits of the training data config

The command-line entry point is ``scripts/evaluate.py``.
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "hierarchical",
    "predictor",
    "gt_mapping",
    "point_datasets",
    "coralnet_eval",
    "pacific_eval",
    "coralscapes_probe",
    "val_loss",
]
