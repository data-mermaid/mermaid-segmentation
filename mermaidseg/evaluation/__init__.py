"""Full-scale evaluation suite for the concept-bottleneck coral model.

This package provides standalone, cluster-ready evaluations that run at *source
resolution* (the model runs at the evaluation resolution and predictions are
sampled/upsampled back to the original image resolution before metrics are
computed):

- CoralNet val / test sources (per source id + aggregate)
- Pacific Labeled Corals validation (per region + aggregate)
- CoralscapesV2 linear probes, 95-class and the published 39-class map
  (dense val / test accuracy / mIoU)
- Loss on the blocks of the chosen ``--split``: val, test, or the CoralNet split-file
  train groups under the concept source mask (train)

``run`` holds the registry of these evaluations and runs the selected ones; ``summary``
writes their ``summary.json`` and ``summary.md``. The command-line entry point is
``scripts/evaluate.py``.
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "hierarchical",
    "gt_mapping",
    "point_datasets",
    "point_sources",
    "rank_outputs",
    "hierarchical_metrics",
    "coralscapes_probe",
    "coralscapes_probe_points",
    "val_loss",
    "run",
    "summary",
]
