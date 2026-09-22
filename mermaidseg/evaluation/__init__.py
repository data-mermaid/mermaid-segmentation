"""Full-scale evaluation suite for the concept-bottleneck coral model.

This package provides standalone, cluster-ready evaluations that run at *source
resolution* (the model runs at its configured ``input_size`` and predictions are
sampled/upsampled back to the original image resolution before metrics are
computed):

- CoralNet validation (per source id + aggregate)
- Pacific Labeled Corals validation (per region + aggregate)
- Benthos zero-shot segmentation (per orthomosaic + aggregate)
- CoralscapesV2 linear probe (dense test accuracy / mIoU)

The command-line entry point is ``scripts/evaluate.py``.
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "predictor",
    "gt_mapping",
    "point_datasets",
    "coralnet_eval",
    "pacific_eval",
    "benthos_eval",
    "coralscapes_probe",
]
