"""Metric policy helpers for training checkpointing and early stopping."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

METRIC_POLICY = {
    "loss": "min",
    "accuracy": "max",
    "miou": "max",
    "f1-score": "max",
}


def canonical_metric_name(metric_of_interest: str) -> str:
    key = metric_of_interest.lower()
    if key not in METRIC_POLICY:
        allowed = ", ".join(sorted(METRIC_POLICY))
        raise ValueError(f"Unsupported metric_of_interest '{metric_of_interest}'. Allowed metrics: {allowed}.")
    return key


def metric_direction(metric_name: str) -> str:
    return METRIC_POLICY[metric_name]


def to_scalar_metric(metric_name: str, metric_value: object) -> float:
    if torch.is_tensor(metric_value):
        if metric_value.ndim != 0:
            raise ValueError(f"metric_of_interest '{metric_name}' must be scalar; got tensor shape {tuple(metric_value.shape)}.")
        return float(metric_value.item())

    if isinstance(metric_value, np.ndarray):
        if metric_value.ndim != 0:
            raise ValueError(f"metric_of_interest '{metric_name}' must be scalar; got array shape {metric_value.shape}.")
        return float(metric_value.item())

    if np.isscalar(metric_value):
        return float(metric_value)

    raise ValueError(f"metric_of_interest '{metric_name}' must be scalar; got {type(metric_value).__name__}.")


def extract_metric_value(
    metric_of_interest: str,
    val_loss: float,
    val_metric_results: dict[str, float | NDArray[np.float64]],
) -> float:
    if metric_of_interest == "loss":
        return float(val_loss)

    if metric_of_interest in val_metric_results:
        return to_scalar_metric(metric_of_interest, val_metric_results[metric_of_interest])

    available = ", ".join(sorted(val_metric_results)) or "(none)"
    raise ValueError(f"metric_of_interest '{metric_of_interest}' is not present in validation metrics. Available metrics: {available}.")
