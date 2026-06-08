"""Metric policy helpers for training checkpointing and early stopping."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

METRIC_POLICY = {
    "loss": "min",
    "accuracy": "max",
}

METRIC_RESULT_KEYS = {
    "loss": "loss",
    "accuracy": "accuracy/classification",
    "accuracy/classification": "accuracy/classification",
}

SUPPORTED_METRIC_NAMES = tuple(sorted(METRIC_POLICY))


def canonical_metric_name(metric_of_interest: str) -> str:
    key = metric_of_interest.lower().strip()
    if key in METRIC_POLICY:
        return key
    if key in METRIC_RESULT_KEYS and METRIC_RESULT_KEYS[key] == key:
        return "accuracy"
    allowed = ", ".join(sorted({*SUPPORTED_METRIC_NAMES, *METRIC_RESULT_KEYS}))
    raise ValueError(
        f"Unsupported metric_of_interest '{metric_of_interest}'. Allowed metrics: {allowed}."
    )


def metric_result_key(metric_of_interest: str) -> str:
    key = metric_of_interest.lower().strip()
    if key in METRIC_RESULT_KEYS:
        return METRIC_RESULT_KEYS[key]
    if key in METRIC_POLICY:
        return key
    allowed = ", ".join(sorted({*SUPPORTED_METRIC_NAMES, *METRIC_RESULT_KEYS}))
    raise ValueError(
        f"Unsupported metric_of_interest '{metric_of_interest}'. Allowed metrics: {allowed}."
    )


def metric_direction(metric_name: str) -> str:
    return METRIC_POLICY[metric_name]


def to_scalar_metric(metric_name: str, metric_value: object) -> float:
    if torch.is_tensor(metric_value):
        if metric_value.ndim != 0:
            raise ValueError(
                f"metric_of_interest '{metric_name}' must be scalar; got tensor shape {tuple(metric_value.shape)}."
            )
        return float(metric_value.item())

    if isinstance(metric_value, np.ndarray):
        if metric_value.ndim != 0:
            raise ValueError(
                f"metric_of_interest '{metric_name}' must be scalar; got array shape {metric_value.shape}."
            )
        return float(metric_value.item())

    if np.isscalar(metric_value):
        return float(metric_value)

    raise ValueError(
        f"metric_of_interest '{metric_name}' must be scalar; got {type(metric_value).__name__}."
    )


def extract_metric_value(
    metric_of_interest: str,
    val_loss: float,
    val_metric_results: dict[str, float | NDArray[np.float64]],
) -> float:
    result_key = metric_result_key(metric_of_interest)
    if result_key == "loss":
        return float(val_loss)

    if result_key in val_metric_results:
        return to_scalar_metric(result_key, val_metric_results[result_key])

    available = ", ".join(sorted(val_metric_results)) or "(none)"
    raise ValueError(
        f"metric_of_interest '{metric_of_interest}' is not present in validation metrics. Available metrics: {available}."
    )