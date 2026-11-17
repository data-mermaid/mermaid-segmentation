"""Metric policy helpers for training checkpointing and early stopping."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

METRIC_POLICY = {
    "loss": "min",
    "accuracy": "max",
    "miou": "max",
    "f1-score": "max",
}

SUPPORTED_METRIC_NAMES = tuple(sorted(METRIC_POLICY))


def canonical_metric_name(metric_of_interest: str) -> str:
    """Return the normalized metric name.

    Args:
        metric_of_interest: User-supplied metric name, in any case.

    Returns:
        The lower-cased, stripped name.

    Raises:
        ValueError: If the name is not a supported metric.
    """
    key = metric_of_interest.lower().strip()
    if key in METRIC_POLICY:
        return key
    allowed = ", ".join(SUPPORTED_METRIC_NAMES)
    raise ValueError(
        f"Unsupported metric_of_interest '{metric_of_interest}'. Allowed metrics: {allowed}."
    )


def metric_direction(metric_name: str) -> str:
    """Return ``"min"`` or ``"max"``, the optimization direction of a metric."""
    return METRIC_POLICY[metric_name]


def to_scalar_metric(metric_name: str, metric_value: object) -> float:
    """Convert a scalar tensor, array, or number to a float.

    Args:
        metric_name: Metric name, used in error messages.
        metric_value: Value to convert.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a scalar.
    """
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
        return float(cast("float", metric_value))

    raise ValueError(
        f"metric_of_interest '{metric_name}' must be scalar; got {type(metric_value).__name__}."
    )


def extract_metric_value(
    metric_of_interest: str,
    val_loss: float,
    val_metric_results: dict[str, float | NDArray[np.float64]],
) -> float:
    """Return the scalar value of the metric of interest.

    Args:
        metric_of_interest: Metric name, in any case.
        val_loss: Validation loss, returned for the ``loss`` metric.
        val_metric_results: Validation metrics by name.

    Returns:
        The metric as a float.

    Raises:
        ValueError: If the metric is missing or not scalar.
    """
    key = metric_of_interest.lower().strip()
    if key == "loss":
        return float(val_loss)

    if key in val_metric_results:
        return to_scalar_metric(key, val_metric_results[key])

    available = ", ".join(sorted(val_metric_results)) or "(none)"
    raise ValueError(
        f"metric_of_interest '{metric_of_interest}' is not present in validation metrics. Available metrics: {available}."
    )
