"""Configuration constants and environment-variable resolution for the CoralNet dataset build."""

from __future__ import annotations

import os

from coralnet_scraper.config import get_bucket, get_prefix, validate_workers

__all__ = [
    "DEFAULT_OUTPUT_S3_PREFIX",
    "DEFAULT_WORKERS",
    "ENV_WORKERS",
    "RESIZE_THRESHOLD",
    "get_bucket",
    "get_prefix",
    "get_workers",
]

DEFAULT_OUTPUT_S3_PREFIX = "etl-outputs/coralnet"
DEFAULT_WORKERS = 16
# Longest edge in pixels; larger images are resized down to it.
RESIZE_THRESHOLD = 2048

ENV_WORKERS = "MERMAID_CORALNET_ETL_WORKERS"


def get_workers(default: int = DEFAULT_WORKERS) -> int:
    """Return the S3 worker count from the environment, or ``default`` when it is unset.

    Args:
        default: Worker count used when ``MERMAID_CORALNET_ETL_WORKERS`` is not set.

    Returns:
        The environment value, validated to be at least 1, or ``default`` when the variable is
        unset.

    Raises:
        ValueError: If the environment value is not an integer of at least 1.
    """
    raw = os.getenv(ENV_WORKERS)
    if raw is None:
        return default
    try:
        workers = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_WORKERS} must be an integer, got {raw!r}") from None
    try:
        return validate_workers(workers)
    except ValueError:
        raise ValueError(f"{ENV_WORKERS} must be >= 1, got {raw!r}") from None
