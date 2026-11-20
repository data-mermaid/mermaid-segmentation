"""Environment-variable configuration for the CoralNet scraper."""

from __future__ import annotations

import math
import os

DEFAULT_PREFIX = "coralnet-public-images"
# I/O-bound: scale beyond cpu_count since threads wait on network, not CPU
DEFAULT_SCRAPER_WORKERS = min(32, max(8, (os.cpu_count() or 4) * 4))
DEFAULT_EXPORT_WORKERS = 3

ENV_BUCKET = "MERMAID_CORALNET_BUCKET"
ENV_PREFIX = "MERMAID_CORALNET_PREFIX"
ENV_SCRAPER_WORKERS = "MERMAID_CORALNET_SCRAPER_WORKERS"
ENV_EXPORT_WORKERS = "MERMAID_CORALNET_EXPORT_WORKERS"
ENV_REQUESTS_PER_SECOND = "MERMAID_CORALNET_REQUESTS_PER_SECOND"


def get_bucket() -> str:
    """Return the destination bucket name; there is no default.

    Raises:
        ValueError: If ``MERMAID_CORALNET_BUCKET`` is unset or empty.
    """
    bucket = os.getenv(ENV_BUCKET)
    if not bucket:
        raise ValueError(f"{ENV_BUCKET} must be set to a non-empty S3 bucket name")
    return bucket


def get_prefix() -> str:
    """Return the destination key prefix without surrounding slashes.

    Raises:
        ValueError: If the prefix is empty after stripping slashes.
    """
    prefix = os.getenv(ENV_PREFIX, DEFAULT_PREFIX).strip("/")
    if not prefix:
        raise ValueError(f"{ENV_PREFIX} must contain a non-empty key prefix")
    return prefix


def validate_workers(workers: int) -> int:
    """Return ``workers`` when it is an integer of at least 1, otherwise raise ``ValueError``."""
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError(f"scraper workers must be an integer, got {workers!r}")
    if workers < 1:
        raise ValueError(f"scraper workers must be >= 1, got {workers}")
    return workers


def get_scraper_workers(default: int = DEFAULT_SCRAPER_WORKERS) -> int:
    """Return the scraper worker count from the environment, or ``default`` when it is unset.

    Args:
        default: Worker count used when ``MERMAID_CORALNET_SCRAPER_WORKERS`` is not set.

    Returns:
        The environment value, validated to be at least 1, or ``default`` unchanged when the
        variable is unset.

    Raises:
        ValueError: If the environment value is not an integer of at least 1.
    """
    return _workers_from_env(ENV_SCRAPER_WORKERS, default)


def get_export_workers(default: int = DEFAULT_EXPORT_WORKERS) -> int:
    """Return the paginated-export worker count from the environment, or ``default`` when unset.

    Args:
        default: Worker count used when ``MERMAID_CORALNET_EXPORT_WORKERS`` is not set.

    Raises:
        ValueError: If the environment value is not an integer of at least 1.
    """
    return _workers_from_env(ENV_EXPORT_WORKERS, default)


def _workers_from_env(name: str, default: int) -> int:
    """Return the validated integer in env var ``name``, or ``default`` when it is unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        workers = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    return validate_workers(workers)


def validate_requests_per_second(rate: float) -> float:
    """Return ``rate`` when it is a finite number > 0, otherwise raise ``ValueError``."""
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"requests per second must be a finite number > 0, got {rate!r}")
    return rate


def get_requests_per_second() -> float | None:
    """Return the CoralNet request-rate limit from the environment, or ``None`` when unset.

    Raises:
        ValueError: If ``MERMAID_CORALNET_REQUESTS_PER_SECOND`` is not a finite number > 0.
    """
    raw = os.getenv(ENV_REQUESTS_PER_SECOND)
    if raw is None:
        return None
    try:
        rate = float(raw)
    except ValueError:
        raise ValueError(f"{ENV_REQUESTS_PER_SECOND} must be a number, got {raw!r}") from None
    return validate_requests_per_second(rate)
