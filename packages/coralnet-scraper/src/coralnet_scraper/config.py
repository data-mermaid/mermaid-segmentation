"""Environment-variable configuration for the CoralNet scraper."""

from __future__ import annotations

import os

DEFAULT_PREFIX = "coralnet-public-images"
# I/O-bound: scale beyond cpu_count since threads wait on network, not CPU
DEFAULT_SCRAPER_WORKERS = min(32, max(8, (os.cpu_count() or 4) * 4))

ENV_BUCKET = "MERMAID_CORALNET_BUCKET"
ENV_PREFIX = "MERMAID_CORALNET_PREFIX"
ENV_SCRAPER_WORKERS = "MERMAID_CORALNET_SCRAPER_WORKERS"


def get_bucket() -> str:
    """Return the destination bucket; there is no default."""
    bucket = os.getenv(ENV_BUCKET)
    if not bucket:
        raise ValueError(f"{ENV_BUCKET} must be set to a non-empty S3 bucket name")
    return bucket


def get_prefix() -> str:
    """Return the destination key prefix without surrounding slashes; it must not be empty."""
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
    raw = os.getenv(ENV_SCRAPER_WORKERS)
    if raw is None:
        return default
    try:
        workers = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_SCRAPER_WORKERS} must be an integer, got {raw!r}") from None
    return validate_workers(workers)
