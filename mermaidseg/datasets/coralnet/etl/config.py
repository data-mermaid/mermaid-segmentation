"""Configuration constants and env-var resolution for the CoralNet ETL.

All defaults can be overridden via environment variables so the ETL can be pointed at staging/dev
buckets or run with a different worker count without code changes.
"""

from __future__ import annotations

import os

DEFAULT_BUCKET = "dev-datamermaid-sm-sources"
DEFAULT_PREFIX = "coralnet-public-images"
DEFAULT_OUTPUT_S3_PREFIX = "etl-outputs/coralnet"
DEFAULT_WORKERS = 16
DEFAULT_IMAGE_WORKERS = 32
DEFAULT_JPEG_FIRST_CHUNK = 65_536
DEFAULT_JPEG_MAX_CHUNK = 1_048_576
DEFAULT_CHECKPOINT_EVERY = 50_000
DEFAULT_RESIZE_THRESHOLD = 2048

ENV_BUCKET = "MERMAID_CORALNET_BUCKET"
ENV_PREFIX = "MERMAID_CORALNET_PREFIX"
ENV_OUTPUT_PREFIX = "MERMAID_CORALNET_OUTPUT_PREFIX"
ENV_WORKERS = "MERMAID_CORALNET_ETL_WORKERS"
ENV_VERSION_OVERRIDE = "MERMAID_CORALNET_VERSION_OVERRIDE"
ENV_ANNOTATIONS_PATH = "MERMAID_CORALNET_ANNOTATIONS_PATH"
ENV_ANNOTATIONS_VERSION = "MERMAID_CORALNET_ANNOTATIONS_VERSION"


def get_bucket() -> str:
    return os.getenv(ENV_BUCKET, DEFAULT_BUCKET)


def get_prefix() -> str:
    return os.getenv(ENV_PREFIX, DEFAULT_PREFIX)


def get_output_s3_prefix() -> str:
    return os.getenv(ENV_OUTPUT_PREFIX, DEFAULT_OUTPUT_S3_PREFIX)


def get_workers(default: int = DEFAULT_WORKERS) -> int:
    raw = os.getenv(ENV_WORKERS)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
