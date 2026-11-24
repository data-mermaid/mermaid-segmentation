"""The ``--store`` option shared by the publishing commands: a local directory or S3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from segdata_etl.common.local_store import LocalStore
from segdata_spec import DATASET_BUCKET

DEFAULT_STORE = Path("outputs") / "segdata-store"
DEFAULT_CACHE_DIR = Path("outputs") / "segdata-cache"
_S3 = "s3"


def add_store_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--store``: ``s3`` or a directory, defaulting to ``DEFAULT_STORE``."""
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=(
            "Where to publish: a local directory (the default, %(default)s) or 's3'. "
            "Publishing to S3 must be requested explicitly."
        ),
    )


def open_store(store: str | Path, *, source_client: Any) -> tuple[Any, str]:
    """Return ``(client, bucket)`` for ``store``.

    ``"s3"`` writes through ``source_client``. A directory gets a ``LocalStore`` laid out as
    ``<dir>/<bucket>/<key>``, with ``source_client`` serving its ``copy_object`` sources.
    """
    if str(store) == _S3:
        return source_client, DATASET_BUCKET
    return LocalStore(Path(store), source_client=source_client), DATASET_BUCKET
