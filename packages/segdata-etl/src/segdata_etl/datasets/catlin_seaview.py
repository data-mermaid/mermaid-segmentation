"""``segdata build catlin_seaview``: stage the ingested Catlin Seaview prefix as a dataset version."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from segdata_etl.datasets import external

DATASET = "catlin_seaview"


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Catlin needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a new ``catlin_seaview`` version and print its version and manifest sha256.

    Raises:
        PublishRefused: The git tree is dirty or unpushed.
    """
    return external.build_point_dataset(
        args,
        client,
        dataset=DATASET,
        annotated_ids=lambda annotations: zip(
            annotations["region"], annotations["image_id"], strict=True
        ),
        what="catlin images",
    )
