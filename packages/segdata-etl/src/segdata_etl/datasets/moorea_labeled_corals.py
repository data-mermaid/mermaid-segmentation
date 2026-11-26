"""``segdata build moorea_labeled_corals``: stage the ingested Moorea prefix as a dataset version."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from segdata_etl.datasets import external

DATASET = "moorea_labeled_corals"


def add_arguments(parser: argparse.ArgumentParser, *, workdir: Path) -> None:
    """Add the shared external-builder options; Moorea needs no others and no ``workdir``."""
    del workdir
    external.add_build_arguments(parser)


def run(args: argparse.Namespace, client: Any) -> int:
    """Stage a new ``moorea_labeled_corals`` version and print its version and manifest sha256.

    Points take the case-merged ``source_label_name``; the licence comes from the ingest manifest.

    Raises:
        PublishRefused: The git tree is dirty or unpushed.
    """
    return external.build_point_dataset(
        args,
        client,
        dataset=DATASET,
        annotated_ids=lambda annotations: zip(
            annotations["year"].astype(str), annotations["image_id"].astype(str), strict=True
        ),
        what="moorea images",
    )
