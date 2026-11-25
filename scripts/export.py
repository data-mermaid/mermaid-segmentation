"""Export a trained checkpoint into a self-describing bundle (``model.json`` + weights)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import boto3

from mermaidseg.export import export_bundle


def main(argv: list[str] | None = None, *, s3_client: Any = None) -> None:
    """Export ``--checkpoint`` into ``--output-dir`` and print the bundle path.

    The run's pinned manifests are checked through ``s3_client`` (default: a boto3 S3 client),
    and its commit against the ``origin/*`` refs of the working directory's checkout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    client = boto3.client("s3") if s3_client is None else s3_client
    print(export_bundle(args.checkpoint, args.output_dir, s3_client=client))


if __name__ == "__main__":
    main()
