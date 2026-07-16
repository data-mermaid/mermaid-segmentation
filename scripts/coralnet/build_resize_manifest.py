"""Build the resized-image manifest from one or more resize checkpoints.

Successive resize runs each produce their own checkpoint (e.g. the main run
plus an RGBA-remainder rerun); this merges them (most recent resize_timestamp
wins), joins the images parquet, and writes the manifest parquet for
downstream consumers.

Usage (SageMaker):
    uv run python scripts/build_resize_manifest.py \
        --checkpoints outputs/resize_full_*.parquet outputs/resize_rgba_*.parquet
"""

from __future__ import annotations

import argparse
import logging

from mermaidseg.datasets.coralnet.etl.io import make_ibis_connection
from mermaidseg.datasets.coralnet.preprocessing.manifest import (
    build_manifest,
    combine_checkpoints,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--run", default="20260526_807b611")
    parser.add_argument("--output-prefix", default="dev/images")
    parser.add_argument("--threshold", type=int, default=2048)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint parquets; the most recent resize_timestamp wins per image",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Manifest destination (default: s3://{bucket}/etl-outputs/coralnet/{run}/resize_manifest_{run}.parquet)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.out or (
        f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}/resize_manifest_{args.run}.parquet"
    )
    images_uri = (
        f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}/coralnet_images_{args.run}.parquet"
    )

    con = make_ibis_connection()

    logger.info("Loading %d checkpoint(s)", len(args.checkpoints))
    checkpoints = [con.read_parquet(p) for p in args.checkpoints]
    for path, t in zip(args.checkpoints, checkpoints, strict=True):
        logger.info("  %s: %d rows", path, t.count().execute())
    combined = combine_checkpoints(checkpoints)
    logger.info("Combined checkpoint: %d rows", combined.count().execute())
    logger.info("Status counts:\n%s", combined.status.value_counts().execute().to_string())

    logger.info("Loading %s", images_uri)
    images = con.read_parquet(images_uri)

    manifest = build_manifest(
        images=images,
        checkpoint=combined,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
    )
    logger.info("Manifest: %d rows", manifest.count().execute())

    manifest.to_parquet(out)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
