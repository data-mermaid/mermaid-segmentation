"""Build the annotation-level CoralNet training parquet (resized URLs + scaled coords).

Joins the ETL annotations parquet onto the images parquet + resize checkpoint(s), resolves the
S3 key of the image to actually load (resized when completed, original for sub-threshold images),
rescales each point annotation to the loaded image's dimensions, and excludes images that
``needs_resize`` but whose resize did not complete. The result is consumed directly by
``CoralNetDataset``.

Usage (SageMaker):
    uv run python scripts/build_coralnet_training_parquet.py \
        --checkpoints outputs/resize_full_*.parquet outputs/resize_rgba_*.parquet
"""

from __future__ import annotations

import argparse
import logging

from mermaidseg.datasets.coralnet.etl.io import make_ibis_connection
from mermaidseg.datasets.coralnet.preprocessing.manifest import (
    build_training_manifest,
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
        help="Resize checkpoint parquets; the most recent resize_timestamp wins per image",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Annotations parquet (default: s3://{bucket}/etl-outputs/coralnet/{run}/coralnet_annotations_{run}.parquet)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Destination (default: s3://{bucket}/etl-outputs/coralnet/{run}/coralnet_training_resized_{run}.parquet)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}"
    annotations_uri = args.annotations or f"{base}/coralnet_annotations_{args.run}.parquet"
    images_uri = f"{base}/coralnet_images_{args.run}.parquet"
    out = args.out or f"{base}/coralnet_training_resized_{args.run}.parquet"

    con = make_ibis_connection()

    logger.info("Loading %d checkpoint(s)", len(args.checkpoints))
    checkpoints = [con.read_parquet(p) for p in args.checkpoints]
    for path, t in zip(args.checkpoints, checkpoints, strict=True):
        logger.info("  %s: %d rows", path, t.count().execute())
    combined = combine_checkpoints(checkpoints)
    logger.info("Combined checkpoint: %d rows", combined.count().execute())

    logger.info("Loading %s", annotations_uri)
    annotations = con.read_parquet(annotations_uri)
    logger.info("Loading %s", images_uri)
    images = con.read_parquet(images_uri)

    manifest = build_training_manifest(
        annotations=annotations,
        images=images,
        checkpoint=combined,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
    )

    n_annotations = annotations.count().execute()
    n_out = manifest.count().execute()
    n_resized = manifest.filter(manifest.uses_resized_image).count().execute()  # noqa: PD004 — ibis API
    logger.info(
        "Training manifest: %d annotation rows (%d excluded), %d on resized images",
        n_out,
        n_annotations - n_out,
        n_resized,
    )

    manifest.to_parquet(out)
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
