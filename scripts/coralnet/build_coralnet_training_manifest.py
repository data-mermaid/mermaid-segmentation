"""Build the CoralNet annotation-level training parquet (resized + original-below-
threshold).

Resolves each image to the key actually loaded at train time — the resized key when a resized
object exists on S3, the original ``s3_key`` for sub-threshold images — and scales each point
annotation's row/col to the loaded image's dimensions, so ``CoralNetDataset`` needs no runtime
scaling.

Resize-existence is read from the S3 LIST of ``<output_prefix>/resized/`` objects, NOT from the
resize checkpoint: historical checkpoints under-record completed resizes (uploads succeeded but the
status was never flushed), so the checkpoint would wrongly drop hundreds of thousands of images
that are genuinely resized on S3. The S3 object listing is the same source of truth the resize scan
trusts.

Usage:
    uv run python scripts/build_coralnet_training_manifest.py --run 20260623_nogit
"""

from __future__ import annotations

import argparse
import logging
import re

import boto3
import ibis
import pandas as pd

from mermaidseg.datasets.coralnet.preprocessing.manifest import build_training_manifest
from mermaidseg.datasets.coralnet.preprocessing.resize import _list_existing_resized_keys

logger = logging.getLogger("build_training_manifest")

# Matches resize._resized_s3_key_for: <prefix>/resized/s<source_id>/images/<image_id>.jpg
_KEY_RE = re.compile(r"/resized/s(\d+)/images/(.+)\.jpg$")


def _resized_existence_table(s3_client, bucket: str, output_prefix: str) -> pd.DataFrame:
    """A (source_id, image_id, status='completed') row for every resized object on
    S3."""
    keys = _list_existing_resized_keys(s3_client, bucket, output_prefix)
    rows = []
    for k in keys:
        m = _KEY_RE.search(k)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    df = pd.DataFrame(rows, columns=["source_id", "image_id"])
    df["status"] = "completed"
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--run", required=True, help="ETL version tag, e.g. 20260623_nogit")
    parser.add_argument("--output-prefix", default="dev/images", help="Prefix holding /resized/")
    parser.add_argument("--threshold", type=int, default=2048)
    parser.add_argument("--profile", default="mermaid-core", help="AWS profile (empty for default)")
    parser.add_argument("--out-key", default=None, help="Override output S3 key")
    parser.add_argument("--dry-run", action="store_true", help="Compute + report, but don't write")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3 = session.client("s3")
    storage_options = {"profile": args.profile} if args.profile else None

    base = f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}"
    ann = pd.read_parquet(
        f"{base}/coralnet_annotations_{args.run}.parquet", storage_options=storage_options
    )
    img = pd.read_parquet(
        f"{base}/coralnet_images_{args.run}.parquet", storage_options=storage_options
    )
    logger.info("Loaded annotations=%s images=%s", f"{len(ann):,}", f"{len(img):,}")

    n_needs = int(img["needs_resize"].sum())
    n_sub = int((~img["needs_resize"]).sum())
    n_nulldim = int((img["width"].isna() | img["height"].isna()).sum())
    logger.info(
        "Images: needs_resize=%s sub_threshold=%s null_dims=%s",
        f"{n_needs:,}",
        f"{n_sub:,}",
        f"{n_nulldim:,}",
    )

    logger.info(
        "Listing resized objects under s3://%s/%s/resized/ ...", args.bucket, args.output_prefix
    )
    ckpt = _resized_existence_table(s3, args.bucket, args.output_prefix)
    logger.info("Resized objects on S3: %s", f"{len(ckpt):,}")

    manifest = build_training_manifest(
        annotations=ibis.memtable(ann),
        images=ibis.memtable(img),
        checkpoint=ibis.memtable(ckpt),
        output_prefix=args.output_prefix,
        threshold=args.threshold,
    ).to_pandas()

    # --- data-quality numbers ---
    n_rows = len(manifest)
    imgs = manifest.drop_duplicates(["source_id", "image_id"])
    n_imgs = len(imgs)
    n_resized = int(imgs["uses_resized_image"].sum())
    n_original = n_imgs - n_resized
    ann_imgs = ann.drop_duplicates(["source_id", "image_id"])
    dropped_imgs = len(ann_imgs) - n_imgs
    dropped_rows = len(ann) - n_rows
    logger.info("=== Training manifest ===")
    logger.info("Annotation rows:        %s  (dropped %s)", f"{n_rows:,}", f"{dropped_rows:,}")
    logger.info("Distinct images:        %s  (dropped %s)", f"{n_imgs:,}", f"{dropped_imgs:,}")
    logger.info("  using RESIZED key:    %s", f"{n_resized:,}")
    logger.info("  using ORIGINAL key:   %s", f"{n_original:,}")

    if args.dry_run:
        logger.info("--dry-run: not writing")
        return 0

    out_key = (
        args.out_key
        or f"etl-outputs/coralnet/{args.run}/coralnet_training_manifest_{args.run}.parquet"
    )
    out_uri = f"s3://{args.bucket}/{out_key}"
    manifest.to_parquet(out_uri, storage_options=storage_options, index=False)
    logger.info("Wrote %s rows -> %s", f"{n_rows:,}", out_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
