"""Post-run completeness check for the CoralNet resize pipeline.

Tallies the resize checkpoint, lists what is actually on S3, reports the gap between the two, and
samples the images marked corrupted_invalid_channels to show what they really are (grayscale / RGBA
files are usually recoverable without redownload).

Usage (SageMaker):     uv run python scripts/check_resize_completeness.py
"""

from __future__ import annotations

import argparse
import io
import logging
from collections import Counter

import boto3
import ibis
import pandas as pd
from PIL import Image

from mermaidseg.datasets.coralnet.etl.io import make_ibis_connection
from mermaidseg.datasets.coralnet.preprocessing.resize import (
    scan_for_missing_resized_images,
)

logger = logging.getLogger(__name__)


def sample_invalid_channel_modes(
    df_bad: pd.DataFrame, bucket: str, sample_size: int
) -> Counter[str]:
    """GET a sample of invalid_channels images and tally their actual PIL format/mode."""
    s3 = boto3.client("s3")
    modes: Counter[str] = Counter()
    sample = df_bad.sample(min(sample_size, len(df_bad)), random_state=0)
    for row in sample.itertuples(index=False):
        try:
            body = s3.get_object(Bucket=bucket, Key=row.s3_key)["Body"].read()
            img = Image.open(io.BytesIO(body))
            modes[f"{img.format}/{img.mode}"] += 1
        except Exception as e:  # noqa: BLE001 - tally unreadables, don't crash
            modes[f"unreadable: {type(e).__name__}"] += 1
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="dev-datamermaid-sm-sources")
    parser.add_argument("--run", default="20260526_807b611")
    parser.add_argument("--output-prefix", default="dev/images")
    parser.add_argument("--threshold", type=int, default=2048)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint parquet (default: outputs/resize_full_{run}.parquet)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=25,
        help="How many invalid_channels images to download and inspect",
    )
    parser.add_argument(
        "--gap-out",
        default="outputs/unexplained_missing.parquet",
        help="Where to write images missing on S3 and not marked corrupted",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    checkpoint_uri = args.checkpoint or f"outputs/resize_full_{args.run}.parquet"
    images_uri = (
        f"s3://{args.bucket}/etl-outputs/coralnet/{args.run}/coralnet_images_{args.run}.parquet"
    )

    # DuckDB httpfs reads S3 via the credential chain, so this works on SageMaker without s3fs.
    con = make_ibis_connection()

    print(f"== Checkpoint: {checkpoint_uri}")
    ckpt = con.read_parquet(checkpoint_uri)
    n_ckpt = ckpt.count().execute()
    print(f"rows: {n_ckpt:,}")
    print(ckpt.status.value_counts().execute().to_string(index=False))
    skipped = ckpt.filter(ckpt.status == "skipped")
    if skipped.count().execute():
        print("\n== Skip reasons")
        print(skipped.skip_reason.value_counts().execute().to_string(index=False))

    print(f"\n== Loading {images_uri}")
    images = con.read_parquet(images_uri)
    n_needs_resize = int(images.needs_resize.sum().execute())
    print(f"needs_resize: {n_needs_resize:,}; checkpoint covers {n_ckpt:,}")

    print("\n== Scanning S3 for missing resized images (1-2 min)")
    todo = scan_for_missing_resized_images(
        df_images=images.to_pandas(),
        bucket=args.bucket,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
    )
    print(f"still missing on S3: {len(todo):,}")

    # Missing on S3 but never flagged corrupted — the genuinely unexplained gap.
    skipped_keys = skipped.select("source_id", "image_id")
    gap = ibis.memtable(todo).anti_join(skipped_keys, ["source_id", "image_id"]).to_pandas()
    print(f"missing AND not marked corrupted: {len(gap):,}")
    if len(gap):
        gap.to_parquet(args.gap_out, index=False)
        print(f"wrote {args.gap_out}")

    df_bad = (
        skipped.filter(skipped.skip_reason == "corrupted_invalid_channels")
        .inner_join(images.select("source_id", "image_id", "s3_key"), ["source_id", "image_id"])
        .to_pandas()
    )
    if len(df_bad):
        print(
            f"\n== Sampling {min(args.sample, len(df_bad))} of {len(df_bad):,} invalid_channels images"
        )
        for key, count in sample_invalid_channel_modes(
            df_bad, args.bucket, args.sample
        ).most_common():
            print(f"{count:3d}  {key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
