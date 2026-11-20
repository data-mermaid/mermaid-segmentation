#!/usr/bin/env python3
r"""Write per-image label counts for the train splits of a data config.

The parquet has one row per (dataset, image, label) in every train split of ``--data-config``,
with the columns ``dataset``, ``image_id``, ``label`` and ``n``. It feeds
``scripts/build_concept_removal.py --label-counts``.

    python scripts/train_label_counts.py --data-config configs/data_config_trainval.yaml \
        --output dataset_stats/label_counts_trainval.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mermaidseg.dataset_reconciliation.label_counts import count_train_labels
from mermaidseg.datasets import build_datasets, setup_local_cache
from mermaidseg.io import setup_config

# Val/test transform size for scripts that only read annotations; the transformed pixels are unused.
_UNUSED_INPUT_SIZE = (256, 256)


def main() -> None:
    """Parse arguments, count the train labels and write the parquet."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--cache-dir", default=None, help="local S3 cache root (default: disabled)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = setup_config({"data": str(args.data_config)})
    if args.cache_dir is not None:
        cfg.data.local_cache_dir = args.cache_dir
    setup_local_cache(cfg.data)
    built = build_datasets(
        cfg.data, input_size=_UNUSED_INPUT_SIZE, names=args.datasets, splits=["train"]
    )
    out = count_train_labels(built)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(
        out.groupby("dataset").agg(
            images=("image_id", "nunique"), labels=("label", "nunique"), n=("n", "sum")
        )
    )
    print(f"Wrote {args.output} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
