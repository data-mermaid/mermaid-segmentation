#!/usr/bin/env python3
"""Per-image source-label counts for the training splits of a data config.

Writes one row per ``(dataset, image, source label)`` in every ``train`` split of ``--data-config``:
``dataset``, ``image_id``, ``label`` (lowercased source label), ``n`` (points for sparse datasets, pixels for dense
ones). Map ``(dataset, label)`` to concepts with a ``class_to_concepts.csv`` afterwards (its
``source_dataset_source`` / ``source_label_class_name`` columns) to count, per taxon, the annotations or the
training images in which it is annotated.

Data loading follows ``scripts/dataset_table_stats.py``: the same cached data as training (image blacklist and
annotation overrides applied by the dataset classes), no crops or transforms, so counts are native annotations.
Ignore labels (pixel id 0, ``unlabeled``, ``unlabeled / unidentified``) are dropped. Set ``MERMAIDSEG_S3_OFFLINE=1``
and ``HF_HUB_OFFLINE=1`` so a cache miss fails instead of downloading.

Usage:
    python scripts/train_label_counts.py \
        --data-config /data/vision/beery/scratch/sauder/mermaid-segmentation-final/configs/data_config_final_768.yaml \
        --output dataset_stats/train_label_counts_final_768.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_table_stats import (  # noqa: E402
    _SPEC_BY_KEY,
    _dense_id_to_name,
    _is_ignore_name,
    _iter_dense_masks,
    _n_images,
    as_label_array,
    strip_transforms,
)

logger = logging.getLogger("train_label_counts")


def sparse_rows(key: str, ds) -> pd.DataFrame:
    ann = ds.df_annotations[["image_id", "source_label_name"]].copy()
    ann["label"] = ann.source_label_name.astype(str).str.strip().str.lower()
    ann = ann[~ann.label.map(_is_ignore_name)]
    out = ann.groupby(["image_id", "label"], observed=True).size().rename("n").reset_index()
    out.insert(0, "dataset", key)
    return out


def dense_rows(key: str, ds) -> pd.DataFrame:
    id2name = _dense_id_to_name(key, ds)
    ids = ds.df_images.image_id.tolist() if key == "benthos_yuval" else list(range(_n_images(key, ds)))
    rows = []
    for i, mask in enumerate(_iter_dense_masks(key, ds)):
        vals, counts = np.unique(as_label_array(mask), return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            name = id2name.get(int(v), f"id_{v}")
            if v > 0 and not _is_ignore_name(name):
                rows.append((key, str(ids[i]), name.strip().lower(), int(c)))
        if (i + 1) % 500 == 0:
            logger.info("%s: %d masks", key, i + 1)
    if len(rows) and i + 1 != len(ids):
        raise RuntimeError(f"{key}: read {i + 1} masks, expected {len(ids)}")
    return pd.DataFrame(rows, columns=["dataset", "image_id", "label", "n"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from mermaidseg.datasets import build_datasets, setup_local_cache
    from mermaidseg.io import setup_config

    data_cfg = strip_transforms(setup_config({"data": str(args.data_config)}).data)
    setup_local_cache(data_cfg)
    built = build_datasets(data_cfg, padding=None, names=args.datasets, splits=["train"], verbose=False)
    parts = []
    for (key, split), ds in built.items():
        if key not in _SPEC_BY_KEY:
            logger.info("skipping %s", key)
            continue
        df = sparse_rows(key, ds) if _SPEC_BY_KEY[key].kind == "sparse" else dense_rows(key, ds)
        df["image_id"] = df.image_id.astype(str)
        logger.info("%s %s: %d images, %d labels, n=%d", key, split, df.image_id.nunique(), df.label.nunique(),
                    df.n.sum())
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(out.groupby("dataset").agg(images=("image_id", "nunique"), labels=("label", "nunique"), n=("n", "sum")))
    print(f"Wrote {args.output} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
