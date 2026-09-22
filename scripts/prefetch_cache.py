#!/usr/bin/env python
"""Warm the local S3 cache so a training run can execute fully offline (no AWS).

This is the *only* step that needs AWS credentials. It constructs every
configured ``(dataset, split)`` (which downloads all manifests / annotation
parquets / label maps) and then reads every image (and, where applicable, every
dense label) through :class:`LocalS3Cache`, write-through to the local cache
directory. Once this succeeds, the GPU training job can run with
``MERMAIDSEG_S3_OFFLINE=1`` and no credentials at all.

Example::

    python scripts/prefetch_cache.py \
        --data-config configs/data_config_all.yaml \
        --training-config configs/training_config_cbm.yaml \
        --workers 32
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mermaidseg.datasets import build_datasets, setup_local_cache
from mermaidseg.datasets.local_cache import LocalS3Cache
from mermaidseg.io import setup_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("prefetch_cache")

# Cap the number of individual load failures we keep per dataset (avoid unbounded
# memory when a whole source is misconfigured).
_MAX_FAILURES_KEPT = 50


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        default="configs/data_config_all.yaml",
        help="Path to the data config YAML.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training_config_cbm.yaml",
        help="Path to the training config YAML (used for the `padding` value).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restrict to these dataset names (default: all configured).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Restrict to these split names, e.g. train val (default: all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent download threads per dataset.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first load failure instead of tallying and continuing.",
    )
    parser.add_argument(
        "--hf-models",
        nargs="*",
        default=None,
        help="Optionally snapshot_download these HF model repos into HF_HOME "
        "(e.g. facebook/dinov3-vitl16-pretrain-lvd1689m) so training can also "
        "run with HF_HUB_OFFLINE=1.",
    )
    return parser.parse_args(argv)


def _warm_dataset(ds, workers: int, fail_fast: bool) -> tuple[int, list[str]]:
    """Read every image (and dense label, if any) of ``ds`` through the cache.

    Returns ``(num_attempted, failures)`` where ``failures`` is a truncated list
    of ``"<idx>: <error>"`` strings.
    """
    df_images = getattr(ds, "df_images", None)
    if df_images is None:
        # HF-backed datasets (e.g. Coralscapes V2) are fully materialized into
        # HF_HOME at construction time; nothing more to warm from S3.
        logger.info("  (no df_images — warmed at construction, skipping row scan)")
        return 0, []

    has_label = hasattr(ds, "read_label")
    rows = df_images.to_dict("records")
    failures: list[str] = []

    def _warm(index_row: tuple[int, dict]) -> None:
        _idx, row = index_row
        ds.read_image(**row)
        if has_label:
            ds.read_label(**row)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_warm, (i, row)): i for i, row in enumerate(rows)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 — tally and continue by default
                if fail_fast:
                    raise
                if len(failures) < _MAX_FAILURES_KEPT:
                    failures.append(f"{idx}: {type(e).__name__}: {e}")
            if done % 1000 == 0:
                logger.info("  ... %d/%d rows", done, len(rows))

    return len(rows), failures


def _download_hf_models(repos: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for repo in repos:
        logger.info("Downloading HF model repo %s into HF_HOME ...", repo)
        snapshot_download(repo_id=repo)
        logger.info("  done: %s", repo)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = setup_config(
        {"data": args.data_config, "training": args.training_config}
    )
    padding = cfg.training.get("padding") if hasattr(cfg, "training") else None

    setup_local_cache(cfg.data)
    cache = LocalS3Cache.get()
    if cache.offline:
        logger.error(
            "MERMAIDSEG_S3_OFFLINE is set — prefetch needs S3 access. Unset it "
            "before running this script."
        )
        return 2
    if not cache.enabled:
        logger.error(
            "Local cache is not configured (data.local_cache_dir is empty). "
            "Prefetch has nowhere to write."
        )
        return 2

    logger.info("Building datasets (this warms manifests / annotations / label maps) ...")
    datasets = build_datasets(
        cfg.data,
        padding=padding,
        names=args.datasets,
        splits=args.splits,
        verbose=True,
    )
    if not datasets:
        logger.error("No datasets matched the given --datasets/--splits filters.")
        return 2

    total_failures = 0
    started = time.time()
    for (name, split), ds in datasets.items():
        t0 = time.time()
        logger.info("Warming %s/%s ...", name, split)
        try:
            attempted, failures = _warm_dataset(ds, args.workers, args.fail_fast)
        except Exception as e:  # noqa: BLE001
            logger.error("Aborting on %s/%s (fail-fast): %s", name, split, e)
            return 1

        stats = cache.snapshot_stats()
        elapsed = time.time() - t0
        logger.info(
            "  %s/%s: attempted=%d local_hits=%d s3_fetches=%d failures=%d (%.1fs)",
            name,
            split,
            attempted,
            stats.local_hits,
            stats.s3_fetches,
            len(failures),
            elapsed,
        )
        for f in failures:
            logger.warning("    failure %s", f)
        total_failures += len(failures)

    if args.hf_models:
        _download_hf_models(args.hf_models)

    logger.info(
        "Prefetch complete in %.1fs. Total load failures: %d",
        time.time() - started,
        total_failures,
    )
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
