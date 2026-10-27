#!/usr/bin/env python3
"""Count the CoralNet training images in the upper tails of the source quality metrics.

Takes one quantile per metric of ``scripts/coralnet_source_report.py``, in this order:

    images, annotated pixels, source labels, genera, coral genera, avg pixels per image

A quantile ``q > 0`` flags every training source whose value is at or above the
``1 - q`` quantile of that metric over the training sources: ``0.1`` is the top 10%,
and ``0`` turns the metric off. All sources tied at the threshold are flagged. A source
is flagged when any metric flags it, and then every one of its images is affected.

Only CoralNet training sources (``data.coralnet.train.whitelist_sources`` in
``--data-config``, exactly what ``CoralNetDataset`` trains on) enter the quantiles and
the counts. Val, test, unlisted sources and other datasets never do.

Per-source stats come from the training manifest, loaded as in the source report, or
with ``--stats-csv`` from a ``source_stats.csv`` written by the report or a
``source_flags.csv`` written by this script (no S3 access needed).

With ``--output-dir``:

    <output-dir>/
      source_flags.csv         every training source: 6 metrics, flag_<metric> per active
                               metric, and flagged (any of them)
      flagged_images.parquet   every image of a flagged source (manifest mode only)
      run.json                 arguments, quantiles, thresholds and the printed summary

Usage:
    python scripts/coralnet_quantile_exploration.py --quantiles 0 0 0 0 0.1 0.01
    python scripts/coralnet_quantile_exploration.py --quantiles 0 0 0 0 0.1 0.01 \\
        --stats-csv coralnet_source_report/source_stats.csv
    python scripts/coralnet_quantile_exploration.py --quantiles 0 0 0 0 0.1 0.01 --output-dir flags
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from coralnet_source_report import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_CONCEPTS_CSV,
    DEFAULT_DATA_CONFIG,
    METRICS,
    compute_source_stats,
    configure_cache,
    load_annotations,
    resolve_cache_dir,
)

logger = logging.getLogger("coralnet_quantile_exploration")

METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS)
OVERLAP = "in 2+ criteria"
UNION = "any (union)"
SUMMARY_COLUMNS: tuple[str, ...] = (
    "criterion",
    "q",
    "threshold",
    "min_value",
    "n_sources",
    "pct_sources",
    "n_images",
    "pct_images",
    "n_pixels",
    "pct_pixels",
)
IMAGES_FILE = "flagged_images.parquet"
UNSUPPORTED_TRAIN_KEYS = frozenset({"blacklist_sources", "class_subset"})


def load_train_sources(data_config: Path) -> list[int]:
    """CoralNet source ids that ``CoralNetDataset`` trains on under ``data_config``."""
    cfg = yaml.safe_load(data_config.read_text())
    train = ((cfg.get("data") or {}).get("coralnet") or {}).get("train")
    if not isinstance(train, dict) or not train.get("whitelist_sources"):
        raise ValueError(f"{data_config} has no data.coralnet.train.whitelist_sources")
    unsupported = sorted(UNSUPPORTED_TRAIN_KEYS & set(train))
    if unsupported:
        raise ValueError(f"{data_config}: data.coralnet.train.{unsupported} is not supported")
    return sorted({int(s) for s in train["whitelist_sources"]})


def stats_from_manifest(
    args: argparse.Namespace, train_ids: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-source metrics and point annotations for the training sources in the manifest."""
    cache_dir = resolve_cache_dir(args.data_config, args.cache_dir)
    if cache_dir and not Path(cache_dir).is_dir():
        logger.info("cache dir %s does not exist; reading from S3 only", cache_dir)
        cache_dir = None
    configure_cache(cache_dir)

    annotations = load_annotations(args.bucket, args.manifest, args.concepts_csv, train_ids)
    if annotations.empty:
        raise SystemExit("no annotations for the training sources")
    stats = compute_source_stats(annotations, dict.fromkeys(train_ids, "train"))
    missing = sorted(set(train_ids) - set(stats.index))
    if missing:
        logger.warning(
            "%d training sources have no annotations in the manifest: %s", len(missing), missing
        )
    return stats[list(METRIC_KEYS)].sort_index(), annotations


def stats_from_csv(path: Path, train_ids: list[int]) -> pd.DataFrame:
    """Per-source metrics of the training sources, read from a precomputed CSV."""
    table = pd.read_csv(path, index_col="source_id")
    table.index = table.index.astype(int)
    missing_columns = [key for key in METRIC_KEYS if key not in table.columns]
    if missing_columns:
        raise ValueError(f"{path} has no column(s) {missing_columns}")
    missing = sorted(set(train_ids) - set(table.index))
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} of {len(train_ids)} training sources "
            f"(e.g. {missing[:10]}); rebuild it from the manifest"
        )
    return table.loc[train_ids, list(METRIC_KEYS)].sort_index()


def quantile_thresholds(stats: pd.DataFrame, quantiles: dict[str, float]) -> dict[str, float]:
    """Lower bound of the top-``q`` tail of every metric with ``q > 0``."""
    return {key: float(stats[key].quantile(1.0 - q)) for key, q in quantiles.items() if q > 0}


def flag_column(key: str) -> str:
    return f"flag_{key}"


def flag_sources(stats: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """A ``flag_<metric>`` column per thresholded metric, plus ``flagged`` (any of them)."""
    flags = pd.DataFrame(
        {flag_column(key): stats[key] >= threshold for key, threshold in thresholds.items()},
        index=stats.index,
    )
    flags["flagged"] = flags.any(axis=1)
    return flags


def _pct(n: float, total: float) -> float:
    return 100.0 * n / total if total else 0.0


def _counts(stats: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    n_images = int(stats.loc[mask, "n_images"].sum())
    n_pixels = int(stats.loc[mask, "n_pixels"].sum())
    return {
        "n_sources": int(mask.sum()),
        "pct_sources": _pct(mask.sum(), len(stats)),
        "n_images": n_images,
        "pct_images": _pct(n_images, stats["n_images"].sum()),
        "n_pixels": n_pixels,
        "pct_pixels": _pct(n_pixels, stats["n_pixels"].sum()),
    }


def summarize(
    stats: pd.DataFrame,
    flags: pd.DataFrame,
    quantiles: dict[str, float],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Sources, images and pixels per criterion, in 2+ criteria, and in the union."""
    rows = []
    for key, threshold in thresholds.items():
        mask = flags[flag_column(key)]
        flagged_values = stats.loc[mask, key]
        rows.append(
            {
                "criterion": key,
                "q": quantiles[key],
                "threshold": threshold,
                "min_value": float(flagged_values.min()) if len(flagged_values) else None,
                **_counts(stats, mask),
            }
        )
    if len(thresholds) >= 2:
        in_several = flags[[flag_column(key) for key in thresholds]].sum(axis=1) >= 2
        rows.append({"criterion": OVERLAP, **_counts(stats, in_several)})
    rows.append({"criterion": UNION, **_counts(stats, flags["flagged"])})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def flagged_images(annotations: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """One row per image of a flagged source, with that source's flags."""
    flagged_ids = flags.index[flags["flagged"]]
    images = (
        annotations.loc[
            annotations["source_id"].isin(flagged_ids), ["source_id", "image_id", "image_key"]
        ]
        .drop_duplicates(["source_id", "image_id"])
        .astype({"image_key": object})
        .reset_index(drop=True)
    )
    return images.join(flags, on="source_id")


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def format_summary(summary: pd.DataFrame, stats: pd.DataFrame, data_config: Path) -> str:
    header = (
        f"CoralNet training ({_display(data_config)}): {len(stats):,} sources, "
        f"{int(stats['n_images'].sum()):,} images, "
        f"{int(stats['n_pixels'].sum()):,} annotated pixels"
    )
    table = [("criterion", "q", "threshold", "sources", "images", "pixels")]
    for row in summary.itertuples(index=False):
        table.append(
            (
                row.criterion,
                "" if pd.isna(row.q) else f"{row.q:g}",
                "" if pd.isna(row.min_value) else f">= {_number(row.min_value)}",
                f"{row.n_sources:,} ({row.pct_sources:.1f}%)",
                f"{row.n_images:,} ({row.pct_images:.1f}%)",
                f"{row.n_pixels:,} ({row.pct_pixels:.1f}%)",
            )
        )
    widths = [max(len(line[i]) for line in table) for i in range(len(table[0]))]
    lines = [header]
    for line in table:
        cells = [line[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(line[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    stats: pd.DataFrame,
    flags: pd.DataFrame,
    annotations: pd.DataFrame | None,
    run_info: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.join(flags).to_csv(out_dir / "source_flags.csv")
    images_path = out_dir / IMAGES_FILE
    if annotations is not None:
        flagged_images(annotations, flags).to_parquet(images_path, index=False)
    elif images_path.exists():
        images_path.unlink()
        logger.info("removed stale %s (the image list needs manifest mode)", images_path)
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    logger.info("wrote %s", out_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from mermaidseg.datasets.coralnet.coralnet_dataset import _resolve_default_manifest_path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=len(METRIC_KEYS),
        required=True,
        metavar="Q",
        help=f"Upper-tail fraction per metric, in [0, 1], in order: {' '.join(METRIC_KEYS)}.",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Read per-source metrics from this CSV instead of the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--manifest",
        default=_resolve_default_manifest_path(),
        help="Manifest key in --bucket, s3:// URI, or local parquet path.",
    )
    parser.add_argument("--bucket", default=os.getenv("MERMAID_CORALNET_BUCKET", DEFAULT_BUCKET))
    parser.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local S3 cache root (default: $MERMAIDSEG_LOCAL_CACHE_DIR or the data config).",
    )
    args = parser.parse_args(argv)
    if not all(0.0 <= q <= 1.0 for q in args.quantiles):
        parser.error(f"--quantiles must be in [0, 1], got {args.quantiles}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    quantiles = dict(zip(METRIC_KEYS, args.quantiles, strict=True))
    train_ids = load_train_sources(args.data_config)
    logger.info("%d CoralNet training sources in %s", len(train_ids), args.data_config)

    annotations: pd.DataFrame | None = None
    if args.stats_csv is not None:
        stats = stats_from_csv(args.stats_csv, train_ids)
    else:
        stats, annotations = stats_from_manifest(args, train_ids)

    thresholds = quantile_thresholds(stats, quantiles)
    flags = flag_sources(stats, thresholds)
    summary = summarize(stats, flags, quantiles, thresholds)
    print(format_summary(summary, stats, args.data_config))

    if args.output_dir is not None:
        run_info = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "n_train_sources_in_config": len(train_ids),
            "n_train_sources_with_stats": len(stats),
            "quantiles": quantiles,
            "thresholds": thresholds,
            "summary": json.loads(summary.to_json(orient="records")),
        }
        write_outputs(args.output_dir, stats, flags, annotations, run_info)


if __name__ == "__main__":
    main()
