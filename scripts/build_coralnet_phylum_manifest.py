#!/usr/bin/env python3
"""Write a CoralNet training manifest whose quantile-flagged sources keep only coarse labels.

Sources are flagged exactly as in ``scripts/coralnet_quantile_exploration.py`` (same
``--quantiles``, ``--stats-csv``, ``--data-config``): only CoralNet training sources can be
flagged. In every image of a flagged source, each point whose label is more detailed than its
phylum is relabelled with the phylum-level CoralNet label from ``class_to_concepts.csv``:

* cnidaria keep their condition: ``live == FALSE`` -> ``dead``, ``bleached == TRUE`` ->
  ``condition_fully bleached``, anything else -> ``coral`` (other coral alive);
* other phyla map to one canonical label (porifera -> ``sponge``, ...);
* lineages without a phylum-level label map to the coarsest label of their class or order
  (phaeophyceae -> ``brown algae, unidentified``, alismatales -> ``seagrass``, ...).

Labels that are already that coarse, non-taxonomic labels (sand, turf, ...), labels without a
CoralNet row in the CSV, and points of unflagged, val and test sources keep their label. A
relabel rewrites ``coralnet_id``, so the class targets and the concepts both follow the new
label; the old id stays in ``original_coralnet_id``.

``data.coralnet.train.annotation_overrides`` and ``image_blacklist`` from ``--data-config`` are
applied first, in that order, matching ``CoralNetDataset``. Source quantiles and the relabel
both see that filtered manifest. The written parquet already contains the override points, so a
training profile must not set ``annotation_overrides`` again.

    <output-dir>/
      coralnet_training_manifest_phylum.parquet  full manifest, flagged sources relabelled
      source_flags.csv                           every training source: metrics and flags
      label_replacements.csv                     every CoralNet label in the CSV: why and to what
      label_changes.csv                          relabelled points per (label, replacement)
      run.json                                   arguments, thresholds, flagged sources, counts

Point the training split at it with ``data.coralnet.train.manifest_path: <parquet path>``.

Usage:
    python scripts/build_coralnet_phylum_manifest.py --quantiles 0 0 0 0 0.1 0.01 \\
        --output-dir data_quality/coralnet_phylum
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from coralnet_quantile_exploration import (  # noqa: E402
    METRIC_KEYS,
    flag_sources,
    format_summary,
    load_train_sources,
    quantile_thresholds,
    stats_from_csv,
    stats_from_manifest,
    summarize,
)

from coralnet_source_report import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_CONCEPTS_CSV,
    DEFAULT_DATA_CONFIG,
    ID2NAME_KEY,
    MANIFEST_COLUMNS,
    MISSING_TAXON,
    configure_cache,
    resolve_cache_dir,
)

logger = logging.getLogger("build_coralnet_phylum_manifest")

MANIFEST_FILE = "coralnet_training_manifest_phylum.parquet"
TAXON_RANKS: tuple[str, ...] = ("phylum", "class", "order", "family", "genus", "species")
CNIDARIA = "cnidaria"
CNIDARIA_LABELS = {"alive": "coral", "dead": "dead", "bleached": "condition_fully bleached"}
COARSE_LABELS: dict[tuple[str, str], str] = {
    ("phylum", "porifera"): "sponge",
    ("phylum", "rhodophyta"): "red algae",
    ("phylum", "chordata"): "tunicate",
    ("phylum", "mollusca"): "molluscs",
    ("phylum", "echinodermata"): "echinoderms",
    ("phylum", "bryozoa"): "bryozoan",
    ("phylum", "cyanobacteriota"): "cyanobacteria",
    ("phylum", "retaria"): "foraminifera",
    # Lineages without a phylum-level CoralNet label.
    ("class", "phaeophyceae"): "brown algae, unidentified",
    ("class", "chrysophyceae"): "chrysophyta",
    ("class", "thecostraca"): "barnacles",
    ("class", "polychaeta"): "polychaeta",
    # Chlorophyta's coarsest row is genus-level; this label's CSV row has no phylum.
    ("class", "ulvophyceae"): "green algae, unidentified",
    ("order", "alismatales"): "seagrass",
}
# Detailed labels whose lineage has no coarser CoralNet label; they keep their label.
UNCHANGED_DEEP_LABELS = frozenset(
    {"chondrilla spp.", "chrysocystis fragilis", "palmophyllum crassum", "verrucaria"}
)

COLLAPSED = "collapsed"
ALREADY_COARSE = "already_coarse"
NON_TAXONOMIC = "non_taxonomic"
NO_COARSE_LABEL = "no_coarse_label"


def load_concept_rows(path: Path) -> pd.DataFrame:
    """CoralNet rows of ``class_to_concepts.csv``: lowercased label index, rank and state columns."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame["source_dataset_source"].str.strip().str.lower() == "coralnet"]
    columns = [*TAXON_RANKS, "live", "bleached"]
    out = frame[columns].apply(lambda col: col.str.strip().str.lower())
    out.index = pd.Index(frame["source_label_class_name"].str.strip().str.lower(), name="label")
    return out[~out.index.duplicated()]


def _coarse_label(row: pd.Series, lineage: dict[str, str]) -> tuple[str, str] | None:
    """``(rank, label)`` of the coarse label this lineage collapses to, if any."""
    if lineage.get("phylum") == CNIDARIA:
        if row["live"] == "false":
            return "phylum", CNIDARIA_LABELS["dead"]
        if row["bleached"] == "true":
            return "phylum", CNIDARIA_LABELS["bleached"]
        return "phylum", CNIDARIA_LABELS["alive"]
    for rank in ("phylum", "class", "order"):
        label = COARSE_LABELS.get((rank, lineage.get(rank, "")))
        if label is not None:
            return rank, label
    return None


def classify(row: pd.Series) -> tuple[str, str | None]:
    """``(reason, replacement)`` for one CSV row; ``replacement`` is set only when collapsed."""
    lineage = {rank: row[rank] for rank in TAXON_RANKS if row[rank] not in MISSING_TAXON}
    if not lineage:
        return NON_TAXONOMIC, None
    if "phylum" in lineage and len(lineage) == 1:
        return ALREADY_COARSE, None
    coarse = _coarse_label(row, lineage)
    if coarse is None:
        return NO_COARSE_LABEL, None
    rank, label = coarse
    finer = TAXON_RANKS[TAXON_RANKS.index(rank) + 1 :]
    if not any(r in lineage for r in finer):
        return ALREADY_COARSE, None
    return COLLAPSED, label


def replacement_table(concepts: pd.DataFrame) -> pd.DataFrame:
    """Reason and replacement for every CoralNet label, after checking the curated tables."""
    decisions = [classify(row) for _, row in concepts.iterrows()]
    table = concepts.assign(
        reason=[reason for reason, _ in decisions],
        replacement=[replacement for _, replacement in decisions],
    )

    targets = set(CNIDARIA_LABELS.values()) | set(COARSE_LABELS.values())
    missing = sorted(targets - set(table.index))
    if missing:
        raise ValueError(f"replacement label(s) {missing} have no CoralNet row in the CSV")
    not_coarse = sorted(t for t in targets if table.loc[t, "reason"] == COLLAPSED)
    if not_coarse:
        raise ValueError(f"replacement label(s) {not_coarse} would themselves be collapsed")
    uncovered = sorted(set(table.index[table["reason"] == NO_COARSE_LABEL]) - UNCHANGED_DEEP_LABELS)
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} detailed label(s) have no coarse label: {uncovered[:20]}; "
            "extend COARSE_LABELS or UNCHANGED_DEEP_LABELS"
        )
    return table


def name_to_id(id2name: dict[str, str]) -> dict[str, int]:
    """Lowercased CoralNet name -> smallest id with that name (as ``CoralNetDataset`` lowercases)."""
    out: dict[str, int] = {}
    for key, name in id2name.items():
        lowered = str(name).lower()
        out[lowered] = min(int(key), out.get(lowered, int(key)))
    return out


def _category_names(codes: pd.Series, id2name: dict[str, str]) -> pd.Series:
    """Lowercased label name per category of a categorical ``coralnet_id`` column."""
    return pd.Series(codes.cat.categories).map(lambda x: str(id2name.get(str(x))).lower())


def points_without_concepts(
    manifest: pd.DataFrame, id2name: dict[str, str], labels: pd.Index, sources: list[int]
) -> int:
    """Points of ``sources`` whose label has no CoralNet row in the CSV (left unchanged)."""
    codes = manifest.loc[manifest["source_id"].isin(sources), "coralnet_id"].astype("category")
    known = _category_names(codes, id2name).isin(labels).to_numpy()
    idx = codes.cat.codes.to_numpy()
    return int((~known[idx[idx >= 0]]).sum() + (idx < 0).sum())


def _ids_like(ids: np.ndarray, column: pd.Series) -> np.ndarray:
    """Integer ids in ``column``'s representation; the loader looks them up as ``str(id)``."""
    dtype = column.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return ids.astype(np.int64)
    if pd.api.types.is_string_dtype(dtype):
        return np.array([str(i) for i in ids], dtype=object)
    raise TypeError(f"coralnet_id must hold integers or digit strings, got {column.dtype}")


def collapse_manifest(
    manifest: pd.DataFrame,
    id2name: dict[str, str],
    replacements: dict[str, str],
    flagged_sources: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel points of flagged sources; return the manifest and one row per changed point."""
    name2id = name_to_id(id2name)
    unknown = sorted(set(replacements.values()) - set(name2id))
    if unknown:
        raise ValueError(f"replacement label(s) {unknown} are not in coralnet_id2name.json")

    old_ids = manifest["coralnet_id"]
    codes = old_ids.astype("category")
    names = _category_names(codes, id2name)
    new_names = names.map(replacements)
    category_changes = new_names.notna().to_numpy()
    new_ids = _ids_like(new_names.map(name2id).fillna(0).to_numpy(dtype=np.int64), old_ids)

    idx = codes.cat.codes.to_numpy()
    change = manifest["source_id"].isin(flagged_sources).to_numpy() & (idx >= 0)
    change[change] = category_changes[idx[change]]

    relabelled = old_ids.copy()
    if isinstance(relabelled.dtype, pd.CategoricalDtype):
        relabelled = relabelled.astype(relabelled.dtype.categories.dtype)
    relabelled[change] = new_ids[idx[change]]
    out = manifest.copy(deep=False)
    out["coralnet_id"] = relabelled
    out["original_coralnet_id"] = old_ids

    changed = pd.DataFrame(
        {
            "source_id": manifest["source_id"].to_numpy()[change],
            "label": names.to_numpy()[idx[change]],
            "replacement": new_names.to_numpy()[idx[change]],
        }
    )
    return out, changed


def label_changes(changed: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if changed.empty:
        return pd.DataFrame(columns=["label", "replacement", "phylum", "n_points", "n_sources"])
    grouped = (
        changed.groupby(["label", "replacement"])
        .agg(n_points=("source_id", "size"), n_sources=("source_id", "nunique"))
        .reset_index()
    )
    grouped.insert(2, "phylum", grouped["label"].map(table["phylum"]))
    return grouped.sort_values("n_points", ascending=False, ignore_index=True)


def _train_dedup_refs(data_config: Path) -> tuple[str | None, str | None]:
    """``(annotation_overrides, image_blacklist)`` on the CoralNet train split, if set."""
    cfg = yaml.safe_load(Path(data_config).read_text())
    train = ((cfg.get("data") or {}).get("coralnet") or {}).get("train") or {}
    if not isinstance(train, dict):
        return None, None
    return train.get("annotation_overrides"), train.get("image_blacklist")


def apply_train_dedup_filters(
    manifest: pd.DataFrame, data_config: Path, bucket: str
) -> pd.DataFrame:
    """Replace override images' points, then drop blacklisted images.

    Same order as ``CoralNetDataset``. Rows and columns of images that are not
    overridden are left as they are, including ``image_s3_key``.
    """
    overrides_ref, blacklist_ref = _train_dedup_refs(data_config)
    if overrides_ref is None and blacklist_ref is None:
        return manifest

    from mermaidseg.datasets.coralnet.coralnet_dataset import CoralNetDataset

    renamed = "image_s3_key" in manifest.columns and "image_key" not in manifest.columns
    frame = manifest.rename(columns={"image_s3_key": "image_key"}) if renamed else manifest
    dataset = CoralNetDataset.__new__(CoralNetDataset)
    dataset.source_bucket = bucket
    dataset.annotation_overrides = overrides_ref
    dataset.image_blacklist = blacklist_ref
    if overrides_ref is not None:
        frame = dataset._apply_annotation_overrides(frame)
    if blacklist_ref is not None:
        blocked = dataset._load_image_blacklist()
        frame = frame.loc[~frame["image_id"].astype(str).isin(blocked)].reset_index(drop=True)
    if renamed:
        frame = frame.rename(columns={"image_key": "image_s3_key"})
    return frame


def _stats_from_filtered_manifest(
    args: argparse.Namespace, train_ids: list[int], manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source quantiles of ``manifest``, which already has the train dedup filters applied."""
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"filtered manifest is missing {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.parquet"
        manifest.to_parquet(path, index=False)
        stats_args = argparse.Namespace(**vars(args))
        stats_args.manifest = str(path)
        return stats_from_manifest(stats_args, train_ids)


def _configure_cache(args: argparse.Namespace) -> None:
    cache_dir = resolve_cache_dir(args.data_config, args.cache_dir)
    if cache_dir and not Path(cache_dir).is_dir():
        logger.info("cache dir %s does not exist; reading from S3 only", cache_dir)
        cache_dir = None
    configure_cache(cache_dir)


def build(args: argparse.Namespace) -> Path:
    from mermaidseg.datasets.local_cache import LocalS3Cache

    out_dir: Path = args.output_dir
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite to replace it")

    table = replacement_table(load_concept_rows(args.concepts_csv))
    replacements = table["replacement"].dropna().to_dict()
    logger.info("%d CoralNet labels collapse to a coarser label", len(replacements))

    quantiles = dict(zip(METRIC_KEYS, args.quantiles, strict=True))
    train_ids = load_train_sources(args.data_config)
    _configure_cache(args)
    cache = LocalS3Cache.get()
    manifest = apply_train_dedup_filters(
        cache.read_parquet_ref(args.manifest, args.bucket), args.data_config, args.bucket
    )
    if args.stats_csv is not None:
        stats = stats_from_csv(args.stats_csv, train_ids)
    else:
        stats, _ = _stats_from_filtered_manifest(args, train_ids, manifest)
    thresholds = quantile_thresholds(stats, quantiles)
    flags = flag_sources(stats, thresholds)
    summary = summarize(stats, flags, quantiles, thresholds)
    print(format_summary(summary, stats, args.data_config))
    flagged_ids = [int(s) for s in flags.index[flags["flagged"]]]

    id2name = cache.read_json(args.bucket, ID2NAME_KEY)
    collapsed, changed = collapse_manifest(manifest, id2name, replacements, flagged_ids)
    changes = label_changes(changed, table)

    n_flagged_points = int(manifest["source_id"].isin(flagged_ids).sum())
    per_phylum = changes.groupby("phylum")["n_points"].sum().sort_values(ascending=False)
    run_info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "quantiles": quantiles,
        "thresholds": thresholds,
        "summary": json.loads(summary.to_json(orient="records")),
        "flagged_sources": flagged_ids,
        "n_manifest_points": len(manifest),
        "n_flagged_points": n_flagged_points,
        "n_changed_points": len(changed),
        "pct_flagged_points_changed": 100.0 * len(changed) / n_flagged_points
        if n_flagged_points
        else 0.0,
        "n_flagged_points_not_in_csv": points_without_concepts(
            manifest, id2name, table.index, flagged_ids
        ),
        "changed_points_per_phylum": {str(k): int(v) for k, v in per_phylum.items()},
    }
    logger.info(
        "relabelled %d of %d points in %d flagged sources",
        len(changed),
        n_flagged_points,
        len(flagged_ids),
    )

    partial = out_dir.with_name(f".{out_dir.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    collapsed.to_parquet(partial / MANIFEST_FILE, index=False)
    stats.join(flags).to_csv(partial / "source_flags.csv")
    table.to_csv(partial / "label_replacements.csv")
    changes.to_csv(partial / "label_changes.csv", index=False)
    (partial / "run.json").write_text(json.dumps(run_info, indent=2))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    partial.rename(out_dir)
    logger.info("wrote %s", out_dir / MANIFEST_FILE)
    return out_dir


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Read per-source metrics from this CSV instead of the manifest.",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not all(0.0 <= q <= 1.0 for q in args.quantiles):
        parser.error(f"--quantiles must be in [0, 1], got {args.quantiles}")
    if not any(q > 0 for q in args.quantiles):
        parser.error("at least one quantile must be > 0, or no source is flagged")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build(args)


if __name__ == "__main__":
    main()
