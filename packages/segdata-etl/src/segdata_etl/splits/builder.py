"""Split the curated CoralNet dataset into train/val/test by source group.

1. Counts. Per source, images and points per training class. Labels map through
   ``labels.parquet`` ``mermaid_benthic_attribute`` and roll up through the MERMAID benthic
   hierarchy into the ``class_subset`` of ``--training-config``; points outside it are not counted.
2. Groups. Connected components of sources that share a byte-identical image in the raw parent
   version, sources with a near-duplicate pair labelled ``dup`` or ``unsure``, and sources of one
   project (a shared admin, and IDs at most ``max_id_gap`` apart in rank or locations at most
   ``max_km`` apart).
3. Eligibility. The groups of ``--test-sources`` are fixed to test. Any other group may go to
   val/test only if none of its sources shares an image with another source and it has at most
   ``max_group_frac`` of an eval split's target size. Everything else is train.
4. Assignment. Simulated annealing (``anneal``) over the eligible groups, best of ``restarts``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from segdata_etl.splits.anneal import anneal
from segdata_etl.splits.grouping import (
    components,
    load_labels,
    near_dup_edges,
    project_edges,
    shared_image_edges,
)
from segdata_etl.splits.inputs import (
    class_counts,
    file_sha256,
    image_keys,
    images_per_source,
    read_class_subset,
    read_source_list,
    require_crc64nvme,
    sync_tables,
)
from segdata_spec.git import GitState, probe_git
from segdata_spec.manifest import Manifest, Pin
from segdata_spec.read import open_taxonomy
from segdata_spec.splits import read_split_file
from segdata_spec.taxonomy import hierarchy_from_table

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
DATASET = "coralnet"
RAW_DATASET = "coralnet-raw"
TAXONOMY = "mermaid-benthic"
INPUT_FILES = ("near_dup_labels", "metadata", "test_sources", "exclude_sources", "training_config")


@dataclass(frozen=True)
class SplitParams:
    """Tuning of the grouping, eligibility rule and annealing objective."""

    val_frac: float = 0.10
    test_frac: float = 0.10
    max_group_frac: float = 0.25
    max_id_gap: int = 3
    max_km: float = 50.0
    min_class_points: int = 500
    min_eval_points: int = 100
    w_coverage: float = 1.0
    w_size: float = 10.0
    iters: int = 200_000
    restarts: int = 8
    seed: int = 0


@dataclass(frozen=True)
class _Grouping:
    group_of: dict[int, int]
    cross_sources: set[int]
    near_only: set[tuple[int, int]]


def _group_sources(
    sources: list[int],
    keys: pd.DataFrame,
    near_labels: pd.DataFrame,
    metadata: pd.DataFrame,
    params: SplitParams,
) -> _Grouping:
    overlap_edges, cross_sources = shared_image_edges(keys, sources)
    near_only = near_dup_edges(near_labels, sources) - overlap_edges
    overlap_edges |= near_only
    cross_sources |= {s for e in near_only for s in e}
    proj = project_edges(metadata, sources, params.max_id_gap, params.max_km)
    comps = components(sources, overlap_edges | {(a, b) for a, b, _ in proj})
    logger.info(
        "%d project links (%d by id, %d by location), %d overlap links "
        "(%d from near duplicates only) -> %d groups",
        len(proj),
        sum(e[2] == "id" for e in proj),
        sum(e[2] == "location" for e in proj),
        len(overlap_edges),
        len(near_only),
        len(comps),
    )
    return _Grouping(
        group_of={s: i for i, c in enumerate(comps) for s in c},
        cross_sources=cross_sources,
        near_only=near_only,
    )


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x: np.ndarray, y: np.ndarray) -> Any:
        return np.sum(
            np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(y > 0, y, 1)), 0)
        )

    return float((kl(p, m) + kl(q, m)) / 2)


def _best_assignment(
    X: np.ndarray,
    n: np.ndarray,
    grp: pd.DataFrame,
    targets: tuple[float, float],
    fracs: tuple[float, float],
    cls_mask: np.ndarray,
    min_pts: np.ndarray,
    params: SplitParams,
) -> tuple[np.ndarray, float]:
    P = X.sum(0)
    best: tuple[np.ndarray, float] | None = None
    for r in range(params.restarts):
        a, c = anneal(
            X=X,
            n=n,
            P=P,
            eligible=grp["eligible"].to_numpy(),
            targets=targets,
            fracs=fracs,
            w_size=params.w_size,
            cls_mask=cls_mask,
            min_pts=min_pts,
            w_cov=params.w_coverage,
            iters=params.iters,
            seed=params.seed + r,
            fixed=grp["fixed"].to_numpy(),
        )
        logger.info("restart %d: cost %.4f", r, c)
        if best is None or c < best[1]:
            best = (a, c)
    if best is None:
        raise ValueError("restarts must be at least 1")
    return best


def _class_distribution(
    split_counts: pd.DataFrame, P: np.ndarray, cls_mask: np.ndarray, pool: Any, min_pts: Any
) -> pd.DataFrame:
    dist = split_counts.copy()
    dist["total"] = P
    for s in SPLITS:
        dist[f"share_{s}"] = split_counts[s] / P
        dist[f"prop_{s}"] = split_counts[s] / split_counts[s].sum()
    dist["in_objective"] = cls_mask
    dist["eligible_points"] = pool
    dist["min_eval_points"] = min_pts
    return dist.sort_values("total", ascending=False)


def _summary(
    src: pd.DataFrame,
    grp: pd.DataFrame,
    split_counts: pd.DataFrame,
    P: np.ndarray,
    min_pts: Any,
    N: float,
) -> dict[str, Any]:
    return {
        "splits": {
            s: {
                "n_sources": int((src["split"] == s).sum()),
                "n_groups": int((grp["split"] == s).sum()),
                "n_images": int(src.loc[src["split"] == s, "n_images"].sum()),
                "frac_images": float(src.loc[src["split"] == s, "n_images"].sum() / N),
                "n_mapped_points": int(split_counts[s].sum()),
                "n_classes_present": int((split_counts[s] > 0).sum()),
                "js_divergence_vs_all": _js(split_counts[s].to_numpy(), P),
            }
            for s in SPLITS
        },
        "classes_missing": {
            s: split_counts.index[split_counts[s] == 0].tolist() for s in ("val", "test")
        },
        "classes_below_min": {
            s: split_counts.index[split_counts[s] < min_pts].tolist() for s in ("val", "test")
        },
    }


def build_split(
    *,
    sources: list[int],
    n_images: pd.Series,
    counts: pd.DataFrame,
    image_keys: pd.DataFrame,
    near_labels: pd.DataFrame,
    metadata: pd.DataFrame,
    test_sources: set[int],
    subset: set[str],
    params: SplitParams,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Assign every source of ``sources`` to train, val or test.

    Args:
        sources: The sorted universe of source ids.
        n_images: Images per source id.
        counts: Points per source id (rows) and training class (columns).
        image_keys: ``source_id``, ``crc64nvme`` and ``size`` per image of the raw version.
        near_labels: Latest near-duplicate label per image pair (``load_labels``).
        metadata: Per-source ``source_id``, ``admins``, ``latitude`` and ``longitude``.
        test_sources: Source ids whose groups go to test.
        subset: The training classes.
        params: Grouping, eligibility and annealing parameters.

    Returns:
        The split table (``group``, ``source_id``, ``split_group``, ``split``), the summary and
        the per-class distribution over the splits.

    Raises:
        ValueError: A test source is not in ``sources``, or no group is both eligible and free
            to move between splits.
    """
    unknown = sorted(test_sources - set(sources))
    if unknown:
        raise ValueError(f"test sources not in the universe: {unknown}")
    n_img = n_images.reindex(sources, fill_value=0)
    counts = counts.reindex(sources, fill_value=0)
    logger.info(
        "%d sources, %d images, %d of %d training classes present",
        len(sources),
        int(n_img.sum()),
        counts.shape[1],
        len(subset),
    )

    grouping = _group_sources(sources, image_keys, near_labels, metadata, params)
    group_of = grouping.group_of

    src = pd.DataFrame(
        {
            "source_id": sources,
            "group": [group_of[s] for s in sources],
            "n_images": n_img.to_numpy(),
            "shares_images": [s in grouping.cross_sources for s in sources],
        }
    )
    grp = src.groupby("group").agg(
        n_images=("n_images", "sum"),
        shares_images=("shares_images", "any"),
        n_sources=("source_id", "size"),
    )
    X = counts.groupby(src.set_index("source_id")["group"]).sum().reindex(grp.index).to_numpy(float)
    n = grp["n_images"].to_numpy(float)
    N = n.sum()
    fracs = (params.val_frac, params.test_frac)
    targets = (fracs[0] * N, fracs[1] * N)
    cap = params.max_group_frac * min(targets)
    grp["eligible"] = ~grp["shares_images"] & (grp["n_images"] <= cap) & (grp["n_images"] > 0)
    test_groups = {group_of[s] for s in test_sources}
    grp["fixed"] = np.where(grp.index.isin(test_groups), SPLITS.index("test"), -1)
    logger.info(
        "eligible groups: %d with %d images (cap %.0f images per group); targets %.0f / %.0f",
        int(grp["eligible"].sum()),
        int(grp.loc[grp["eligible"], "n_images"].sum()),
        cap,
        targets[0],
        targets[1],
    )

    if not (grp["eligible"] & (grp["fixed"] < 0)).any():
        raise ValueError(
            "no group is both eligible and free to move: every eligible group holds a "
            "--test-sources source, none passes --max-group-frac, or every remaining group "
            "shares an image with another source"
        )
    P = X.sum(0)
    cls_mask = params.min_class_points <= P
    pool = X[(grp["eligible"] | (grp["fixed"] >= 0)).to_numpy()].sum(0)
    min_pts = np.minimum(params.min_eval_points, pool / 4)
    assignment, cost = _best_assignment(X, n, grp, targets, fracs, cls_mask, min_pts, params)
    grp["split"] = np.array(SPLITS)[assignment]
    src["split"] = src["group"].map(grp["split"])

    split_counts = pd.DataFrame(
        {s: X[grp["split"].to_numpy() == s].sum(0) for s in SPLITS}, index=counts.columns
    )
    summary = {
        "n_groups": len(grp),
        "n_eligible_groups": int(grp["eligible"].sum()),
        "test_sources": sorted(test_sources),
        "fixed_to_test": sorted(
            int(s) for s in src.loc[src["group"].isin(test_groups), "source_id"]
        ),
        "n_near_dup_links": len(grouping.near_only),
        "cost": float(cost),
        **_summary(src, grp, split_counts, P, min_pts, N),
        "classes_absent_from_coralnet": sorted(subset - set(counts.columns)),
    }
    split_table = pd.DataFrame(
        {
            "group": "s" + src["source_id"].astype(str),
            "source_id": src["source_id"],
            "split_group": src["group"],
            "split": src["split"],
        }
    ).sort_values(["split", "split_group", "source_id"], ignore_index=True)
    return split_table, summary, _class_distribution(split_counts, P, cls_mask, pool, min_pts)


def write_outputs(
    out_dir: Path,
    split_table: pd.DataFrame,
    summary: dict[str, Any],
    class_distribution: pd.DataFrame,
    meta: dict[str, Any],
) -> Path:
    """Write the split file, its meta, the class distribution and the summary; return the split file.

    The written split file is read back through ``read_split_file``, which raises ``ValueError``
    when it is not a valid split file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / f"{DATASET}.csv"
    split_table.to_csv(split_file, index=False)
    (out_dir / f"{DATASET}.meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    class_distribution.to_csv(out_dir / "class_distribution.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    read_split_file(split_file)
    return split_file


def _pins(args: argparse.Namespace) -> tuple[dict[str, Pin], Pin]:
    datasets = {
        DATASET: Pin(name=DATASET, version=args.version, manifest_sha256=args.manifest_sha256),
        RAW_DATASET: Pin(
            name=RAW_DATASET, version=args.raw_version, manifest_sha256=args.raw_manifest_sha256
        ),
    }
    taxonomy = Pin(
        name=TAXONOMY, version=args.taxonomy_version, manifest_sha256=args.taxonomy_manifest_sha256
    )
    return datasets, taxonomy


def _meta(
    args: argparse.Namespace,
    pins: dict[str, Pin],
    excluded: set[int],
    params: SplitParams,
    summary: dict[str, Any],
    git: GitState,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "file": f"{DATASET}.csv",
        "bucket": args.bucket,
        "pins": {
            name: {"version": pin.version, "manifest_sha256": pin.manifest_sha256}
            for name, pin in pins.items()
        },
        "inputs": {
            name: {"path": str(getattr(args, name)), "sha256": file_sha256(getattr(args, name))}
            for name in INPUT_FILES
        },
        "excluded_sources": sorted(excluded),
        "parameters": asdict(params),
        "git": {"sha": git.sha, "dirty": git.dirty},
        "summary": summary,
    }


def params_from_args(args: argparse.Namespace) -> SplitParams:
    """Collect the ``SplitParams`` fields from parsed arguments."""
    return SplitParams(**{name: getattr(args, name) for name in SplitParams.__dataclass_fields__})


def require_raw_parent(manifest: Manifest, raw: Pin) -> None:
    """Refuse a raw pin that is not the version the curated manifest declares as its parent."""
    declared = [p for p in manifest.parents if p.dataset == RAW_DATASET]
    if not any(
        p.version == raw.version and p.manifest_sha256 == raw.manifest_sha256 for p in declared
    ):
        found = [(p.version, p.manifest_sha256) for p in declared]
        raise ValueError(
            f"the {RAW_DATASET} pin ({raw.version}, {raw.manifest_sha256}) is not a "
            f"{RAW_DATASET} parent of {manifest.dataset} {manifest.version}; declared: {found}"
        )


def _with_excluded(split_table: pd.DataFrame, excluded: set[int]) -> pd.DataFrame:
    """Append one ``excluded`` row, with an empty ``split_group``, per source id of ``excluded``."""
    ids = sorted(excluded)
    rows = pd.DataFrame(
        {
            "group": [f"s{s}" for s in ids],
            "source_id": pd.Series(ids, dtype=split_table["source_id"].dtype),
            "split_group": pd.Series([pd.NA] * len(ids), dtype="Int64"),
            "split": "excluded",
        }
    )
    return pd.concat([split_table.astype({"split_group": "Int64"}), rows], ignore_index=True)


def run(args: argparse.Namespace, client: Any) -> int:
    """Sync the pinned tables, build the split and write it to ``args.out_dir``.

    Raises:
        ValueError: The raw pin is not a parent of the curated version, or a raw image of a
            universe source has no CRC64NVME.
    """
    git = probe_git(Path(__file__).resolve().parent)
    datasets, taxonomy = _pins(args)
    versions = sync_tables(client, datasets, args.cache_dir, args.bucket, taxonomy)
    curated = versions[DATASET]
    require_raw_parent(curated.manifest, datasets[RAW_DATASET])
    tax = open_taxonomy(args.cache_dir, args.bucket, taxonomy)
    excluded = read_source_list(args.exclude_sources)
    n_images = images_per_source(curated.images)
    sources = sorted(set(n_images.index) - excluded)
    raw_keys = image_keys(versions[RAW_DATASET].images)
    universe_keys = raw_keys[raw_keys["source_id"].isin(sources)]
    require_crc64nvme(universe_keys["source_id"], universe_keys["crc64nvme"])
    subset = read_class_subset(args.training_config)
    params = params_from_args(args)
    split_table, summary, dist = build_split(
        sources=sources,
        n_images=n_images,
        counts=class_counts(
            curated.label_counts,
            curated.labels,
            curated.images,
            hierarchy_from_table(tax.benthic_attributes),
            subset,
        ),
        image_keys=raw_keys,
        near_labels=load_labels(args.near_dup_labels),
        metadata=pd.read_csv(args.metadata),
        test_sources=read_source_list(args.test_sources),
        subset=subset,
        params=params,
    )
    split_table = _with_excluded(split_table, excluded & set(n_images.index))
    meta = _meta(args, {**datasets, TAXONOMY: taxonomy}, excluded, params, summary, git)
    split_file = write_outputs(args.out_dir, split_table, summary, dist, meta)
    logger.info("wrote %s", split_file)
    return 0
