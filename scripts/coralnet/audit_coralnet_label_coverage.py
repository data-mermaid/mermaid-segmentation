"""Audit how CoralNet annotations map into the training target space.

Mirrors :class:`mermaidseg.dataset_reconciliation.registry.SourceLabelRegistry`'s CoralNet
source->target resolution (numeric ``provider_id`` -> MERMAID benthic-attribute name, optional
roll-up to a parent in the benthic hierarchy, then the ``class_subset`` whitelist) and reports,
per split, how many annotations:

- **unmapped**: the ``coralnet_id`` has no MERMAID mapping (absent, or mapped to ``None``);
- **mapped-out-of-subset**: mapped to a target that (even after roll-up) is not in the class subset;
- **in-subset**: resolves to a trained class.

Both statuses that are NOT ``in-subset`` collapse to background/ignore at training time — so this
quantifies exactly what CoralNet contributes and what is dropped under the current 77-class subset.

Usage::

    AWS_PROFILE=mermaid-core uv run python scripts/coralnet/audit_coralnet_label_coverage.py \
        --manifest s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/20260623_nogit/coralnet_training_manifest_20260623_nogit.parquet

Outputs (default ``reports/``): a per-``coralnet_id`` parquet and a markdown summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from mermaidseg.dataset_reconciliation.concepts import initialize_benthic_hierarchy
from mermaidseg.dataset_reconciliation.label_mapping import fetch_coralnet_to_mermaid
from mermaidseg.dataset_reconciliation.registry import roll_up_label

DEFAULT_MANIFEST = (
    "s3://dev-datamermaid-sm-sources/etl-outputs/coralnet/20260623_nogit/"
    "coralnet_training_manifest_20260623_nogit.parquet"
)


def _resolve_final_target(
    coralnet_id: str,
    mapping: dict[str, str | None],
    hierarchy: dict[str, str] | None,
    subset_lower: set[str],
) -> tuple[str | None, str | None]:
    """Return ``(mapped_target, final_in_subset_target)`` for one coralnet id,
    lowercased.

    Mirrors registry resolution: lookup -> optional roll-up -> subset membership.
    """
    mapped = mapping.get(coralnet_id)
    mapped_lower = mapped.lower() if mapped is not None else None
    if mapped_lower is None:
        return None, None
    if hierarchy is not None:
        try:
            final = roll_up_label(mapped_lower, hierarchy, subset_lower)
        except KeyError:
            final = mapped_lower if mapped_lower in subset_lower else None
    else:
        final = mapped_lower if mapped_lower in subset_lower else None
    return mapped_lower, final


def _status(mapped: str | None, final: str | None) -> str:
    if mapped is None:
        return "unmapped"
    if final is None:
        return "mapped-out-of-subset"
    return "in-subset"


def _pct(n: int, d: int) -> str:
    return f"{(n / d):.1%}" if d else "n/a"


def build_report(
    manifest: str,
    training_config: str,
    data_config: str,
) -> tuple[pd.DataFrame, dict]:
    tc = yaml.safe_load(Path(training_config).read_text())["training"]
    subset_lower = {s.lower() for s in tc["class_subset"]}
    roll_up = bool(tc.get("label_roll_up", False))

    dc = yaml.safe_load(Path(data_config).read_text())["data"]
    val_sources = set(dc["coralnet"]["val"]["whitelist_sources"])

    mapping = fetch_coralnet_to_mermaid()
    hierarchy = initialize_benthic_hierarchy() if roll_up else None

    df = pd.read_parquet(manifest, columns=["source_id", "coralnet_id"])
    df["coralnet_id"] = df["coralnet_id"].astype(str)
    df["split"] = df["source_id"].isin(val_sources).map({True: "val", False: "train"})

    per_id = (
        df.groupby("coralnet_id")
        .agg(
            total=("coralnet_id", "size"),
            val=("split", lambda s: int((s == "val").sum())),
            train=("split", lambda s: int((s == "train").sum())),
        )
        .reset_index()
    )
    resolved = per_id["coralnet_id"].map(
        lambda cid: _resolve_final_target(cid, mapping, hierarchy, subset_lower)
    )
    per_id["mapped_target"] = [r[0] for r in resolved]
    per_id["final_target"] = [r[1] for r in resolved]
    per_id["status"] = [_status(m, f) for m, f in resolved]
    per_id = per_id.sort_values("total", ascending=False).reset_index(drop=True)

    def split_totals(split: str) -> dict:
        col = split
        tot = int(per_id[col].sum())
        by_status = per_id.groupby("status")[col].sum()
        in_sub = int(by_status.get("in-subset", 0))
        mapped = in_sub + int(by_status.get("mapped-out-of-subset", 0))
        return {
            "annotations": tot,
            "mapped": mapped,
            "mapped_pct": _pct(mapped, tot),
            "in_subset": in_sub,
            "in_subset_pct": _pct(in_sub, tot),
        }

    summary = {
        "roll_up": roll_up,
        "subset_size": len(subset_lower),
        "mapping_entries": len(mapping),
        "distinct_coralnet_ids": int(per_id.shape[0]),
        "val": split_totals("val"),
        "train": split_totals("train"),
    }
    return per_id, summary


def write_markdown(per_id: pd.DataFrame, summary: dict, out_md: Path) -> None:
    def row(s: dict) -> str:
        return (
            f"| {s['annotations']:,} | {s['mapped']:,} ({s['mapped_pct']}) "
            f"| {s['in_subset']:,} ({s['in_subset_pct']}) |"
        )

    top_unmapped = per_id[per_id.status == "unmapped"].head(15)
    top_excluded = (
        per_id[per_id.status == "mapped-out-of-subset"]
        .groupby("mapped_target")["total"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    lines = [
        "# CoralNet label-coverage audit",
        "",
        "How CoralNet annotations resolve into the training target space "
        f"(class subset = {summary['subset_size']} classes, "
        f"`label_roll_up={summary['roll_up']}`, {summary['mapping_entries']} API mappings, "
        f"{summary['distinct_coralnet_ids']} distinct coralnet_ids).",
        "",
        "Annotations that are **not** `in-subset` collapse to background/ignore at training time.",
        "",
        "| Split | Annotations | Mapped to a class | In 77-class subset (trained) |",
        "|---|---|---|---|",
        f"| val | {row(summary['val'])[1:]}",
        f"| train | {row(summary['train'])[1:]}",
        "",
        "## Top unmapped coralnet_ids by annotation count",
        "",
        "These `coralnet_id`s have no MERMAID mapping and are candidates for mapping on the "
        "MERMAID side.",
        "",
        "| coralnet_id | annotations | val | train |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.coralnet_id} | {r.total:,} | {r.val:,} | {r.train:,} |"
        for r in top_unmapped.itertuples()
    ]
    lines += [
        "",
        "## Top mapped-but-excluded target classes by annotation count",
        "",
        "These map to a valid MERMAID benthic attribute but fall outside the current 77-class "
        "subset (even after roll-up) — candidates for a subset expansion (needs taxonomy review).",
        "",
        "| target (mapped, lowercased) | annotations |",
        "|---|---|",
    ]
    lines += [f"| {name} | {int(cnt):,} |" for name, cnt in top_excluded.items()]
    lines.append("")
    out_md.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="CoralNet training manifest parquet (s3:// or local)",
    )
    p.add_argument("--training-config", default="configs/training_config_dinov3_linear.yaml")
    p.add_argument("--data-config", default="configs/data_config_coralnet_mermaid.yaml")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args(argv)

    per_id, summary = build_report(args.manifest, args.training_config, args.data_config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "coralnet_label_coverage.parquet"
    md_path = out_dir / "coralnet_label_coverage.md"
    per_id.to_parquet(parquet_path, index=False)
    write_markdown(per_id, summary, md_path)

    print(f"val:   {summary['val']}")
    print(f"train: {summary['train']}")
    print(f"wrote {parquet_path} and {md_path}")


if __name__ == "__main__":
    main()
