"""Check a split file against a dataset version for images shared across splits.

Two images are the same when their ``(crc64nvme, size)`` match; ``run`` refuses images without a
CRC64NVME in listed sources; ``check_leakage`` on its own skips them. Groups the split file does not
list are ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from segdata_etl.splits.inputs import require_crc64nvme, sync_tables
from segdata_spec.manifest import Pin
from segdata_spec.splits import read_split_file

DATASET = "coralnet"
IMAGE_KEY = ["crc64nvme", "size"]
# (eval split, other split): each eval split is compared with every split ahead of it.
EVAL_PAIRS = (("val", "train"), ("test", "train"), ("test", "val"))


@dataclass(frozen=True)
class LeakageReport:
    """Per-group duplicate counts, cross-split group pairs and split_groups spanning splits."""

    per_source: pd.DataFrame
    source_pairs: pd.DataFrame
    spanning_split_groups: pd.DataFrame

    @property
    def has_leakage(self) -> bool:
        """True when any eval image, group pair or split_group crosses splits."""
        return bool(
            self.per_source["n_dup"].sum() > 0
            or len(self.source_pairs)
            or len(self.spanning_split_groups)
        )


def _keyed_images(split_table: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    split_of = split_table.set_index("group")["split"]
    keyed = images[["group", *IMAGE_KEY]].copy()
    keyed["split"] = keyed["group"].map(split_of)
    return keyed[keyed["split"].notna()]


def _per_source(keyed: pd.DataFrame) -> pd.DataFrame:
    comparable = keyed[keyed["crc64nvme"].notna()]
    tables = []
    for eval_split, other_split in EVAL_PAIRS:
        other = pd.MultiIndex.from_frame(
            comparable.loc[comparable["split"] == other_split, IMAGE_KEY]
        )
        x = keyed[keyed["split"] == eval_split]
        dup = pd.MultiIndex.from_frame(x[IMAGE_KEY]).isin(other) & x["crc64nvme"].notna().to_numpy()
        t = (
            x.assign(dup=dup)
            .groupby("group")["dup"]
            .agg(n_images="size", n_dup="sum")
            .reset_index()
        )
        t["pct_dup"] = 100 * t["n_dup"] / t["n_images"]
        t.insert(0, "eval_split", eval_split)
        t.insert(1, "other_split", other_split)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def _source_pairs(keyed: pd.DataFrame) -> pd.DataFrame:
    rows = keyed[keyed["crc64nvme"].notna()].drop_duplicates(["group", *IMAGE_KEY])
    pairs = rows.merge(rows, on=IMAGE_KEY, suffixes=("_a", "_b"))
    pairs = pairs[(pairs["group_a"] < pairs["group_b"]) & (pairs["split_a"] != pairs["split_b"])]
    return (
        pairs.groupby(["group_a", "group_b", "split_a", "split_b"])
        .size()
        .rename("n_shared")
        .reset_index()
    )


def _spanning_split_groups(split_table: pd.DataFrame) -> pd.DataFrame:
    if "split_group" not in split_table.columns:
        return pd.DataFrame({"split_group": [], "splits": []})
    splits = split_table.groupby("split_group")["split"].agg(lambda s: ";".join(sorted(set(s))))
    return splits[splits.str.contains(";")].rename("splits").reset_index()


def check_leakage(split_table: pd.DataFrame, images: pd.DataFrame) -> LeakageReport:
    """Find images and split_groups that cross splits.

    Args:
        split_table: ``group`` and ``split``, with an optional ``split_group``.
        images: ``group``, ``crc64nvme`` and ``size`` per image.
    """
    keyed = _keyed_images(split_table, images)
    return LeakageReport(
        per_source=_per_source(keyed),
        source_pairs=_source_pairs(keyed),
        spanning_split_groups=_spanning_split_groups(split_table),
    )


def format_report(report: LeakageReport) -> str:
    """Render the report as plain text."""
    lines = []
    for (eval_split, other_split), t in report.per_source.groupby(
        ["eval_split", "other_split"], sort=False
    ):
        n_dup, n_images = int(t["n_dup"].sum()), int(t["n_images"].sum())
        lines.append(f"{eval_split} images also in {other_split}: {n_dup:,}/{n_images:,}")
        leaking = t[t["n_dup"] > 0].drop(columns=["eval_split", "other_split"])
        if len(leaking):
            lines.append(leaking.round(1).to_string(index=False))
    lines.append(f"cross-split group pairs: {len(report.source_pairs)}")
    if len(report.source_pairs):
        lines.append(report.source_pairs.to_string(index=False))
    lines.append(f"split_groups spanning splits: {len(report.spanning_split_groups)}")
    if len(report.spanning_split_groups):
        lines.append(report.spanning_split_groups.to_string(index=False))
    return "\n".join(lines)


def write_report(report: LeakageReport, out_dir: Path) -> None:
    """Write the report's three tables as CSVs in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report.per_source.to_csv(out_dir / "leakage_per_source.csv", index=False)
    report.source_pairs.to_csv(out_dir / "leakage_pairs.csv", index=False)
    report.spanning_split_groups.to_csv(out_dir / "leakage_split_groups.csv", index=False)


def run(args: argparse.Namespace, client: Any) -> int:
    """Check ``args.split_file`` against the pinned version; return 1 on leakage, else 0.

    Raises:
        ValueError: A listed group holds an image without a CRC64NVME.
    """
    split_table = read_split_file(args.split_file).to_pandas()
    pin = Pin(name=DATASET, version=args.version, manifest_sha256=args.manifest_sha256)
    version = sync_tables(client, {DATASET: pin}, args.cache_dir, args.bucket)[DATASET]
    images = version.images.to_pandas()
    listed = images[images["group"].isin(split_table["group"])]
    require_crc64nvme(listed["group"], listed["crc64nvme"])
    report = check_leakage(split_table, images)
    print(format_report(report))
    if args.out_dir is not None:
        write_report(report, args.out_dir)
    return 1 if report.has_leakage else 0
