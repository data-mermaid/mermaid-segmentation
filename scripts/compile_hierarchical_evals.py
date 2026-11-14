#!/usr/bin/env python3
"""Compile ``evaluate.py --hierarchical`` outputs into one CSV, one row per (run, step, resolution).

Expects ``<root>/<run>/step<N>_res<R>/summary.json`` (as written by
``scripts/run_hier_eval.sh``) and reads only finished cells (``.done``). Columns are the
standard point-eval headline metrics plus the pooled ``hierarchical`` block for CoralNet and
Pacific: consistency, mistake severity, ECE, rare-taxon accuracy, expected tree distance,
partial-label subtree mass and inter-head KL.

Benthos and CoralscapesV2 are added as ``benthos/miou`` and ``coralscapes/miou``. ``--fallback
RUN=DIR`` fills evals missing from a run's cells from ``DIR/step<N>_res<R>/summary.json`` (the
baseline's Benthos / Coralscapes results live in Jonathan's original eval outputs).

Usage:
    python scripts/compile_hierarchical_evals.py --root eval_out/ft50k_hier
    python scripts/compile_hierarchical_evals.py --root eval_out/ft50k_hier --print genus
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
PAIRS = tuple(f"{c}->{f}" for c, f in zip(RANKS, RANKS[1:], strict=False))
EVALS = {"coralnet": "coralnet", "pacific": "pacific"}
CELL = re.compile(r"step(\d+)_res(\d+)$")


def _metric_keys() -> list[str]:
    keys = ["class_accuracy", "class_miou"]
    keys += [f"tax_{r}_acc_{subset}" for r in RANKS for subset in ("all", "living")]
    keys += ["binary_macro_f1", "bleached_f1"]
    keys += [f"consistency/{p}" for p in PAIRS] + ["consistency/overall", "consistency/overall_labeled"]
    keys += [f"severity/{r}/normalized" for r in RANKS] + ["severity/mean_normalized"]
    keys += ["ece/mean"]
    keys += [f"rare/{r}/{s}_macro_acc" for r in ("family", "genus") for s in ("head", "tail")]
    keys += [f"expected_distance/{r}" for r in RANKS]
    keys += [f"partial/{p}/subtree_mass" for p in PAIRS]
    keys += [f"kl/{p}" for p in PAIRS]
    keys += [f"tv/{p}" for p in PAIRS]
    return keys


METRICS = _metric_keys()


def _value(pooled: dict, key: str) -> float:
    if key in ("class_accuracy", "class_miou"):
        v = pooled.get(key)
    elif key.startswith("tax_"):
        _, rank, _, subset = key.split("_")
        v = (pooled.get("taxonomic") or {}).get(rank, {}).get(f"acc_{subset}")
    elif key == "binary_macro_f1":
        v = (pooled.get("binary") or {}).get("macro_f1")
    elif key == "bleached_f1":
        v = (pooled.get("bleached") or {}).get("f1")
    else:
        v = (pooled.get("hierarchical") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")


BASELINE_FALLBACK = (
    "cbm_ft50k_baseline="
    "/data/vision/beery/scratch/sauder/mermaid-segmentation/eval_out/hires/cbm_hires_res256_mb32"
)


def _dense_miou(summary: dict, name: str) -> float:
    section = summary.get(name)
    if not isinstance(section, dict) or "error" in section:
        return float("nan")
    if name == "benthos":
        v = (section.get("benthos") or {}).get("miou")
    else:
        v = (((section.get("feature_kinds") or {}).get("concepts") or {}).get("test") or {}).get("miou")
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")


def build_rows(root: Path, fallbacks: dict[str, Path] | None = None) -> list[dict]:
    fallbacks = fallbacks or {}
    rows = []
    for cell in sorted(root.glob("*/step*_res*")):
        match = CELL.search(cell.name)
        if not match or not (cell / ".done").exists():
            continue
        summary = json.loads((cell / "summary.json").read_text())
        row: dict = {"run": cell.parent.name, "step": int(match[1]), "resolution": int(match[2])}
        fallback_path = fallbacks.get(cell.parent.name, Path("/nonexistent")) / cell.name / "summary.json"
        fallback = json.loads(fallback_path.read_text()) if fallback_path.exists() else {}
        for prefix, name in EVALS.items():
            source = summary if name in summary else fallback
            pooled = (source.get(name) or {}).get("pooled") or {}
            for key in METRICS:
                row[f"{prefix}/{key}"] = _value(pooled, key)
        for name in ("benthos", "coralscapes"):
            value = _dense_miou(summary, name)
            row[f"{name}/miou"] = value if not math.isnan(value) else _dense_miou(fallback, name)
        rows.append(row)
    rows.sort(key=lambda r: (r["run"], r["resolution"], r["step"]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("eval_out/ft50k_hier"))
    p.add_argument("--output", type=Path, default=None, help="CSV path (default: <root>/compiled.csv)")
    p.add_argument("--fallback", action="append", default=[BASELINE_FALLBACK],
                   help="RUN=DIR: read evals missing from RUN's cells from DIR/step<N>_res<R>/summary.json")
    p.add_argument("--print", dest="print_filter", default=None,
                   help="Also print CoralNet columns containing this substring")
    args = p.parse_args()

    fallbacks = {k: Path(v) for k, v in (f.split("=", 1) for f in args.fallback)}
    rows = build_rows(args.root, fallbacks)
    output = args.output or args.root / "compiled.csv"
    if rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} finished cells -> {output}")

    if args.print_filter and rows:
        cols = [c for c in rows[0] if c.startswith("coralnet/") and args.print_filter in c]
        print("run step res " + " ".join(c.removeprefix("coralnet/") for c in cols))
        for r in rows:
            print(r["run"], r["step"], r["resolution"], " ".join(f"{r[c]:.4f}" for c in cols))


if __name__ == "__main__":
    main()
