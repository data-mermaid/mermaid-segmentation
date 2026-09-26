#!/usr/bin/env python3
"""Compile the loss-weight sweep's per-eval outputs into a single CSV.

Meant to run on the cluster checkout (``$SCRATCH/mermaid-segmentation``), where
the eval jobs actually write their results. Each eval lands in

    eval_out/sweep_loss_weights/<run>/epoch<N>_res<R>/

and is only considered finished once ``<out>/.done`` exists (the launcher and
eval sbatch treat that marker, not the incrementally-written ``summary.json``,
as the "finished" signal -- see slurm/sweep_loss_weights_eval.sbatch).

The output is one row per (run, epoch, resolution) grid cell. Headline metrics
are filled only when ``.done`` exists and ``summary.json`` parses; every other
case (not started, still running, or ``.done`` with a missing/unreadable
summary) leaves the metric columns as NaN. An individual eval recorded as
``{"error": ...}`` inside an otherwise finished summary leaves just that eval's
columns as NaN.

Usage:
    python scripts/compile_sweep_loss_weights.py
    python scripts/compile_sweep_loss_weights.py \
        --root eval_out/sweep_loss_weights \
        --output eval_out/sweep_loss_weights/compiled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Expected sweep grid (must match slurm/sweep_loss_weights_train.sbatch and
# slurm/sweep_loss_weights_launch_evals.sh).
# ---------------------------------------------------------------------------
# (per_pixel_loss_weight, per_image_loss_weight) pairs, indexed by array task.
LOSS_WEIGHT_PAIRS: list[tuple[str, str]] = [
    ("1.0", "0.0"),
    ("0.9", "0.1"),
    ("0.75", "0.25"),
    ("0.5", "0.5"),
    ("0.25", "0.75"),
    ("0.1", "0.9"),
    ("0.0", "1.0"),
]
# epochs: 10 in configs/training_config_cbm_sweep.yaml -> checkpoints epoch0..9.
EPOCHS: list[int] = list(range(10))
RESOLUTIONS: list[int] = [256, 512]

# Taxonomic ranks, in TAXONOMIC_CONCEPTS order
# (mermaidseg/dataset_reconciliation/concepts.py).
TAX_RANKS: list[str] = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Benthos spec class names (configs/eval/benthos_zero_shot.yaml), used as the
# keys of the pooled "benthos" per_class table.
BENTHOS_CLASSES: list[str] = [
    "Hard Coral",
    "Soft Coral",
    "Sand",
    "Rock",
    "Sponge",
    "Other",
    "Algae",
    "Calcareous Algae",
]

# Coralscapes probe feature kinds (scripts/evaluate.py default --probe-features).
CORALSCAPES_FEATURE_KINDS: list[str] = ["concepts"]

NAN = float("nan")

_DIR_RE = re.compile(r"^epoch(?P<epoch>\d+)_res(?P<res>\d+)$")


def _slug(name: str) -> str:
    """Column-friendly form of a class name: 'Calcareous Algae' -> 'calcareous_algae'."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _run_name(pix: str, img: str) -> str:
    return f"cbm_lossw_pix{pix}_img{img}"


# ---------------------------------------------------------------------------
# Column schema. Fixed regardless of what has completed so the header is stable.
# ---------------------------------------------------------------------------
ID_COLUMNS = ["run", "pix_weight", "img_weight", "epoch", "resolution", "done"]


def _point_eval_columns(prefix: str) -> list[str]:
    """Columns for a CoralNet/Pacific-style point eval, from its `pooled` block."""
    cols = [f"{prefix}_class_accuracy", f"{prefix}_class_miou"]
    for rank in TAX_RANKS:
        cols.append(f"{prefix}_tax_{rank}_acc_all")
        cols.append(f"{prefix}_tax_{rank}_acc_living")
    cols.append(f"{prefix}_binary_macro_accuracy")
    cols.append(f"{prefix}_binary_macro_f1")
    return cols


def _benthos_columns() -> list[str]:
    cols = ["benthos_accuracy", "benthos_miou"]
    for name in BENTHOS_CLASSES:
        cols.append(f"benthos_iou_{_slug(name)}")
    return cols


def _coralscapes_columns() -> list[str]:
    cols: list[str] = []
    for kind in CORALSCAPES_FEATURE_KINDS:
        cols.append(f"coralscapes_{kind}_test_accuracy")
        cols.append(f"coralscapes_{kind}_test_miou")
        cols.append(f"coralscapes_{kind}_probe_train_accuracy")
    return cols


METRIC_COLUMNS = (
    _point_eval_columns("coralnet")
    + _point_eval_columns("pacific")
    + _benthos_columns()
    + _coralscapes_columns()
)
ALL_COLUMNS = ID_COLUMNS + METRIC_COLUMNS


# ---------------------------------------------------------------------------
# Metric extraction. Every getter returns NaN for anything missing / malformed.
# ---------------------------------------------------------------------------
def _num(value: object) -> float:
    """Coerce a JSON value to float, mapping None/non-numeric to NaN."""
    if isinstance(value, bool):
        return NAN
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else NAN
    return NAN


def _as_dict(value: object) -> dict:
    """Return `value` if it's an error-free dict, else an empty dict."""
    if isinstance(value, dict) and "error" not in value:
        return value
    return {}


def _extract_point_eval(prefix: str, section: object) -> dict[str, float]:
    cols = _point_eval_columns(prefix)
    out = {c: NAN for c in cols}
    pooled = _as_dict(_as_dict(section).get("pooled"))
    if not pooled:
        return out

    out[f"{prefix}_class_accuracy"] = _num(pooled.get("class_accuracy"))
    out[f"{prefix}_class_miou"] = _num(pooled.get("class_miou"))

    taxonomic = pooled.get("taxonomic")
    if isinstance(taxonomic, dict):
        for rank in TAX_RANKS:
            rd = taxonomic.get(rank)
            if isinstance(rd, dict):
                out[f"{prefix}_tax_{rank}_acc_all"] = _num(rd.get("acc_all"))
                out[f"{prefix}_tax_{rank}_acc_living"] = _num(rd.get("acc_living"))

    binary = pooled.get("binary")
    if isinstance(binary, dict):
        out[f"{prefix}_binary_macro_accuracy"] = _num(binary.get("macro_accuracy"))
        out[f"{prefix}_binary_macro_f1"] = _num(binary.get("macro_f1"))

    return out


def _extract_benthos(section: object) -> dict[str, float]:
    cols = _benthos_columns()
    out = {c: NAN for c in cols}
    pooled = _as_dict(_as_dict(section).get("benthos"))
    if not pooled:
        return out

    out["benthos_accuracy"] = _num(pooled.get("accuracy"))
    out["benthos_miou"] = _num(pooled.get("miou"))

    per_class = pooled.get("per_class")
    if isinstance(per_class, dict):
        for name in BENTHOS_CLASSES:
            cd = per_class.get(name)
            if isinstance(cd, dict):
                out[f"benthos_iou_{_slug(name)}"] = _num(cd.get("iou"))
    return out


def _extract_coralscapes(section: object) -> dict[str, float]:
    cols = _coralscapes_columns()
    out = {c: NAN for c in cols}
    feature_kinds = _as_dict(section).get("feature_kinds")
    if not isinstance(feature_kinds, dict):
        return out

    for kind in CORALSCAPES_FEATURE_KINDS:
        kd = feature_kinds.get(kind)
        if not isinstance(kd, dict):
            continue
        test = kd.get("test")
        if isinstance(test, dict):
            out[f"coralscapes_{kind}_test_accuracy"] = _num(test.get("accuracy"))
            out[f"coralscapes_{kind}_test_miou"] = _num(test.get("miou"))
        probe = kd.get("probe")
        if isinstance(probe, dict):
            out[f"coralscapes_{kind}_probe_train_accuracy"] = _num(probe.get("train_accuracy"))
    return out


def _empty_metrics() -> dict[str, float]:
    return {c: NAN for c in METRIC_COLUMNS}


def _metrics_from_summary(summary: dict) -> dict[str, float]:
    metrics = _empty_metrics()
    metrics.update(_extract_point_eval("coralnet", summary.get("coralnet")))
    metrics.update(_extract_point_eval("pacific", summary.get("pacific")))
    metrics.update(_extract_benthos(summary.get("benthos")))
    metrics.update(_extract_coralscapes(summary.get("coralscapes")))
    return metrics


def _read_row(run: str, pix: str, img: str, epoch: int, res: int, out_dir: Path) -> dict:
    """Build one CSV row. Metrics stay NaN unless `.done` + a parseable summary."""
    row: dict[str, object] = {
        "run": run,
        "pix_weight": pix,
        "img_weight": img,
        "epoch": epoch,
        "resolution": res,
        "done": False,
    }
    row.update(_empty_metrics())

    if not (out_dir / ".done").exists():
        return row
    row["done"] = True

    summary_path = out_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return row
    if not isinstance(summary, dict):
        return row

    row.update(_metrics_from_summary(summary))
    return row


def _discover_extra_cells(root: Path, expected: set[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Find on-disk `epoch*_res*` dirs outside the expected grid (e.g. extra epochs)."""
    extras: list[tuple[str, int, int]] = []
    if not root.is_dir():
        return extras
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        run = run_dir.name
        for cell_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            m = _DIR_RE.match(cell_dir.name)
            if not m:
                continue
            key = (run, int(m.group("epoch")), int(m.group("res")))
            if key not in expected:
                extras.append(key)
    return extras


def _pix_img_for_run(run: str) -> tuple[str, str]:
    """Recover (pix, img) weight strings from a run directory name."""
    m = re.match(r"^cbm_lossw_pix(?P<pix>.+?)_img(?P<img>.+)$", run)
    if m:
        return m.group("pix"), m.group("img")
    return "", ""


def build_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    expected: set[tuple[str, int, int]] = set()

    for pix, img in LOSS_WEIGHT_PAIRS:
        run = _run_name(pix, img)
        run_dir = root / run
        for epoch in EPOCHS:
            for res in RESOLUTIONS:
                expected.add((run, epoch, res))
                out_dir = run_dir / f"epoch{epoch}_res{res}"
                rows.append(_read_row(run, pix, img, epoch, res, out_dir))

    for run, epoch, res in _discover_extra_cells(root, expected):
        pix, img = _pix_img_for_run(run)
        out_dir = root / run / f"epoch{epoch}_res{res}"
        rows.append(_read_row(run, pix, img, epoch, res, out_dir))

    rows.sort(key=lambda r: (str(r["run"]), int(r["epoch"]), int(r["resolution"])))
    return rows


def _format_value(value: object) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLUMNS)
        for row in rows:
            writer.writerow([_format_value(row.get(c, NAN)) for c in ALL_COLUMNS])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "sweep_loss_weights",
        help="Directory holding the per-run eval outputs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: <root>/compiled.csv).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root
    output = args.output or (root / "compiled.csv")

    rows = build_rows(root)
    write_csv(rows, output)

    n_done = sum(1 for r in rows if r["done"])
    n_total = len(rows)
    print(
        f"Wrote {n_total} rows to {output} "
        f"({n_done} done, {n_total - n_done} missing/incomplete -> NaN)."
    )


if __name__ == "__main__":
    main()
