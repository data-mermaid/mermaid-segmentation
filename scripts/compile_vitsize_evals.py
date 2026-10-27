#!/usr/bin/env python3
"""Compile the ViT-size sweep's per-eval outputs into a single CSV.

Meant to run on the cluster checkout (``$SCRATCH/mermaid-segmentation``), where
the eval jobs write their results. Each eval lands in

    eval_out/vitsize/<run>/step<N>_res<R>/

and is only considered finished once ``<out>/.done`` exists (see
slurm/vitsize_eval.sbatch). ``model_step{N}`` is the checkpoint after N
optimizer updates. The grid is ViT-S/B/L/H+ x steps 25000..150000 x {256, 512}.

Headline metrics match ``scripts/compile_sweep_loss_weights.py`` (including the
merged ``extended/`` evals) and are filled only when ``.done`` exists and
``summary.json`` parses. Every other cell stays in the CSV with metric columns
as NaN.

Usage:
    python scripts/compile_vitsize_evals.py
    python scripts/compile_vitsize_evals.py \\
        --root eval_out/vitsize \\
        --output eval_out/vitsize/compiled.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from compile_sweep_loss_weights import (  # noqa: E402
    METRIC_COLUMNS,
    STATUS_COLUMNS,
    _format_value,
    load_cell,
)

# Must match slurm/vitsize_train.sbatch SIZES and configs/training_config_cbm_vitsize.yaml.
VIT_SIZES: list[str] = ["vits", "vitb", "vitl", "vithplus"]
TOTAL_STEPS = 150_000
CHECKPOINT_EVERY = 25_000
STEPS: list[int] = list(range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY))
RESOLUTIONS: list[int] = [256, 512]
RUN_PREFIX = "cbm_vitsize"

ID_COLUMNS = ["run", "vit", "steps", "resolution", *STATUS_COLUMNS]
ALL_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

_DIR_RE = re.compile(r"^step(?P<step>\d+)_res(?P<res>\d+)$")


def run_name(vit: str) -> str:
    return f"{RUN_PREFIX}_{vit}"


def vit_from_run(run: str) -> str | None:
    prefix = f"{RUN_PREFIX}_"
    if not run.startswith(prefix) or len(run) == len(prefix):
        return None
    return run[len(prefix) :]


def _row(run: str, vit: str, steps: int, resolution: int, out_dir: Path) -> dict:
    row: dict[str, object] = {
        "run": run,
        "vit": vit,
        "steps": steps,
        "resolution": resolution,
    }
    row.update(load_cell(out_dir))
    return row


def _discover_extra_cells(
    root: Path, expected: set[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """On-disk ``step*_res*`` dirs for this sweep that are outside the expected grid."""
    extras: list[tuple[str, int, int]] = []
    if not root.is_dir():
        return extras
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if vit_from_run(run_dir.name) is None:
            continue
        for cell_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            match = _DIR_RE.match(cell_dir.name)
            if not match:
                continue
            key = (run_dir.name, int(match.group("step")), int(match.group("res")))
            if key not in expected:
                extras.append(key)
    return extras


def build_rows(root: Path) -> list[dict]:
    """One row per expected grid cell, plus any extra on-disk cells for this sweep."""
    rows: list[dict] = []
    expected: set[tuple[str, int, int]] = set()

    for vit in VIT_SIZES:
        run = run_name(vit)
        for steps in STEPS:
            for res in RESOLUTIONS:
                expected.add((run, steps, res))
                rows.append(_row(run, vit, steps, res, root / run / f"step{steps}_res{res}"))

    for run, steps, res in _discover_extra_cells(root, expected):
        vit = vit_from_run(run) or ""
        rows.append(_row(run, vit, steps, res, root / run / f"step{steps}_res{res}"))

    rows.sort(key=lambda r: (str(r["vit"]), int(r["steps"]), int(r["resolution"])))
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLUMNS)
        for row in rows:
            writer.writerow([_format_value(row.get(c)) for c in ALL_COLUMNS])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "vitsize",
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
    output = args.output or (args.root / "compiled.csv")
    rows = build_rows(args.root)
    write_csv(rows, output)
    n_done = sum(1 for row in rows if row["done"])
    n_extended = sum(1 for row in rows if row["done_extended"])
    print(
        f"Wrote {len(rows)} rows to {output} "
        f"({n_done} done, {len(rows) - n_done} missing/incomplete -> NaN; "
        f"{n_extended} extended done)."
    )


if __name__ == "__main__":
    main()
