#!/usr/bin/env python3
"""Compile the higher-resolution fine-tune's per-eval outputs into a single CSV.

Meant to run on the cluster checkout (``$SCRATCH/mermaid-segmentation``), where
the eval jobs write their results. Each eval lands in

    eval_out/hires/<run>/step<N>_res<R>/

and is only considered finished once ``<out>/.done`` exists (see
slurm/hires_eval.sbatch). ``model_step{N}`` is the checkpoint after N optimizer
updates of the fine-tune. The grid is train resolutions {256, 384, 512, 768}
x steps 10000..50000 x prediction resolutions {256, 384, 512, 768, 1024}.

``total_steps`` is 700000 (the cooled rank-16 source) plus the fine-tune step.
Headline metrics match ``scripts/compile_sweep_loss_weights.py`` (including the
merged ``extended/`` evals) and are filled only when ``.done`` exists and
``summary.json`` parses. Every other cell stays in the CSV with metric columns
as NaN.

Usage:
    python scripts/compile_hires_evals.py
    python scripts/compile_hires_evals.py \\
        --root eval_out/hires \\
        --output eval_out/hires/compiled.csv
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

# Must match slurm/hires_train.sbatch and configs/training_config_cbm_hires.yaml.
# (train resolution, microbatch, grad accum). Effective batch is 32.
TRAIN_GRID: list[tuple[int, int, int]] = [
    (256, 32, 1),
    (384, 16, 2),
    (512, 8, 4),
    (768, 4, 8),
]
TRAIN_RESOLUTIONS: list[int] = [res for res, _, _ in TRAIN_GRID]
MICROBATCHES: list[int] = [mb for _, mb, _ in TRAIN_GRID]
ACCUM_STEPS: list[int] = [accum for _, _, accum in TRAIN_GRID]
SOURCE_STEPS = 700_000
TOTAL_STEPS = 50_000
CHECKPOINT_EVERY = 10_000
FINETUNE_STEPS: list[int] = list(range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY))
PRED_RESOLUTIONS: list[int] = [256, 384, 512, 768, 1024]
RUN_PREFIX = "cbm_hires"

ID_COLUMNS = [
    "run",
    "train_resolution",
    "microbatch",
    "grad_accum_steps",
    "finetune_steps",
    "total_steps",
    "pred_resolution",
    *STATUS_COLUMNS,
]
ALL_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

_RUN_RE = re.compile(rf"^{RUN_PREFIX}_res(?P<res>\d+)_mb(?P<mb>\d+)$")
_DIR_RE = re.compile(r"^step(?P<step>\d+)_res(?P<res>\d+)$")


def run_name(resolution: int, microbatch: int) -> str:
    return f"{RUN_PREFIX}_res{resolution}_mb{microbatch}"


def parse_run(run: str) -> tuple[int, int] | None:
    match = _RUN_RE.match(run)
    if match is None:
        return None
    return int(match.group("res")), int(match.group("mb"))


def grad_accum_steps(resolution: int, microbatch: int) -> int | None:
    for res, mb, accum in TRAIN_GRID:
        if res == resolution and mb == microbatch:
            return accum
    return None


def _row(
    run: str,
    train_resolution: int,
    microbatch: int,
    finetune_steps: int,
    pred_resolution: int,
    out_dir: Path,
) -> dict:
    accum = grad_accum_steps(train_resolution, microbatch)
    row: dict[str, object] = {
        "run": run,
        "train_resolution": train_resolution,
        "microbatch": microbatch,
        "grad_accum_steps": "" if accum is None else accum,
        "finetune_steps": finetune_steps,
        "total_steps": SOURCE_STEPS + finetune_steps,
        "pred_resolution": pred_resolution,
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
        if parse_run(run_dir.name) is None:
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

    for resolution, microbatch, _accum in TRAIN_GRID:
        run = run_name(resolution, microbatch)
        for steps in FINETUNE_STEPS:
            for pred in PRED_RESOLUTIONS:
                expected.add((run, steps, pred))
                rows.append(
                    _row(
                        run,
                        resolution,
                        microbatch,
                        steps,
                        pred,
                        root / run / f"step{steps}_res{pred}",
                    )
                )

    for run, steps, pred in _discover_extra_cells(root, expected):
        parsed = parse_run(run)
        if parsed is None:
            continue
        resolution, microbatch = parsed
        rows.append(
            _row(run, resolution, microbatch, steps, pred, root / run / f"step{steps}_res{pred}")
        )

    rows.sort(
        key=lambda r: (
            int(r["train_resolution"]),
            int(r["microbatch"]),
            int(r["finetune_steps"]),
            int(r["pred_resolution"]),
        )
    )
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
        default=_REPO_ROOT / "eval_out" / "hires",
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
