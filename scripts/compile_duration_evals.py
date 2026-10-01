#!/usr/bin/env python3
"""Compile duration-ablation evals (peak LR and cooldown) into one CSV.

Meant to run on the cluster checkout (``$SCRATCH/mermaid-segmentation``), where
the eval jobs write their results. The two launchers land outputs in

    eval_out/duration_peak/cbm_duration_pix0.5_img0.5/epoch<N>_res<R>/
    eval_out/duration_cooldown/cbm_duration_pix0.5_img0.5_cd<S>/epoch0_res<R>/

``model_epoch{N}`` of the constant-LR run is the checkpoint after
``(N+1)*25000`` steps. A cooldown row with ``constant_lr_steps == S`` is the
eval of that same checkpoint after the extra 25k linear decay to LR 0, so
``total_steps`` is ``S + 25000`` on cooled rows and ``S`` on peak rows.

An eval counts as finished only when ``<out>/.done`` exists (see
slurm/sweep_loss_weights_eval.sbatch). Headline metrics match
``scripts/compile_sweep_loss_weights.py`` and are filled only when ``.done``
exists and ``summary.json`` parses. Every other cell stays in the CSV with
metric columns as NaN, so the grid is stable while jobs are still running.

Usage:
    python scripts/compile_duration_evals.py
    python scripts/compile_duration_evals.py \\
        --peak-root eval_out/duration_peak \\
        --cooldown-root eval_out/duration_cooldown \\
        --output eval_out/duration_compiled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from compile_sweep_loss_weights import (  # noqa: E402
    METRIC_COLUMNS,
    _empty_metrics,
    _format_value,
    _metrics_from_summary,
)

# Must match configs/training_config_cbm_duration.yaml and the cooldown launcher.
MAIN_RUN = "cbm_duration_pix0.5_img0.5"
ITERS_PER_EPOCH = 25000
PEAK_EPOCHS: list[int] = list(range(40))  # model_epoch0 .. model_epoch39 (25k .. 1M)
COOLDOWN_ITERS = 25000
# Same points as slurm/duration_launch_cooldowns.sh (225k is omitted).
COOLDOWN_SOURCE_STEPS: list[int] = [
    25000,
    75000,
    125000,
    175000,
    275000,
    375000,
    475000,
    575000,
    675000,
    775000,
    875000,
    975000,
]
RESOLUTIONS: list[int] = [256, 512]

ID_COLUMNS = [
    "cooldown",
    "constant_lr_steps",
    "total_steps",
    "ckpt_epoch",
    "resolution",
    "run",
    "done",
]
ALL_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

_DIR_RE = re.compile(r"^epoch(?P<epoch>\d+)_res(?P<res>\d+)$")
_CD_RUN_RE = re.compile(rf"^{re.escape(MAIN_RUN)}_cd(?P<steps>\d+)$")


def peak_steps(epoch: int) -> int:
    """Constant-LR steps at which ``model_epoch{epoch}`` was written."""
    return (epoch + 1) * ITERS_PER_EPOCH


def ckpt_epoch_for_steps(steps: int) -> int:
    """``model_epoch`` index of the constant-LR checkpoint taken at ``steps``."""
    return steps // ITERS_PER_EPOCH - 1


def cooldown_run_name(steps: int) -> str:
    return f"{MAIN_RUN}_cd{steps}"


def _load_metrics(out_dir: Path) -> tuple[bool, dict[str, float]]:
    """Return ``(done, metrics)``. Metrics stay NaN unless `.done` and a dict summary."""
    metrics = _empty_metrics()
    if not (out_dir / ".done").exists():
        return False, metrics

    summary_path = out_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return True, metrics
    if not isinstance(summary, dict):
        return True, metrics
    return True, _metrics_from_summary(summary)


def _row(
    *,
    cooldown: bool,
    steps: int,
    resolution: int,
    run: str,
    out_dir: Path,
) -> dict:
    done, metrics = _load_metrics(out_dir)
    row: dict[str, object] = {
        "cooldown": cooldown,
        "constant_lr_steps": steps,
        "total_steps": steps + (COOLDOWN_ITERS if cooldown else 0),
        "ckpt_epoch": ckpt_epoch_for_steps(steps),
        "resolution": resolution,
        "run": run,
        "done": done,
    }
    row.update(metrics)
    return row


def _iter_cells(root: Path) -> list[tuple[str, int, int, Path]]:
    """On-disk ``(run, dir_epoch, resolution, path)`` cells under ``root``."""
    found: list[tuple[str, int, int, Path]] = []
    if not root.is_dir():
        return found
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            match = _DIR_RE.match(cell_dir.name)
            if not match:
                continue
            found.append(
                (run_dir.name, int(match.group("epoch")), int(match.group("res")), cell_dir)
            )
    return found


def _interpret_cell(run: str, dir_epoch: int) -> tuple[bool, int] | None:
    """Map a directory to ``(cooldown, constant_lr_steps)``, or None if unrecognized."""
    cd = _CD_RUN_RE.match(run)
    if cd:
        return True, int(cd.group("steps"))
    if run == MAIN_RUN:
        return False, peak_steps(dir_epoch)
    return None


def build_rows(peak_root: Path, cooldown_root: Path) -> list[dict]:
    """One row per expected grid cell, plus any extra on-disk cells outside that grid."""
    rows: list[dict] = []
    seen: set[tuple[str, int, int]] = set()

    for epoch in PEAK_EPOCHS:
        steps = peak_steps(epoch)
        for res in RESOLUTIONS:
            seen.add((MAIN_RUN, epoch, res))
            rows.append(
                _row(
                    cooldown=False,
                    steps=steps,
                    resolution=res,
                    run=MAIN_RUN,
                    out_dir=peak_root / MAIN_RUN / f"epoch{epoch}_res{res}",
                )
            )

    for steps in COOLDOWN_SOURCE_STEPS:
        run = cooldown_run_name(steps)
        for res in RESOLUTIONS:
            seen.add((run, 0, res))
            rows.append(
                _row(
                    cooldown=True,
                    steps=steps,
                    resolution=res,
                    run=run,
                    out_dir=cooldown_root / run / f"epoch0_res{res}",
                )
            )

    for root in (peak_root, cooldown_root):
        for run, dir_epoch, res, cell_dir in _iter_cells(root):
            key = (run, dir_epoch, res)
            if key in seen:
                continue
            interpreted = _interpret_cell(run, dir_epoch)
            if interpreted is None:
                continue
            seen.add(key)
            cooldown, steps = interpreted
            rows.append(
                _row(
                    cooldown=cooldown,
                    steps=steps,
                    resolution=res,
                    run=run,
                    out_dir=cell_dir,
                )
            )

    rows.sort(
        key=lambda r: (int(r["constant_lr_steps"]), bool(r["cooldown"]), int(r["resolution"]))
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
        "--peak-root",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "duration_peak",
        help="Eval outputs for constant-LR checkpoints (no cooldown).",
    )
    p.add_argument(
        "--cooldown-root",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "duration_cooldown",
        help="Eval outputs for the 25k linear-cooldown checkpoints.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "duration_compiled.csv",
        help="CSV output path.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_rows(args.peak_root, args.cooldown_root)
    write_csv(rows, args.output)

    def _count(cooldown: bool) -> tuple[int, int]:
        subset = [r for r in rows if r["cooldown"] is cooldown]
        return sum(1 for r in subset if r["done"]), len(subset)

    peak_done, peak_total = _count(False)
    cd_done, cd_total = _count(True)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  peak:     {peak_done}/{peak_total} done")
    print(f"  cooldown: {cd_done}/{cd_total} done")


if __name__ == "__main__":
    main()
