#!/usr/bin/env python3
"""Compile the data-quality sweep's per-eval outputs into a single CSV.

Meant to run on the cluster checkout (``$SCRATCH/mermaid-segmentation``), where
the eval jobs write their results. Each run trains on one data-quality profile
(scripts/build_data_quality_profile.py), and each of its evals lands in

    eval_out/data_quality/cbm_dq_<profile>/step<N>_res<R>/

(see slurm/data_quality_eval.sbatch) and is only considered finished once
``<out>/.done`` exists. The profiles are those with eval outputs plus every profile
built under ``<dq-root>/profiles``, so a built profile without evals still gets its
rows. The sweep is the ViT-size sweep's ViT-L recipe trained on degraded data, so its
baseline is ``cbm_vitsize_vitl``, read from ``eval_out/vitsize/cbm_vitsize_vitl/`` and
reported as profile ``baseline``. The grid is {baseline, profiles} x steps
25000..150000 x {256, 512}, plus any other ``step*_res*`` cell of a profile on disk.

Every row describes its profile: ``sparse`` and ``phylum_q_<metric>`` (parsed from the
name; the baseline has no degradation, so false and 0), and, read from the phylum
component's ``run.json``, ``phylum_n_flagged_sources``, ``phylum_n_changed_points`` and
``phylum_pct_flagged_points_changed`` (NaN when that file can't be read).

Headline metrics match ``scripts/compile_sweep_loss_weights.py`` and are filled
only when ``.done`` exists and ``summary.json`` parses. Every other cell stays
in the CSV with metric columns as NaN.

Usage:
    python scripts/compile_data_quality_evals.py
    python scripts/compile_data_quality_evals.py \\
        --root eval_out/data_quality \\
        --baseline-root eval_out/vitsize \\
        --dq-root /data/vision/beery/scratch/sauder/mermaid-segmentation/data_quality \\
        --output eval_out/data_quality/compiled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_data_quality_profile import (  # noqa: E402
    DATA_CONFIG_FILE,
    DEFAULT_DQ_ROOT,
    PROFILES_DIR,
    parse_profile_name,
    phylum_component_dir,
)
from compile_sweep_loss_weights import (  # noqa: E402
    METRIC_COLUMNS,
    STATUS_COLUMNS,
    _format_value,
    load_cell,
)
from compile_vitsize_evals import RESOLUTIONS, STEPS  # noqa: E402
from compile_vitsize_evals import run_name as vitsize_run_name  # noqa: E402
from coralnet_quantile_exploration import METRIC_KEYS  # noqa: E402

RUN_PREFIX = "cbm_dq"
BASELINE = "baseline"
BASELINE_RUN = vitsize_run_name("vitl")

PHYLUM_Q_COLUMNS = [f"phylum_q_{key}" for key in METRIC_KEYS]
PHYLUM_STAT_COLUMNS = [
    "phylum_n_flagged_sources",
    "phylum_n_changed_points",
    "phylum_pct_flagged_points_changed",
]
PROFILE_COLUMNS = ["sparse", *PHYLUM_Q_COLUMNS, *PHYLUM_STAT_COLUMNS]
ID_COLUMNS = ["run", "profile", *PROFILE_COLUMNS, "steps", "resolution", *STATUS_COLUMNS]
ALL_COLUMNS = ID_COLUMNS + METRIC_COLUMNS

_DIR_RE = re.compile(r"^step(?P<step>\d+)_res(?P<res>\d+)$")


def run_name(profile: str) -> str:
    return f"{RUN_PREFIX}_{profile}"


def profile_from_run(run: str) -> str | None:
    """Profile a ``cbm_dq_<profile>`` run trains on; None for any other run."""
    profile = run.removeprefix(f"{RUN_PREFIX}_")
    if profile == run:
        return None
    try:
        parse_profile_name(profile)
    except ValueError:
        return None
    return profile


def discover_profiles(root: Path, dq_root: Path) -> list[str]:
    """Profiles with eval outputs under ``root`` or built under ``dq_root``."""
    found: set[str] = set()
    if root.is_dir():
        for run_dir in root.iterdir():
            profile = profile_from_run(run_dir.name) if run_dir.is_dir() else None
            if profile is not None:
                found.add(profile)
    profiles_dir = dq_root / PROFILES_DIR
    if profiles_dir.is_dir():
        for profile_dir in profiles_dir.iterdir():
            built = (profile_dir / DATA_CONFIG_FILE).is_file()
            if built and profile_from_run(run_name(profile_dir.name)) is not None:
                found.add(profile_dir.name)
    return sorted(found)


def _phylum_stats(dq_root: Path, quantiles: Sequence[float] | None) -> dict[str, object]:
    if quantiles is None:
        return dict(zip(PHYLUM_STAT_COLUMNS, (0, 0, 0.0), strict=True))
    try:
        info = json.loads((phylum_component_dir(dq_root, quantiles) / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        return dict.fromkeys(PHYLUM_STAT_COLUMNS, math.nan)
    return {
        "phylum_n_flagged_sources": len(info["flagged_sources"]),
        "phylum_n_changed_points": info["n_changed_points"],
        "phylum_pct_flagged_points_changed": info["pct_flagged_points_changed"],
    }


def profile_columns(profile: str, dq_root: Path) -> dict[str, object]:
    sparse, quantiles = (False, None) if profile == BASELINE else parse_profile_name(profile)
    return {
        "sparse": sparse,
        **dict(zip(PHYLUM_Q_COLUMNS, quantiles or [0.0] * len(METRIC_KEYS), strict=True)),
        **_phylum_stats(dq_root, quantiles),
    }


def _cells_on_disk(run_dir: Path) -> set[tuple[int, int]]:
    """``(steps, resolution)`` of every ``step*_res*`` dir under ``run_dir``."""
    if not run_dir.is_dir():
        return set()
    cells = set()
    for cell_dir in run_dir.iterdir():
        match = _DIR_RE.match(cell_dir.name)
        if match and cell_dir.is_dir():
            cells.add((int(match.group("step")), int(match.group("res"))))
    return cells


def build_rows(root: Path, baseline_root: Path, dq_root: Path) -> list[dict]:
    """Baseline grid, plus each profile's grid and its other cells on disk."""
    grid = {(steps, res) for steps in STEPS for res in RESOLUTIONS}
    runs = [(BASELINE_RUN, BASELINE, baseline_root / BASELINE_RUN, grid)]
    for profile in discover_profiles(root, dq_root):
        run_dir = root / run_name(profile)
        runs.append((run_name(profile), profile, run_dir, grid | _cells_on_disk(run_dir)))

    rows: list[dict] = []
    for run, profile, run_dir, cells in runs:
        columns = profile_columns(profile, dq_root)
        for steps, res in cells:
            row = {"run": run, "profile": profile, **columns, "steps": steps, "resolution": res}
            row.update(load_cell(run_dir / f"step{steps}_res{res}"))
            rows.append(row)
    rows.sort(key=lambda r: (r["profile"] != BASELINE, r["profile"], r["steps"], r["resolution"]))
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
        default=_REPO_ROOT / "eval_out" / "data_quality",
        help="Directory holding the data-quality runs' eval outputs.",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=_REPO_ROOT / "eval_out" / "vitsize",
        help=f"Directory holding the {BASELINE_RUN} baseline's eval outputs.",
    )
    p.add_argument(
        "--dq-root",
        type=Path,
        default=DEFAULT_DQ_ROOT,
        help="Data-quality root holding the built profiles and their components.",
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
    rows = build_rows(args.root, args.baseline_root, args.dq_root)
    write_csv(rows, output)
    n_done = sum(1 for row in rows if row["done"])
    n_extended = sum(1 for row in rows if row["done_extended"])
    n_profiles = len({row["profile"] for row in rows}) - 1
    print(
        f"Wrote {len(rows)} rows for the baseline and {n_profiles} profile(s) to {output} "
        f"({n_done} done, {len(rows) - n_done} missing/incomplete -> NaN; "
        f"{n_extended} extended done)."
    )


if __name__ == "__main__":
    main()
