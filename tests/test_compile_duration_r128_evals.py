"""Grid compilation for the rank-128 duration evals."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from scripts.compile_duration_evals import COOLDOWN_SOURCE_STEPS as R16_COOLDOWN_STEPS
from scripts.compile_duration_evals import MAIN_RUN as R16_MAIN_RUN
from scripts.compile_duration_r128_evals import (
    COOLDOWN_SOURCE_STEPS,
    MAIN_RUN,
    PEAK_EPOCHS,
    build_parser,
    build_rows,
)

_REPO = Path(__file__).resolve().parents[1]


def _write_done(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary))
    (out_dir / ".done").write_text("")


def _summary() -> dict:
    return {
        "coralnet": {
            "pooled": {
                "class_accuracy": 0.5,
                "class_miou": 0.25,
                "bleached": {"accuracy": 0.1, "precision": 0.2, "recall": 0.3, "f1": 0.4},
            }
        }
    }


def test_targets_the_rank128_run_and_not_the_rank16_csv():
    text = (_REPO / "slurm" / "duration_r128_train.sbatch").read_text()
    assert re.search(r'^RUN="([^"]+)"', text, re.MULTILINE).group(1) == MAIN_RUN
    assert MAIN_RUN != R16_MAIN_RUN

    launcher = (_REPO / "slurm" / "duration_r128_launch_cooldowns.sh").read_text()
    match = re.search(r"^STEPS=\(([^)]*)\)", launcher, re.MULTILINE)
    assert match is not None
    assert [int(part) for part in match.group(1).split()] == COOLDOWN_SOURCE_STEPS
    assert COOLDOWN_SOURCE_STEPS == R16_COOLDOWN_STEPS

    args = build_parser().parse_args([])
    assert args.peak_root.name == "duration_r128_peak"
    assert args.cooldown_root.name == "duration_r128_cooldown"
    assert args.output.name == "duration_r128_compiled.csv"


def test_empty_roots_use_the_rank128_run_name(tmp_path: Path):
    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    n_peak = len(PEAK_EPOCHS) * 2
    n_cd = len(COOLDOWN_SOURCE_STEPS) * 2
    assert len(rows) == n_peak + n_cd
    assert {row["run"] for row in rows if row["cooldown"] is False} == {MAIN_RUN}
    assert all(str(row["run"]).startswith(f"{MAIN_RUN}_cd") for row in rows if row["cooldown"])


def test_rank16_directories_are_ignored(tmp_path: Path):
    rank128 = tmp_path / "peak" / MAIN_RUN / "epoch0_res256"
    _write_done(rank128, _summary())
    rank16 = tmp_path / "peak" / R16_MAIN_RUN / "epoch0_res256"
    _write_done(rank16, _summary())

    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    by_key = {(row["run"], row["constant_lr_steps"], row["resolution"]): row for row in rows}
    assert by_key[(MAIN_RUN, 25_000, 256)]["done"] is True
    assert by_key[(MAIN_RUN, 25_000, 256)]["coralnet_class_accuracy"] == 0.5
    assert all(row["run"] != R16_MAIN_RUN for row in rows)
    assert all(math.isnan(row["coralnet_class_accuracy"]) or row["run"] == MAIN_RUN for row in rows)
