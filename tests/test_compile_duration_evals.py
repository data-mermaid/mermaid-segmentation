"""Grid compilation for peak-LR and cooldown duration evals."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from scripts.compile_duration_evals import (
    COOLDOWN_SOURCE_STEPS,
    ITERS_PER_EPOCH,
    MAIN_RUN,
    PEAK_EPOCHS,
    build_rows,
    ckpt_epoch_for_steps,
    peak_steps,
    write_csv,
)
from scripts.compile_sweep_loss_weights import METRIC_COLUMNS

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
        },
        "pacific": {"error": "missing"},
    }


def test_cooldown_steps_match_launcher():
    text = (_REPO / "slurm" / "duration_launch_cooldowns.sh").read_text()
    match = re.search(r"^STEPS=\(([^)]*)\)", text, re.MULTILINE)
    assert match is not None
    assert [int(part) for part in match.group(1).split()] == COOLDOWN_SOURCE_STEPS


def test_peak_grid_matches_duration_config():
    text = (_REPO / "configs" / "training_config_cbm_duration.yaml").read_text()
    epochs = int(re.search(r"^    epochs: (\d+)$", text, re.MULTILINE).group(1))
    iters = int(re.search(r"^    iterations_per_train_epoch: (\d+)$", text, re.MULTILINE).group(1))
    assert list(range(epochs)) == PEAK_EPOCHS
    assert iters == ITERS_PER_EPOCH
    assert peak_steps(0) == 25_000
    assert peak_steps(epochs - 1) == epochs * iters
    assert ckpt_epoch_for_steps(75_000) == 2


def test_empty_roots_emit_full_grid_with_nan_metrics(tmp_path: Path):
    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    n_peak = len(PEAK_EPOCHS) * 2
    n_cd = len(COOLDOWN_SOURCE_STEPS) * 2
    assert len(rows) == n_peak + n_cd
    assert all(row["done"] is False for row in rows)
    assert all(math.isnan(row["coralnet_class_accuracy"]) for row in rows)

    by_key = {(row["constant_lr_steps"], row["cooldown"], row["resolution"]): row for row in rows}
    peak = by_key[(50_000, False, 512)]
    assert peak["ckpt_epoch"] == 1
    assert peak["total_steps"] == 50_000
    assert peak["run"] == MAIN_RUN

    cooled = by_key[(75_000, True, 256)]
    assert cooled["ckpt_epoch"] == 2
    assert cooled["total_steps"] == 100_000
    assert cooled["run"] == f"{MAIN_RUN}_cd75000"

    steps = [row["constant_lr_steps"] for row in rows]
    assert steps == sorted(steps)
    # At a shared duration point, the peak row comes before the cooled row.
    pair = [row for row in rows if row["constant_lr_steps"] == 25_000 and row["resolution"] == 256]
    assert [row["cooldown"] for row in pair] == [False, True]


def test_done_summary_fills_metrics_and_ignores_partial_summary(tmp_path: Path):
    peak = tmp_path / "peak" / MAIN_RUN / "epoch0_res256"
    _write_done(peak, _summary())

    # summary.json without .done is still in progress and must not be scored.
    partial = tmp_path / "cooldown" / f"{MAIN_RUN}_cd25000" / "epoch0_res256"
    partial.mkdir(parents=True)
    (partial / "summary.json").write_text(json.dumps(_summary()))

    broken = tmp_path / "peak" / MAIN_RUN / "epoch1_res512"
    broken.mkdir(parents=True)
    (broken / ".done").write_text("")
    (broken / "summary.json").write_text("{not json")

    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    by_key = {(row["constant_lr_steps"], row["cooldown"], row["resolution"]): row for row in rows}

    done = by_key[(25_000, False, 256)]
    assert done["done"] is True
    assert done["coralnet_class_accuracy"] == 0.5
    assert done["coralnet_bleached_f1"] == 0.4
    assert math.isnan(done["pacific_class_accuracy"])

    assert by_key[(25_000, True, 256)]["done"] is False
    assert math.isnan(by_key[(25_000, True, 256)]["coralnet_class_accuracy"])

    assert by_key[(50_000, False, 512)]["done"] is True
    assert math.isnan(by_key[(50_000, False, 512)]["coralnet_class_accuracy"])


def test_extra_on_disk_cells_are_included(tmp_path: Path):
    extra_peak = tmp_path / "peak" / MAIN_RUN / "epoch40_res256"
    _write_done(extra_peak, _summary())
    extra_cd = tmp_path / "cooldown" / f"{MAIN_RUN}_cd225000" / "epoch0_res512"
    _write_done(extra_cd, _summary())
    # Unrelated run directories are ignored.
    (tmp_path / "peak" / "other_run" / "epoch0_res256").mkdir(parents=True)

    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    by_key = {(row["constant_lr_steps"], row["cooldown"], row["resolution"]): row for row in rows}
    assert by_key[(1_025_000, False, 256)]["done"] is True
    assert by_key[(225_000, True, 512)]["total_steps"] == 250_000
    assert all(row["run"] != "other_run" for row in rows)


def test_write_csv_header_matches_sweep_metrics(tmp_path: Path):
    rows = build_rows(tmp_path / "peak", tmp_path / "cooldown")
    output = tmp_path / "duration_compiled.csv"
    write_csv(rows, output)
    with output.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first = next(reader)
    assert header[:7] == [
        "cooldown",
        "constant_lr_steps",
        "total_steps",
        "ckpt_epoch",
        "resolution",
        "run",
        "done",
    ]
    assert header[7:] == METRIC_COLUMNS
    assert first[0] == "false"
    assert first[6] == "false"
    assert first[7] == "nan"
