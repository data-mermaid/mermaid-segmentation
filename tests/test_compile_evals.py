"""scripts/compile_evals.py on synthetic eval cells."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.compile_evals import (
    COMPILED_CSV,
    PER_SOURCE_CSV,
    build_rows,
    checkpoint_step,
    main,
    run_params,
)

POOLED = {
    "class_accuracy": 0.8,
    "class_miou": 0.4,
    "binary": {"macro_accuracy": 0.9, "macro_f1": 0.7},
    "taxonomic": {"genus": {"acc_all": 0.6, "macro_f1_coral": 0.3}},
    "bleached": {"accuracy": 0.95, "f1": 0.5},
    "hierarchical": {"consistency/overall": 0.99, "severity/mean_normalized": 0.1},
}
SUMMARY = {
    "checkpoint": "nbs/model_checkpoints/x/model_step100000",
    "coralnet_val": {"pooled": POOLED, "macro": {"class_accuracy": 0.7}, "images_processed": 10},
    "pacific": {"pooled": POOLED, "macro": {"class_accuracy": 0.6}, "images_processed": 5},
    "coralscapes_val": {
        "feature_kinds": {"concepts": {"val": {"accuracy": 0.5, "miou": 0.2}, "probe": {"k": 10}}},
        "bleached": {"f1": 0.4},
        "hierarchical": {"ece/mean": 0.05},
    },
    "coralscapes39_val": {
        "by_k": {"50": {"val": {"accuracy": 0.55, "miou": 0.25}, "probe": {"k": 50}}}
    },
    "val_loss": {
        "overall": {"loss": 1.5, "genus": 0.3, "multi_hot/bleached": 0.1, "multi_hot/sand": 0.2},
        "per_dataset": {
            "coralnet": {"loss": 1.2, "multi_hot/bleached": 0.05, "multi_hot/sand": 0.3},
        },
    },
}


def _cell(root: Path, family: str, run: str, cell: str, summary: dict | None = None) -> Path:
    path = root / family / run / cell
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary or SUMMARY))
    (path / ".done").touch()
    return path


def test_checkpoint_steps():
    assert checkpoint_step("cbm_lossw_pix0.5_img0.5", "model_step100000") == 100000
    assert checkpoint_step("cbm_duration_peak", "model_epoch10") == 275000
    with pytest.raises(ValueError, match="only compiled for the peak run"):
        checkpoint_step("cbm_tax_kl", "model_epoch3")


@pytest.mark.parametrize(
    ("run", "step", "expected"),
    [
        ("cbm_lossw_pix0.9_img0.1", 100000, {"pixel_weight": 0.9, "image_weight": 0.1}),
        ("cbm_tax_tree", 100000, {"tax_loss": "tree", "tax_weight": 0.5}),
        ("cbm_dsize_exp3_drop20", 100000, {"ds_experiment": "exp3", "ds_drop": 20}),
        ("cbm_sparse", 100000, {"ds_experiment": "sparse"}),
        ("cbm_vitsize_vithplus", 100000, {"vit_size": "vithplus"}),
        ("cbm_duration_peak", 275000, {"duration_phase": "peak", "duration_branch_step": 275000}),
        (
            "cbm_duration_cd275000",
            25000,
            {"duration_phase": "cooldown", "duration_branch_step": 275000, "train_steps": 300000},
        ),
        ("cbm_hires_res768", 25000, {"train_resolution": 768, "train_steps": 325000}),
    ],
)
def test_run_params(run, step, expected):
    params = run_params(run, step)
    assert {key: params[key] for key in expected} == expected


def test_rows_flatten_every_section(tmp_path: Path):
    cell = _cell(tmp_path, "lossw", "cbm_lossw_pix0.5_img0.5", "model_step100000_res512")
    (cell / "coralnet_val").mkdir()
    (cell / "coralnet_val" / "per_unit.csv").write_text("unit,class_accuracy\n4,0.7\n5,0.8\n")
    _cell(tmp_path, "duration", "cbm_duration_peak", "model_epoch0_res512")
    unfinished = tmp_path / "lossw" / "cbm_lossw_pix1.0_img0.0" / "model_step100000_res512"
    unfinished.mkdir(parents=True)
    (unfinished / "summary.json").write_text(json.dumps(SUMMARY))

    rows, source_rows = build_rows(tmp_path)
    assert [(row["run"], row["step"]) for row in rows] == [
        ("cbm_duration_peak", 25000),
        ("cbm_lossw_pix0.5_img0.5", 100000),
    ]
    row = rows[1]
    assert row["family"] == "lossw" and row["resolution"] == 512
    assert row["coralnet/pooled/class_accuracy"] == 0.8
    assert row["coralnet/pooled/tax_genus_acc_all"] == 0.6
    assert row["coralnet/pooled/bleached_f1"] == 0.5
    assert row["coralnet/pooled/hier/consistency/overall"] == 0.99
    assert row["coralnet/macro/class_accuracy"] == 0.7
    assert row["pacific/macro/class_accuracy"] == 0.6
    assert row["coralscapes/miou"] == 0.2
    assert row["coralscapes/probe_k"] == 10
    assert row["coralscapes/bleached_f1"] == 0.4
    assert row["coralscapes/hier/ece/mean"] == 0.05
    assert row["coralscapes39/k50/miou"] == 0.25
    assert row["val_loss/loss"] == 1.5
    assert row["val_loss/multi_hot/sand"] == 0.2
    assert row["val_loss/coralnet/multi_hot/bleached"] == 0.05
    assert "val_loss/coralnet/multi_hot/sand" not in row

    assert [(r["run"], r["split"], r["unit"]) for r in source_rows] == [
        ("cbm_lossw_pix0.5_img0.5", "coralnet_val", "4"),
        ("cbm_lossw_pix0.5_img0.5", "coralnet_val", "5"),
    ]


def test_failed_sections_are_left_out(tmp_path: Path):
    summary = {**SUMMARY, "pacific": {"error": "boom"}, "val_loss": {"error": "boom"}}
    _cell(tmp_path, "taxloss", "cbm_tax_kl", "model_step100000_res512", summary)
    (row,), _ = build_rows(tmp_path)
    assert "coralnet/pooled/class_accuracy" in row
    assert not any(key.startswith(("pacific/", "val_loss/")) for key in row)


def test_main_writes_both_tables(tmp_path: Path):
    _cell(tmp_path, "hires", "cbm_hires_res512", "model_step25000_res1024")
    main(["--eval-root", str(tmp_path)])
    with (tmp_path / COMPILED_CSV).open(newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["run"] == "cbm_hires_res512" and row["resolution"] == "1024"
    assert row["train_resolution"] == "512" and row["train_steps"] == "325000"
    with (tmp_path / PER_SOURCE_CSV).open(newline="") as f:
        assert list(csv.DictReader(f)) == []
