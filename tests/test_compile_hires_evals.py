"""Grid compilation for the higher-resolution fine-tune."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from scripts.compile_hires_evals import (
    ACCUM_STEPS,
    CHECKPOINT_EVERY,
    FINETUNE_STEPS,
    MICROBATCHES,
    PRED_RESOLUTIONS,
    SOURCE_STEPS,
    TOTAL_STEPS,
    TRAIN_GRID,
    TRAIN_RESOLUTIONS,
    build_parser,
    build_rows,
    run_name,
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


def _yaml_int(text: str, key: str) -> int:
    match = re.search(rf"^ +{key}: (\d+)\b", text, re.MULTILINE)
    assert match is not None, key
    return int(match.group(1))


def _paren_list(text: str, name: str) -> list[str]:
    match = re.search(rf"^{name}=\(([^)]*)\)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1).split()


def _as_256(text: str, size: int) -> str:
    return (
        text.replace(f"size: [{size}, {size}]", "size: [256, 256]")
        .replace(f"width: {size}", "width: 256")
        .replace(f"height: {size}", "height: 256")
    )


def test_grid_matches_the_train_sbatch_and_config():
    sbatch = (_REPO / "slurm" / "hires_train.sbatch").read_text()
    assert _paren_list(sbatch, "RESOLUTIONS") == [str(r) for r in TRAIN_RESOLUTIONS]
    assert _paren_list(sbatch, "MICROBATCHES") == [str(m) for m in MICROBATCHES]
    assert _paren_list(sbatch, "ACCUM_STEPS") == [str(a) for a in ACCUM_STEPS]
    assert "--array=0-3" in sbatch
    assert "--partition=vision-beery-a100" in sbatch
    assert "--cpus-per-task=32" in sbatch
    assert "--mem=96G" in sbatch
    assert "--time=71:59:00" in sbatch
    assert "MERMAIDSEG_NUM_WORKERS=28" in sbatch
    assert "--init-weights" in sbatch
    assert "--init-checkpoint" not in sbatch
    assert "DATA_SEED=3004" in sbatch
    assert "RESTART_DATA_SEED=4004" in sbatch
    assert "DATA_SEED=$RESTART_DATA_SEED" in sbatch
    assert "data_seed=${DATA_SEED}" in sbatch
    assert "cbm_duration_pix0.5_img0.5_cd675000/model_epoch0" in sbatch
    assert "WANDB_RUN_GROUP=cbm_hires" in sbatch
    assert "data_config_hires_" in sbatch
    assert 'RUN="${RUN_PREFIX}_res${RES}_mb${MICROBATCH}"' in sbatch
    for _resolution, microbatch, accum in TRAIN_GRID:
        assert microbatch * accum == 32

    training = (_REPO / "configs" / "training_config_cbm_hires.yaml").read_text()
    assert _yaml_int(training, "iterations_per_train_epoch") == TOTAL_STEPS
    assert _yaml_int(training, "checkpoint_every_steps") == CHECKPOINT_EVERY
    assert _yaml_int(training, "warmup_iters") == 4000
    assert _yaml_int(training, "total_iters") == TOTAL_STEPS - 4000
    assert list(range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY)) == FINETUNE_STEPS
    assert FINETUNE_STEPS[-1] == TOTAL_STEPS

    eval_sbatch = (_REPO / "slurm" / "hires_eval.sbatch").read_text()
    assert "--partition=vision-beery-l40s" in eval_sbatch
    assert "--cpus-per-task=32" in eval_sbatch
    assert "--mem=64G" in eval_sbatch
    assert "--time=4:00:00" in eval_sbatch
    assert "--num-workers 16" in eval_sbatch
    assert "configs/data_config_dedup.yaml" in eval_sbatch
    assert "EVAL_BATCH=8" in eval_sbatch
    assert "EVAL_BATCH=2" in eval_sbatch
    assert "EVAL_BATCH=1" in eval_sbatch
    for pred in PRED_RESOLUTIONS:
        assert str(pred) in eval_sbatch

    launch_train = (_REPO / "slurm" / "hires_launch_train.sh").read_text()
    assert _paren_list(launch_train, "RESOLUTIONS") == [str(r) for r in TRAIN_RESOLUTIONS]
    assert _paren_list(launch_train, "MICROBATCHES") == [str(m) for m in MICROBATCHES]
    assert f"TARGET_STEP={TOTAL_STEPS}" in launch_train
    assert "latest >= TARGET_STEP" in launch_train
    assert "complete at step" in launch_train
    assert "resume from step" in launch_train

    launch_eval = (_REPO / "slurm" / "hires_launch_evals.sh").read_text()
    assert 'CKPT_GLOB="nbs/model_checkpoints/cbm_hires_*/model_step*"' in launch_eval
    assert 'OUT_BASE="eval_out/hires"' in launch_eval
    assert _paren_list(launch_eval, "RESOLUTIONS") == [str(r) for r in PRED_RESOLUTIONS]
    assert "cbm_vitsize" not in launch_eval
    assert "cbm_duration" not in launch_eval

    model_input = {
        256: _REPO / "configs" / "model_config_cbm_dpt_lora_vitl_256.yaml",
        384: _REPO / "configs" / "model_config_cbm_dpt_lora_vitl_384.yaml",
        512: _REPO / "configs" / "model_config_cbm_dpt_lora_vitl.yaml",
        768: _REPO / "configs" / "model_config_cbm_dpt_lora_vitl_768.yaml",
        1024: _REPO / "configs" / "model_config_cbm_dpt_lora_vitl_1024.yaml",
    }
    for pred, path in model_input.items():
        assert path.is_file()
        assert f"input_size: [{pred}, {pred}]" in path.read_text()
        assert "lora_r: 16" in path.read_text()

    args = build_parser().parse_args([])
    assert args.root.name == "hires"
    assert args.output is None


def test_data_configs_match_dedup_except_spatial_size():
    base = (_REPO / "configs" / "data_config_dedup.yaml").read_text()
    for size in (256, 384, 512, 768):
        text = (_REPO / "configs" / f"data_config_hires_{size}.yaml").read_text()
        assert "max_size: 1024" in text
        assert "2562" in text
        assert _as_256(text, size) == base
        if size != 256:
            assert f"size: [{size}, {size}]" in text
            assert "size: [256, 256]" not in text


def test_empty_root_is_the_full_grid(tmp_path: Path):
    rows = build_rows(tmp_path / "hires")
    assert len(rows) == len(TRAIN_GRID) * len(FINETUNE_STEPS) * len(PRED_RESOLUTIONS)
    assert {row["run"] for row in rows} == {run_name(res, mb) for res, mb, _accum in TRAIN_GRID}
    assert all(row["done"] is False for row in rows)
    assert all(math.isnan(row["coralnet_class_accuracy"]) for row in rows)
    sample = [
        row["finetune_steps"]
        for row in rows
        if row["train_resolution"] == 768 and row["pred_resolution"] == 1024
    ]
    assert sample == FINETUNE_STEPS
    hit = next(
        row for row in rows if row["run"] == run_name(384, 16) and row["finetune_steps"] == 10000
    )
    assert hit["microbatch"] == 16
    assert hit["grad_accum_steps"] == 2
    assert hit["total_steps"] == SOURCE_STEPS + 10000


def test_done_cell_fills_metrics_and_other_runs_are_ignored(tmp_path: Path):
    root = tmp_path / "hires"
    done = root / run_name(512, 8) / "step50000_res768"
    _write_done(done, _summary())
    other = root / "cbm_vitsize_vitl" / "step50000_res256"
    _write_done(other, _summary())
    extra = root / run_name(256, 32) / "step5000_res1024"
    _write_done(extra, _summary())

    rows = build_rows(root)
    by_key = {(row["run"], row["finetune_steps"], row["pred_resolution"]): row for row in rows}
    hit = by_key[(run_name(512, 8), 50_000, 768)]
    assert hit["done"] is True
    assert hit["train_resolution"] == 512
    assert hit["grad_accum_steps"] == 4
    assert hit["total_steps"] == SOURCE_STEPS + 50_000
    assert hit["coralnet_class_accuracy"] == 0.5
    missing = by_key[(run_name(256, 32), 10_000, 256)]
    assert missing["done"] is False
    assert math.isnan(missing["coralnet_class_accuracy"])
    assert all(row["run"] != "cbm_vitsize_vitl" for row in rows)
    extra_row = by_key[(run_name(256, 32), 5_000, 1024)]
    assert extra_row["done"] is True
    assert extra_row["total_steps"] == SOURCE_STEPS + 5_000
