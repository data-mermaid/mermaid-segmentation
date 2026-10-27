"""Grid compilation for the ViT-size sweep."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from scripts.compile_vitsize_evals import (
    CHECKPOINT_EVERY,
    RESOLUTIONS,
    STEPS,
    TOTAL_STEPS,
    VIT_SIZES,
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


def test_grid_matches_the_train_sbatch_and_config():
    sbatch = (_REPO / "slurm" / "vitsize_train.sbatch").read_text()
    sizes = re.search(r"^SIZES=\(([^)]*)\)$", sbatch, re.MULTILINE)
    assert sizes is not None
    assert sizes.group(1).split() == VIT_SIZES
    assert "--array=0-3" in sbatch
    assert "--cpus-per-task=32" in sbatch
    assert "--mem=96G" in sbatch
    assert "--time=71:59:00" in sbatch
    assert "MERMAIDSEG_NUM_WORKERS=28" in sbatch
    assert "--batch-size 64" in sbatch
    assert "--grad-accum-steps" not in sbatch

    training = (_REPO / "configs" / "training_config_cbm_vitsize.yaml").read_text()
    assert _yaml_int(training, "iterations_per_train_epoch") == TOTAL_STEPS
    assert _yaml_int(training, "checkpoint_every_steps") == CHECKPOINT_EVERY
    assert _yaml_int(training, "warmup_iters") == 4000
    assert _yaml_int(training, "total_iters") == TOTAL_STEPS - 4000
    assert list(range(CHECKPOINT_EVERY, TOTAL_STEPS + 1, CHECKPOINT_EVERY)) == STEPS
    assert STEPS[-1] == TOTAL_STEPS

    for vit in VIT_SIZES:
        if vit == "vitl":
            train_cfg = _REPO / "configs" / "model_config_cbm_dpt_lora_vitl_256.yaml"
            eval_cfg = _REPO / "configs" / "model_config_cbm_dpt_lora_vitl.yaml"
        else:
            train_cfg = _REPO / "configs" / f"model_config_cbm_dpt_lora_{vit}_256.yaml"
            eval_cfg = _REPO / "configs" / f"model_config_cbm_dpt_lora_{vit}.yaml"
        assert train_cfg.is_file()
        assert eval_cfg.is_file()
        assert "input_size: [256, 256]" in train_cfg.read_text()
        assert "input_size: [512, 512]" in eval_cfg.read_text()

    args = build_parser().parse_args([])
    assert args.root.name == "vitsize"
    assert args.output is None


def test_empty_root_is_the_full_grid(tmp_path: Path):
    rows = build_rows(tmp_path / "vitsize")
    assert len(rows) == len(VIT_SIZES) * len(STEPS) * len(RESOLUTIONS)
    assert {row["run"] for row in rows} == {run_name(vit) for vit in VIT_SIZES}
    assert all(row["done"] is False for row in rows)
    assert all(math.isnan(row["coralnet_class_accuracy"]) for row in rows)
    assert [
        row["steps"] for row in rows if row["vit"] == "vits" and row["resolution"] == 256
    ] == STEPS


def test_done_cell_fills_metrics_and_other_runs_are_ignored(tmp_path: Path):
    root = tmp_path / "vitsize"
    done = root / run_name("vitl") / "step150000_res256"
    _write_done(done, _summary())
    other = root / "cbm_duration_pix0.5_img0.5" / "step150000_res256"
    _write_done(other, _summary())
    extra = root / run_name("vithplus") / "step10000_res512"
    _write_done(extra, _summary())

    rows = build_rows(root)
    by_key = {(row["run"], row["steps"], row["resolution"]): row for row in rows}
    hit = by_key[(run_name("vitl"), 150_000, 256)]
    assert hit["done"] is True
    assert hit["vit"] == "vitl"
    assert hit["coralnet_class_accuracy"] == 0.5
    missing = by_key[(run_name("vits"), 25_000, 512)]
    assert missing["done"] is False
    assert math.isnan(missing["coralnet_class_accuracy"])
    assert all(row["run"] != "cbm_duration_pix0.5_img0.5" for row in rows)
    assert by_key[(run_name("vithplus"), 10_000, 512)]["done"] is True
