"""Extended evals: launcher wiring and merging extended/ outputs into the compiled CSVs."""

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path

from scripts.compile_sweep_loss_weights import EXTENDED_DIR, METRIC_COLUMNS, load_cell

_REPO = Path(__file__).resolve().parents[1]
_SLURM = _REPO / "slurm"

# Sweep name in slurm/extended_launch_evals.sh -> that sweep's own eval launcher.
_SWEEP_LAUNCHERS = {
    "lossw": "sweep_loss_weights_launch_evals.sh",
    "duration_peak": "duration_launch_peak_evals.sh",
    "duration_cooldown": "duration_launch_evals.sh",
    "vitsize": "vitsize_launch_evals.sh",
    "hires": "hires_launch_evals.sh",
    "data_quality": "data_quality_launch_evals.sh",
}


def _write(out_dir: Path, summary: dict, done: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary))
    if done:
        (out_dir / ".done").write_text("")


def _binary(f1: float) -> dict:
    return {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": f1}


def _legacy_summary() -> dict:
    return {
        "coralnet": {
            "pooled": {
                "class_accuracy": 0.5,
                "bleached": _binary(0.4),
                "binary": {"per_concept": {"anthropogenic": _binary(0.3)}},
            }
        },
        "pacific": {"pooled": {"class_accuracy": 0.2}},
        "coralscapes": {
            "feature_kinds": {
                "concepts": {
                    "probe": {"train_accuracy": 0.99},
                    "test": {"accuracy": 0.6, "miou": 0.3},
                }
            },
            "bleached": _binary(0.45),
        },
    }


def _coralscapes_split(split: str, accuracy: float) -> dict:
    return {
        "split": split,
        "feature_kinds": {
            "concepts": {
                "probe": {"train_accuracy": 0.98},
                split: {"accuracy": accuracy, "miou": 0.1},
            }
        },
        "bleached": _binary(0.5),
        "anthropogenic": _binary(0.6),
        "hierarchical": {"consistency/overall": 0.97, "ece/mean": 0.05, "severity/genus": 1.5},
    }


def _extended_summary() -> dict:
    hier = {"consistency/overall": 0.95, "consistency/family->genus": 0.9, "ece/genus": 0.1}
    return {
        "coralnet_val": {
            "pooled": {"class_accuracy": 0.55, "anthropogenic": _binary(0.35), "hierarchical": hier}
        },
        "coralnet_test": {"pooled": {"class_accuracy": 0.65, "bleached": _binary(0.25)}},
        "coralscapes_val": _coralscapes_split("val", 0.7),
        "coralscapes_test": _coralscapes_split("test", 0.61),
    }


def test_existing_columns_are_kept_and_names_are_unique():
    assert len(METRIC_COLUMNS) == len(set(METRIC_COLUMNS))
    for column in (
        "coralnet_class_accuracy",
        "coralnet_bleached_f1",
        "pacific_class_accuracy",
        "benthos_miou",
        "coralscapes_concepts_test_accuracy",
        "coralscapes_concepts_probe_train_accuracy",
        "coralscapes_bleached_f1",
        "coralnet_test_class_accuracy",
        "coralnet_test_hier_consistency_family_genus",
        "coralscapes_concepts_val_miou",
        "coralscapes_val_anthropogenic_f1",
        "coralscapes_hier_ece_mean",
        "coralnet_hier_severity_mean_normalized",
    ):
        assert column in METRIC_COLUMNS, column


def test_legacy_cell_reads_old_keys_and_per_concept_anthropogenic(tmp_path: Path):
    _write(tmp_path, _legacy_summary())
    row = load_cell(tmp_path)
    assert row["done"] is True
    assert row["done_extended"] is False
    assert row["coralnet_class_accuracy"] == 0.5
    assert row["coralnet_bleached_f1"] == 0.4
    assert row["coralnet_anthropogenic_f1"] == 0.3
    assert row["coralscapes_concepts_test_accuracy"] == 0.6
    assert row["coralscapes_concepts_probe_train_accuracy"] == 0.99
    assert row["coralscapes_bleached_f1"] == 0.45
    for column in (
        "coralnet_test_class_accuracy",
        "coralnet_hier_consistency_overall",
        "coralscapes_anthropogenic_f1",
        "coralscapes_concepts_val_accuracy",
    ):
        assert math.isnan(row[column]), column


def test_extended_summary_wins_and_fills_new_columns(tmp_path: Path):
    _write(tmp_path, _legacy_summary())
    _write(tmp_path / EXTENDED_DIR, _extended_summary())
    row = load_cell(tmp_path)
    assert row["done_extended"] is True
    assert row["coralnet_class_accuracy"] == 0.55
    assert row["coralnet_anthropogenic_f1"] == 0.35
    assert row["coralnet_hier_consistency_family_genus"] == 0.9
    assert row["coralnet_hier_ece_genus"] == 0.1
    assert math.isnan(row["coralnet_hier_ece_mean"])
    assert row["coralnet_test_class_accuracy"] == 0.65
    assert row["coralnet_test_bleached_f1"] == 0.25
    assert row["pacific_class_accuracy"] == 0.2
    assert row["coralscapes_concepts_test_accuracy"] == 0.61
    assert row["coralscapes_concepts_val_accuracy"] == 0.7
    assert row["coralscapes_concepts_probe_train_accuracy"] == 0.98
    assert row["coralscapes_val_anthropogenic_f1"] == 0.6
    assert row["coralscapes_hier_severity_genus"] == 1.5
    assert row["coralscapes_val_hier_consistency_overall"] == 0.97


def test_unfinished_or_failed_extended_evals_fall_back_to_the_original(tmp_path: Path):
    _write(tmp_path, _legacy_summary())
    _write(tmp_path / EXTENDED_DIR, _extended_summary(), done=False)
    row = load_cell(tmp_path)
    assert row["done_extended"] is False
    assert row["coralnet_class_accuracy"] == 0.5
    assert math.isnan(row["coralnet_test_class_accuracy"])

    failed = {**_extended_summary(), "coralnet_val": {"error": "boom"}}
    _write(tmp_path / EXTENDED_DIR, failed)
    row = load_cell(tmp_path)
    assert row["done_extended"] is True
    assert row["coralnet_class_accuracy"] == 0.5
    assert row["coralnet_test_class_accuracy"] == 0.65


def _launcher_var(text: str, name: str) -> str:
    match = re.search(rf'^{name}=(?:"([^"]*)"|\(([^)]*)\))$', text, re.MULTILINE)
    assert match is not None, name
    return match.group(1) if match.group(1) is not None else match.group(2)


def test_launcher_table_matches_each_sweep_launcher_and_eval_sbatch():
    text = (_SLURM / "extended_launch_evals.sh").read_text()
    specs = re.findall(
        r'^\s+"([a-z0-9_]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^"]+)"$', text, re.MULTILINE
    )
    assert {spec[0] for spec in specs} == set(_SWEEP_LAUNCHERS)
    for sweep, ckpt_glob, eval_sbatch, out_base, resolutions in specs:
        launcher = (_SLURM / _SWEEP_LAUNCHERS[sweep]).read_text()
        assert _launcher_var(launcher, "CKPT_GLOB") == ckpt_glob, sweep
        assert _launcher_var(launcher, "EVAL_SBATCH") == eval_sbatch, sweep
        assert _launcher_var(launcher, "OUT_BASE") == out_base, sweep
        assert _launcher_var(launcher, "RESOLUTIONS").split() == resolutions.split(), sweep
        sbatch = (_REPO / eval_sbatch).read_text()
        assert "#SBATCH --qos=vision-beery-free\n#SBATCH --requeue\n" in sbatch, eval_sbatch
        assert 'EVAL_ARGS="${4:---evals all}"' in sbatch, eval_sbatch
        assert "    $EVAL_ARGS \\" in sbatch, eval_sbatch
        assert "--evals all \\" not in sbatch, eval_sbatch
    assert f'"$out_base/$run/${{tag}}_res${{res}}/{EXTENDED_DIR}"' in text


def test_launcher_eval_args_parse():
    from scripts.evaluate import build_parser

    eval_args = _launcher_var((_SLURM / "extended_launch_evals.sh").read_text(), "EVAL_ARGS")
    args = build_parser().parse_args(
        ["--checkpoint", "c", "--output-dir", "o", *shlex.split(eval_args)]
    )
    assert args.evals == ["coralnet", "coralscapes"]
    assert args.coralnet_splits == ["val", "test"]
    assert args.coralscapes_splits == ["val", "test"]
