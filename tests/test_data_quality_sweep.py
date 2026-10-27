"""Wiring of the data-quality sweep: profiles, slurm scripts and eval compilation."""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import build_data_quality_profile as profiles
from scripts.build_coralnet_phylum_manifest import MANIFEST_FILE
from scripts.build_data_quality_profile import (
    DATA_CONFIG_FILE,
    DEFAULT_BASE_CONFIG,
    DEFAULT_DQ_ROOT,
    PROFILE_FILE,
    PROFILES_DIR,
    parse_profile_name,
    phylum_component_dir,
    profile_config,
    profile_name,
)
from scripts.compile_data_quality_evals import (
    ALL_COLUMNS,
    BASELINE,
    BASELINE_RUN,
    PHYLUM_Q_COLUMNS,
    PHYLUM_STAT_COLUMNS,
    RESOLUTIONS,
    STEPS,
    build_parser,
    build_rows,
    run_name,
)
from scripts.compile_data_quality_evals import main as compile_main

_REPO = Path(__file__).resolve().parents[1]
_SLURM = _REPO / "slurm"
Q = ("0", "0", "0", "0", "0", "0.1")
QF = (0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
NAME_TO_BUILD_ARGS = "sed 's/_/ /g; s/phylum /phylum --quantiles /'"


def _flatten(tree: dict, prefix: tuple = ()) -> dict[tuple, object]:
    out: dict[tuple, object] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flatten(value, (*prefix, key)))
        else:
            out[(*prefix, key)] = value
    return out


def _bash_array(text: str, name: str) -> list[str]:
    match = re.search(rf"^\s*{name}=\(([^)]*)\)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1).split()


# ---------- profile names ----------


@pytest.mark.parametrize(
    ("sparse", "quantiles", "name"),
    [
        (True, None, "sparse"),
        (False, QF, "phylum_0_0_0_0_0_0.1"),
        (True, QF, "sparse_phylum_0_0_0_0_0_0.1"),
        (False, (0.25, 0, 0, 0, 0.1, 0.01), "phylum_0.25_0_0_0_0.1_0.01"),
    ],
)
def test_profile_names_round_trip(sparse, quantiles, name):
    assert profile_name(sparse, quantiles) == name
    expected = None if quantiles is None else tuple(map(float, quantiles))
    assert parse_profile_name(name) == (sparse, expected)


def test_equal_quantiles_give_one_name():
    tenth = [float(q) for q in ("0", "0.0", "0", "0", "0", "0.10")]
    assert profile_name(False, tenth) == "phylum_0_0_0_0_0_0.1"


@pytest.mark.parametrize(
    ("sparse", "quantiles"),
    [
        (False, None),
        (False, (0, 0, 0, 0, 0, 0)),
        (False, (0, 0, 0, 0, 0, 1.5)),
        (False, (0, 0, 0, 0, 0.1)),
        (True, (0, 0, 0, 0, 0, 0.1234567)),
    ],
)
def test_invalid_profiles_are_rejected(sparse, quantiles):
    with pytest.raises(ValueError):
        profile_name(sparse, quantiles)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "baseline",
        "phylum",
        "sparse_sparse",
        "phylum_sparse",
        "sparse_",
        "phylum_0_0_0_0_0_0",
        "phylum_0_0_0_0_0_0.10",
        "phylum_0_0_0_0_0_0.1_0",
        "phylum_0_0_0_0_0_0.1_sparse",
        ".sparse.lock",
    ],
)
def test_malformed_names_are_rejected(name):
    with pytest.raises(ValueError):
        parse_profile_name(name)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["dense"],
        ["phylum"],
        ["sparse", "sparse"],
        ["sparse", "--quantiles", *Q],
        ["sparse", "--stats-csv", "stats.csv"],
        ["phylum", "--quantiles", "0", "0", "0", "0", "0", "0"],
    ],
)
def test_cli_rejects_invalid_selections(argv):
    with pytest.raises(SystemExit):
        profiles._parse_args(argv)


# ---------- profile builder ----------


@pytest.fixture
def stubs(monkeypatch) -> dict[str, list]:
    """Component builders that only create the directory, and a check that records its call."""
    calls: dict[str, list] = {"built": [], "verified": []}

    def build(out_dir: Path, *args) -> None:
        calls["built"].append(out_dir.name)
        out_dir.mkdir(parents=True)

    def verify(data_config: Path, names) -> None:
        calls["verified"].append((data_config.parent.name, list(names)))

    monkeypatch.setattr(profiles, "build_sparse_component", build)
    monkeypatch.setattr(profiles, "build_phylum_component", build)
    monkeypatch.setattr(profiles, "verify_profile", verify)
    return calls


def _build(dq_root: Path, *argv: str) -> Path:
    return profiles.build(profiles._parse_args([*argv, "--dq-root", str(dq_root)]))


def _profile_dirs(dq_root: Path) -> list[str]:
    return sorted(p.name for p in (dq_root / PROFILES_DIR).iterdir() if p.is_dir())


def test_profile_config_is_the_base_plus_the_degraded_training_splits(tmp_path, stubs):
    from mermaidseg.io import setup_config

    out = _build(tmp_path, "sparse", "phylum", "--quantiles", *Q)
    name = "sparse_phylum_0_0_0_0_0_0.1"
    assert out == tmp_path.resolve() / PROFILES_DIR / name

    components = tmp_path.resolve() / "components"
    added = {
        ("data", "coralscapes_v2", "train", "hf_repo_id"): str(
            components / "coralscapes_v2_sparse25_pad3_seed0"
        ),
        ("data", "ucsd_mosaics", "train", "hf_repo_id"): str(
            components / "ucsd_mosaics_sparse25_pad3_seed0"
        ),
        ("data", "coralnet", "train", "manifest_path"): str(
            components / "coralnet_phylum_0_0_0_0_0_0.1" / MANIFEST_FILE
        ),
    }
    base = _flatten(yaml.safe_load(DEFAULT_BASE_CONFIG.read_text()))
    config = _flatten(yaml.safe_load((out / DATA_CONFIG_FILE).read_text()))
    dropped = ("data", "coralnet", "train", "annotation_overrides")
    assert dropped in base
    assert dropped not in config
    assert config == {key: value for key, value in (base | added).items() if key != dropped}
    assert not set(added) & set(base)

    cfg = setup_config({"data": str(out / DATA_CONFIG_FILE)})
    assert (
        cfg.data.coralnet.train.manifest_path
        == added[("data", "coralnet", "train", "manifest_path")]
    )
    assert "transform" in cfg.data.coralscapes_v2.train

    info = json.loads((out / PROFILE_FILE).read_text())
    assert info["name"] == name
    assert info["sparse"] is True
    assert list(info["quantiles"].values()) == list(QF)
    assert info["overrides"] == {".".join(path): value for path, value in added.items()}
    assert stubs["verified"] == [
        (f".{name}.partial", ["coralnet", "coralscapes_v2", "ucsd_mosaics"])
    ]


def test_profiles_share_components(tmp_path, stubs):
    _build(tmp_path, "sparse")
    _build(tmp_path, "phylum", "--quantiles", *Q)
    built = [
        "coralscapes_v2_sparse25_pad3_seed0",
        "ucsd_mosaics_sparse25_pad3_seed0",
        "coralnet_phylum_0_0_0_0_0_0.1",
    ]
    assert stubs["built"] == built

    _build(tmp_path, "phylum", "sparse", "--quantiles", *Q)
    assert stubs["built"] == built
    assert _profile_dirs(tmp_path) == [
        "phylum_0_0_0_0_0_0.1",
        "sparse",
        "sparse_phylum_0_0_0_0_0_0.1",
    ]


def test_existing_profile_needs_overwrite(tmp_path, stubs):
    out = _build(tmp_path, "sparse")
    (out / "stale").write_text("")
    with pytest.raises(SystemExit, match="--overwrite"):
        _build(tmp_path, "sparse")

    _build(tmp_path, "sparse", "--overwrite")
    assert not (out / "stale").exists()
    assert (out / DATA_CONFIG_FILE).is_file()
    assert len(stubs["built"]) == 2


def test_failed_check_leaves_no_profile(tmp_path, stubs, monkeypatch):
    def broken(data_config: Path, names) -> None:
        raise RuntimeError("cannot load the train split")

    monkeypatch.setattr(profiles, "verify_profile", broken)
    with pytest.raises(RuntimeError, match="cannot load"):
        _build(tmp_path, "phylum", "--quantiles", *Q)
    assert _profile_dirs(tmp_path) == []


def test_profile_config_needs_the_overridden_sections():
    base = {"data": {"coralnet": {"train": "None"}}}
    with pytest.raises(ValueError, match="data.coralnet.train"):
        profile_config(base, {("data", "coralnet", "train", "manifest_path"): "x"})


def test_phylum_profile_drops_annotation_overrides_and_keeps_the_blacklist():
    base = {
        "data": {
            "coralnet": {
                "train": {
                    "whitelist_sources": [1],
                    "image_blacklist": "bl.parquet",
                    "annotation_overrides": "ov.parquet",
                }
            },
            "coralscapes_v2": {"train": {}},
        }
    }
    phylum = profile_config(
        base, {("data", "coralnet", "train", "manifest_path"): "phylum.parquet"}
    )
    train = phylum["data"]["coralnet"]["train"]
    assert train["manifest_path"] == "phylum.parquet"
    assert train["image_blacklist"] == "bl.parquet"
    assert train["whitelist_sources"] == [1]
    assert "annotation_overrides" not in train

    sparse = profile_config(base, {("data", "coralscapes_v2", "train", "hf_repo_id"): "sparse"})
    assert sparse["data"]["coralnet"]["train"]["annotation_overrides"] == "ov.parquet"


def test_components_are_built_with_the_builders_own_parsers(tmp_path, monkeypatch):
    sparse_builder = importlib.import_module("build_sparse_dense_datasets")
    phylum_builder = importlib.import_module("build_coralnet_phylum_manifest")
    seen = []
    monkeypatch.setattr(sparse_builder, "build", seen.append)
    monkeypatch.setattr(phylum_builder, "build", seen.append)

    profiles.build_sparse_component(tmp_path / "sparse", "ucsd_mosaics", 8)
    profiles.build_phylum_component(
        tmp_path / "phylum", QF, DEFAULT_BASE_CONFIG, tmp_path / "s.csv"
    )
    sparse_args, phylum_args = seen
    assert sparse_args.dataset == "ucsd_mosaics"
    assert sparse_args.output_dir == tmp_path / "sparse"
    assert (sparse_args.n_points, sparse_args.padding, sparse_args.seed) == (25, 3, 0)
    assert sparse_args.num_proc == 8
    assert tuple(phylum_args.quantiles) == QF
    assert phylum_args.output_dir == tmp_path / "phylum"
    assert phylum_args.data_config == DEFAULT_BASE_CONFIG
    assert phylum_args.stats_csv == tmp_path / "s.csv"


# ---------- slurm ----------


def _launch(dq_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["bash", str(_SLURM / "data_quality_launch_train.sh"), "--dry-run", *args],
        env={**os.environ, "DQ_ROOT": str(dq_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_launcher_submits_every_built_profile(tmp_path):
    names = ["phylum_0_0_0_0_0_0.1", "sparse"]
    for name in [*names, ".sparse_phylum_0_0_0_0_0_0.1.partial"]:
        (tmp_path / PROFILES_DIR / name).mkdir(parents=True)
        (tmp_path / PROFILES_DIR / name / DATA_CONFIG_FILE).write_text("")
    (tmp_path / PROFILES_DIR / ".sparse.lock").write_text("")

    out = _launch(tmp_path)
    for name in names:
        submit = f"sbatch --job-name=mermaid-dq-{name} slurm/data_quality_train.sbatch {name}"
        assert f"{submit} (dry-run)" in out
    assert out.count("would submit") == len(names)


def test_launcher_prints_the_build_command_of_an_unbuilt_profile(tmp_path):
    out = _launch(tmp_path, "sparse_phylum_0_0_0_0_0_0.1")
    build = "sbatch slurm/data_quality_build.sbatch sparse phylum --quantiles 0 0 0 0 0 0.1"
    assert f"not built: {build}" in out
    assert "would submit" not in out


def test_train_sbatch_matches_the_vitl_recipe():
    sbatch = (_SLURM / "data_quality_train.sbatch").read_text()
    assert "--array" not in sbatch
    assert 'RUN="cbm_dq_${PROFILE}"' in sbatch
    assert run_name("sparse") == "cbm_dq_sparse"
    assert f'DATA_CONFIG="$DQ_ROOT/{PROFILES_DIR}/${{PROFILE}}/{DATA_CONFIG_FILE}"' in sbatch
    assert '--data-config "$DATA_CONFIG"' in sbatch
    assert 'MODEL_CONFIG="../configs/model_config_cbm_dpt_lora_vitl_256.yaml"' in sbatch
    assert 'TRAINING_CONFIG="../configs/training_config_cbm_vitsize.yaml"' in sbatch
    assert "--per-pixel-loss-weight 0.5" in sbatch
    assert "--per-image-loss-weight 0.5" in sbatch
    assert "--data-seed" not in sbatch
    assert "--batch-size" not in sbatch
    assert "MERMAIDSEG_S3_OFFLINE=1" in sbatch

    vitsize = (_SLURM / "vitsize_train.sbatch").read_text()
    assert "training_config_cbm_vitsize.yaml" in vitsize
    assert BASELINE_RUN == "cbm_vitsize_vitl"
    assert "vitl" in _bash_array(vitsize, "SIZES")
    for directive in ("--partition=", "--cpus-per-task=", "--mem=", "--time="):
        line = re.compile(rf"^#SBATCH {directive}.*$", re.MULTILINE)
        assert line.search(sbatch).group(0) == line.search(vitsize).group(0)

    training = yaml.safe_load((_REPO / "configs" / "training_config_cbm_vitsize.yaml").read_text())
    training = training["training"]
    assert training["iterations_per_train_epoch"] == STEPS[-1] == 150_000
    assert training["epochs"] == 1
    assert training["scheduler"]["type"] == "LinearLR"
    assert training["scheduler"]["warmup_iters"] == 4000
    assert training["scheduler"]["total_iters"] == 150_000 - 4000
    assert training["scheduler"]["end_factor"] == 0.0
    assert training["scheduler"]["step_every"] == "iteration"
    assert training["batch_size"] == 32
    assert training["grad_accum_steps"] == 1
    assert training["checkpoint_every_steps"] == STEPS[0]
    assert training["padding"] == profiles.PADDING

    model = (_REPO / "configs" / "model_config_cbm_dpt_lora_vitl_256.yaml").read_text()
    assert "dinov3-vitl16" in model
    assert "input_size: [256, 256]" in model


def test_slurm_scripts_and_python_agree_on_the_profiles():
    train = (_SLURM / "data_quality_train.sbatch").read_text()
    launch = (_SLURM / "data_quality_launch_train.sh").read_text()
    build = (_SLURM / "data_quality_build.sbatch").read_text()

    scratch = Path("/data/vision/beery/scratch/sauder")
    assert scratch / "mermaid-segmentation" / "data_quality" == DEFAULT_DQ_ROOT
    for sbatch in (train, build):
        assert "export SCRATCH=/data/vision/beery/scratch/sauder" in sbatch
        assert "DQ_ROOT=$SCRATCH/mermaid-segmentation/data_quality" in sbatch
    assert f'DQ_ROOT="${{DQ_ROOT:-{DEFAULT_DQ_ROOT}}}"' in launch
    assert f'PROFILE_ROOT="$DQ_ROOT/{PROFILES_DIR}"' in launch
    assert f'"$PROFILE_ROOT"/*/{DATA_CONFIG_FILE}' in launch
    assert build_parser().parse_args([]).dq_root == DEFAULT_DQ_ROOT

    assert "scripts/build_data_quality_profile.py" in build
    assert '--dq-root "$DQ_ROOT"' in build
    assert '"$@"' in build
    assert NAME_TO_BUILD_ARGS in train
    assert NAME_TO_BUILD_ARGS in launch


def test_eval_scripts_use_the_unmodified_eval_data():
    evaluation = (_SLURM / "data_quality_eval.sbatch").read_text()
    launch = (_SLURM / "data_quality_launch_evals.sh").read_text()
    assert "--data-config configs/data_config_dedup.yaml" in evaluation
    assert "cbm_dq_*" in evaluation
    assert 'CKPT_GLOB="nbs/model_checkpoints/cbm_dq_*/model_step*"' in launch
    assert 'OUT_BASE="eval_out/data_quality"' in launch
    assert f"RESOLUTIONS=({' '.join(map(str, RESOLUTIONS))})" in launch
    assert 'EVAL_SBATCH="slurm/data_quality_eval.sbatch"' in launch

    vitsize_launch = (_SLURM / "vitsize_launch_evals.sh").read_text()
    assert 'OUT_BASE="eval_out/vitsize"' in vitsize_launch
    args = build_parser().parse_args([])
    assert args.root.name == "data_quality"
    assert args.baseline_root.name == "vitsize"


# ---------- compile ----------


def _write_done(out_dir: Path, accuracy: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"coralnet": {"pooled": {"class_accuracy": accuracy, "class_miou": 0.25}}}
    (out_dir / "summary.json").write_text(json.dumps(summary))
    (out_dir / ".done").write_text("")


def _write_profile(dq_root: Path, name: str) -> None:
    out = dq_root / PROFILES_DIR / name
    out.mkdir(parents=True)
    (out / DATA_CONFIG_FILE).write_text("")


def test_empty_roots_are_the_baseline_grid(tmp_path: Path):
    rows = build_rows(tmp_path / "data_quality", tmp_path / "vitsize", tmp_path / "dq")
    assert len(rows) == len(STEPS) * len(RESOLUTIONS)
    assert {row["run"] for row in rows} == {BASELINE_RUN}
    assert {row["profile"] for row in rows} == {BASELINE}
    assert all(row["done"] is False for row in rows)
    assert all(math.isnan(row["coralnet_class_accuracy"]) for row in rows)


def test_profiles_come_from_evals_and_built_profiles(tmp_path: Path):
    root = tmp_path / "data_quality"
    baseline_root = tmp_path / "vitsize"
    dq_root = tmp_path / "dq"
    _write_done(baseline_root / BASELINE_RUN / "step150000_res256", 0.7)
    _write_done(baseline_root / "cbm_vitsize_vits" / "step150000_res256", 0.1)
    _write_done(root / run_name("phylum_0_0_0_0_0_0.1") / "step150000_res256", 0.6)
    _write_done(root / run_name("sparse") / "step10000_res512", 0.2)
    _write_done(root / "cbm_other" / "step150000_res256", 0.3)
    _write_done(root / "cbm_dq_phylum" / "step150000_res256", 0.3)
    _write_profile(dq_root, "sparse_phylum_0_0_0_0_0_0.1")
    _write_profile(dq_root, ".sparse.partial")

    rows = build_rows(root, baseline_root, dq_root)
    assert list(dict.fromkeys(row["profile"] for row in rows)) == [
        BASELINE,
        "phylum_0_0_0_0_0_0.1",
        "sparse",
        "sparse_phylum_0_0_0_0_0_0.1",
    ]
    assert len(rows) == 4 * len(STEPS) * len(RESOLUTIONS) + 1
    by_key = {(row["run"], row["steps"], row["resolution"]): row for row in rows}
    assert by_key[(BASELINE_RUN, 150_000, 256)]["coralnet_class_accuracy"] == 0.7
    assert (
        by_key[(run_name("phylum_0_0_0_0_0_0.1"), 150_000, 256)]["coralnet_class_accuracy"] == 0.6
    )
    assert by_key[(run_name("sparse"), 10_000, 512)]["done"] is True
    combined = by_key[(run_name("sparse_phylum_0_0_0_0_0_0.1"), 150_000, 256)]
    assert math.isnan(combined["coralnet_class_accuracy"])


def test_rows_describe_their_profile(tmp_path: Path):
    dq_root = tmp_path / "dq"
    component = phylum_component_dir(dq_root, QF)
    component.mkdir(parents=True)
    run_info = {
        "flagged_sources": [1, 2, 3],
        "n_changed_points": 42,
        "pct_flagged_points_changed": 12.5,
    }
    (component / "run.json").write_text(json.dumps(run_info))
    for name in (
        "sparse",
        "phylum_0_0_0_0_0_0.1",
        "sparse_phylum_0_0_0_0_0_0.1",
        "phylum_0.2_0_0_0_0_0",
    ):
        _write_profile(dq_root, name)

    rows = build_rows(tmp_path / "data_quality", tmp_path / "vitsize", dq_root)
    final = {
        row["profile"]: row
        for row in rows
        if row["steps"] == STEPS[-1] and row["resolution"] == 256
    }

    def described(profile: str) -> tuple:
        row = final[profile]
        return (
            row["sparse"],
            [row[c] for c in PHYLUM_Q_COLUMNS],
            [row[c] for c in PHYLUM_STAT_COLUMNS],
        )

    zeros = [0.0] * len(PHYLUM_Q_COLUMNS)
    assert described(BASELINE) == (False, zeros, [0, 0, 0.0])
    assert described("sparse") == (True, zeros, [0, 0, 0.0])
    assert described("phylum_0_0_0_0_0_0.1") == (False, list(QF), [3, 42, 12.5])
    assert described("sparse_phylum_0_0_0_0_0_0.1") == (True, list(QF), [3, 42, 12.5])
    unread = final["phylum_0.2_0_0_0_0_0"]
    assert unread["phylum_q_n_images"] == 0.2
    assert all(math.isnan(unread[c]) for c in PHYLUM_STAT_COLUMNS)


def test_csv_has_every_column(tmp_path: Path):
    dq_root = tmp_path / "dq"
    _write_profile(dq_root, "sparse")
    output = tmp_path / "compiled.csv"
    compile_main(
        [
            f"--root={tmp_path / 'data_quality'}",
            f"--baseline-root={tmp_path / 'vitsize'}",
            f"--dq-root={dq_root}",
            f"--output={output}",
        ]
    )
    with output.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ALL_COLUMNS
        rows = list(reader)
    assert {row["sparse"] for row in rows if row["profile"] == "sparse"} == {"true"}
    assert {row["sparse"] for row in rows if row["profile"] == BASELINE} == {"false"}
