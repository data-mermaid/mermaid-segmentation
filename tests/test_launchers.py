"""slurm/ job bodies, launchers and the experiment configs they point at.

The launchers run in --dry-run mode from a temporary copy of slurm/, so checkpoints
and profiles are faked per test and nothing is submitted.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.build_trainval_data_config import HEADER, build_final_data_config

_REPO = Path(__file__).resolve().parents[1]
_SLURM = _REPO / "slurm"
_CONFIGS = _REPO / "configs"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(_SLURM, root / "slurm")
    (root / "nbs" / "model_checkpoints").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$QUEUE_FILE" 2>/dev/null || true\n')
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 'sbatch called in a test' >&2\nexit 1\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return root


def _launch(repo: Path, name: str, *args: str, queued: list[str] = ()) -> tuple[str, list]:
    env = {k: v for k, v in os.environ.items() if k not in ("DS_ROOT", "SPARSE_ROOT", "EVAL_ROOT")}
    env["PATH"] = f"{repo.parent / 'bin'}:{env['PATH']}"
    queue_file = repo.parent / "queue.txt"
    queue_file.write_text("".join(f"{job}\n" for job in queued))
    env["QUEUE_FILE"] = str(queue_file)
    result = subprocess.run(
        ["bash", str(repo / "slurm" / f"launch_{name}.sh"), "--dry-run", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "  would submit: sbatch "
    jobs = [
        shlex.split(line[len(prefix) :])
        for line in result.stdout.splitlines()
        if line.startswith(prefix)
    ]
    return result.stdout, jobs


def _opt(job: list[str], name: str) -> str | None:
    for token in job:
        if token.startswith(f"--{name}="):
            return token.split("=", 1)[1]
    return None


def _arg(job: list[str], flag: str) -> str | None:
    return job[job.index(flag) + 1] if flag in job else None


def _runs(jobs: list[list[str]]) -> dict[str, list[str]]:
    """Jobs by name; every launcher submits one job per run."""
    runs = {_opt(job, "job-name"): job for job in jobs}
    assert len(runs) == len(jobs), "a run was submitted more than once"
    return runs


def _sbatch_header(name: str) -> dict[str, str]:
    header = {}
    for line in (_SLURM / name).read_text().splitlines():
        match = re.match(r"#SBATCH --([\w-]+)(?:=(.*))?$", line)
        if match:
            header[match.group(1)] = match.group(2) or ""
    return header


def _training_config(name: str) -> dict:
    return yaml.safe_load((_CONFIGS / f"training_config_cbm_{name}.yaml").read_text())["training"]


# ---------- scripts ----------


def test_every_script_parses():
    scripts = [*_SLURM.glob("*.sh"), *_SLURM.glob("*.sbatch")]
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_job_resources():
    train = _sbatch_header("train.sbatch")
    assert train["time"] == "71:59:00"
    assert train["partition"] == "vision-beery-a100"
    assert train["qos"] == "vision-beery-main"
    assert train["gres"] == "gpu:1"
    assert "/scratch/sauder/mermaid-segmentation/logs/" in train["output"]


def test_no_retired_paths_or_exports():
    for path in _SLURM.iterdir():
        text = path.read_text()
        assert "mermaid-segmentation-final" not in text, path.name
        assert "engaging" not in text.lower(), path.name
        code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        assert not any("--export" in line for line in code), path.name
    assert 'REPO="$SCRATCH/mermaid-segmentation"' in (_SLURM / "common.sh").read_text()


# ---------- configs ----------


@pytest.mark.parametrize(
    ("name", "updates", "warmup"),
    [
        ("final", 300_000, 4000),
        ("final_hires", 50_000, 4000),
    ],
)
def test_decaying_configs_reach_zero_at_the_last_update(name, updates, warmup):
    cfg = _training_config(name)
    scheduler = cfg["scheduler"]
    assert cfg["epochs"] * cfg["iterations_per_train_epoch"] == updates
    assert scheduler["type"] == "LinearLR" and scheduler["end_factor"] == 0.0
    assert scheduler["warmup_iters"] == warmup
    assert scheduler["warmup_iters"] + scheduler["total_iters"] == updates
    assert cfg["batch_size"] * cfg.get("grad_accum_steps", 1) == 32
    assert cfg["loss"]["per_pixel_loss_weight"] == cfg["loss"]["per_image_loss_weight"] == 0.5


def test_only_the_final_configs_regularize_the_taxonomy():
    for name in ("final", "final_hires"):
        assert _training_config(name)["taxonomic_loss"]["kl_weight"] == 0.5
    for path in _CONFIGS.glob("training_config_*.yaml"):
        assert "ramp" not in path.read_text(), path.name


def test_trainval_config_is_rebuilt_from_the_base_config():
    rebuilt = build_final_data_config((_CONFIGS / "data_config.yaml").read_text(), header=HEADER)
    assert rebuilt == (_CONFIGS / "data_config_trainval.yaml").read_text()


def test_launchers_use_existing_configs():
    referenced = set()
    for path in _SLURM.glob("launch_*.sh"):
        referenced |= set(re.findall(r"configs/([\w.]+\.yaml)", path.read_text()))
    referenced = {name for name in referenced if "$" not in name}
    assert referenced
    for name in referenced:
        assert (_CONFIGS / name).is_file(), name


# ---------- training launchers ----------


def test_final_runs_the_768_fine_tune_after_the_final_run(repo):
    _, jobs = _launch(repo, "final")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_final", "cbm_final_res768"]
    final, hires = runs["cbm_final"], runs["cbm_final_res768"]
    for job in (final, hires):
        assert _arg(job, "--data-config") == "../configs/data_config_trainval.yaml"
    assert _arg(final, "--training-config") == "../configs/training_config_cbm_final.yaml"
    assert _opt(final, "dependency") is None
    assert _opt(hires, "dependency") == "afterok:100001"
    assert _opt(hires, "gres") == "gpu:4"
    assert _arg(hires, "--init-weights") == "nbs/model_checkpoints/cbm_final/model_step300000"
    assert _arg(hires, "--resolution") == "768"
    assert _arg(hires, "--training-config") == "../configs/training_config_cbm_final_hires.yaml"
    assert int(_arg(hires, "--batch-size")) * int(_arg(hires, "--grad-accum-steps")) == 32
