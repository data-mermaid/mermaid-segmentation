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
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from scripts.build_trainval_data_config import HEADER, build_final_data_config

_REPO = Path(__file__).resolve().parents[2]
_SLURM = _REPO / "slurm"
_CONFIGS = _REPO / "configs"

# Placeholder site values; the launcher takes them from the environment.
_SITE_ENV = {
    "SLURM_ACCOUNT": "acct-x",
    "SLURM_QOS": "qos-x",
    "SLURM_PARTITION": "part-x",
}


_SITE_VARS = (*_SITE_ENV, "MERMAIDSEG_LOG_DIR")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(_SLURM, root / "slurm")
    (root / "outputs" / "model_checkpoints").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$QUEUE_FILE" 2>/dev/null || true\n')
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 'sbatch called in a test' >&2\nexit 1\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return root


def _run_launcher(
    repo: Path,
    name: str,
    *args: str,
    queued: Sequence[str] = (),
    env_overrides: dict[str, str] | None = None,
    unset: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _SITE_VARS}
    env.update(_SITE_ENV)
    log_dir = repo.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    env["MERMAIDSEG_LOG_DIR"] = str(log_dir)
    env.update(env_overrides or {})
    for var in unset:
        env.pop(var, None)
    env["MERMAIDSEG_ENV_FILE"] = str(repo.parent / "absent.env")
    env["PATH"] = f"{repo.parent / 'bin'}:{env['PATH']}"
    queue_file = repo.parent / "queue.txt"
    queue_file.write_text("".join(f"{job}\n" for job in queued))
    env["QUEUE_FILE"] = str(queue_file)
    return subprocess.run(
        ["bash", str(repo / "slurm" / f"launch_{name}.sh"), "--dry-run", *args],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _launch(repo: Path, name: str, *args: str, **kwargs) -> tuple[str, list]:
    result = _run_launcher(repo, name, *args, **kwargs)
    assert result.returncode == 0, result.stderr
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


def _runs(jobs: list[list[str]]) -> dict[str | None, list[str]]:
    """Jobs by name; every launcher submits one job per run."""
    runs = {_opt(job, "job-name"): job for job in jobs}
    assert len(runs) == len(jobs), "a run was submitted more than once"
    return runs


def _training_config(name: str) -> dict:
    return yaml.safe_load((_CONFIGS / f"training_config_cbm_{name}.yaml").read_text())["training"]


# ---------- scripts ----------


_SITE_PATH = re.compile(r"/(?:data|archive|home)/")
_SITE_SBATCH = re.compile(r"^\s*#SBATCH\s+--(?:account|qos|partition)\b", re.MULTILINE)


def test_no_site_specific_values_are_tracked():
    files = [path for path in _SLURM.rglob("*") if path.is_file()]
    assert files
    for path in files:
        text = path.read_text()
        assert not _SITE_PATH.search(text), f"absolute site path in {path.name}"
        assert not _SITE_SBATCH.search(text), f"site SBATCH option in {path.name}"


# ---------- train.sbatch ----------


@pytest.fixture
def job(tmp_path: Path) -> Path:
    """A fixture checkout whose train.sbatch runs against fake srun, uv and nvidia-smi."""
    root = tmp_path / "checkout"
    shutil.copytree(_SLURM, root / "slurm")
    (root / "outputs" / "model_checkpoints" / "RUN").mkdir(parents=True)
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/bash\n")
    bin_dir = tmp_path / "jobbin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text('#!/bin/bash\nprintf \'%s\\n\' "$@" > "$SRUN_ARGV_FILE"\n')
    (bin_dir / "uv").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    (tmp_path / "cache").mkdir()
    (tmp_path / "job.env").write_text(
        f"MERMAIDSEG_SCRATCH={tmp_path / 'scratch'}\n"
        f"MERMAIDSEG_CACHE_DIR={tmp_path / 'cache'}\n"
        "WANDB_API_KEY=test-key\n"
    )
    return root


def _run_train(job: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    tmp_path = job.parent
    argv_file = tmp_path / "srun_argv.txt"
    env = {k: v for k, v in os.environ.items() if k not in _SITE_VARS}
    env.update(
        PATH=f"{tmp_path / 'jobbin'}:{env['PATH']}",
        SLURM_SUBMIT_DIR=str(job),
        MERMAIDSEG_ENV_FILE=str(tmp_path / "job.env"),
        CUDA_VISIBLE_DEVICES="0",
        SLURM_CPUS_PER_TASK="32",
        SLURM_JOB_ID=f"test-{tmp_path.name}",
        SRUN_ARGV_FILE=str(argv_file),
    )
    result = subprocess.run(
        ["bash", str(job / "slurm" / "train.sbatch"), *args],
        cwd=job,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    argv = argv_file.read_text().splitlines() if argv_file.exists() else []
    return result, argv


def test_train_from_scratch_runs_the_pipeline_and_marks_done(job):
    result, argv = _run_train(job, "RUN", "--", "--batch-size", "8")
    assert result.returncode == 0, result.stderr
    assert argv[0] == str(job / ".venv" / "bin" / "python")
    assert argv[1] == "scripts/train.py"
    assert _arg(argv, "--run-name") == "RUN"
    assert _arg(argv, "--num-workers") == "28"
    assert _arg(argv, "--nccl-timeout-min") == "30"
    assert _arg(argv, "--preempt-checkpoint-every-steps") == "2000"
    assert _arg(argv, "--output-dir") == str(job / "outputs")
    assert _arg(argv, "--cache-dir") == str(job.parent / "cache")
    assert _arg(argv, "--batch-size") == "8"
    assert "--checkpoint" not in argv
    assert (job / "outputs" / "model_checkpoints" / "RUN" / ".done").exists()


def test_train_resumes_from_the_newest_checkpoint(job):
    ckpt_dir = job / "outputs" / "model_checkpoints" / "RUN"
    older, newer = ckpt_dir / "model_step2000", ckpt_dir / "model_preempt4000"
    older.touch()
    newer.touch()
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert _arg(argv, "--checkpoint") == str(newer)


def test_train_exits_2_when_the_env_file_is_not_readable(job):
    env_file = job.parent / "job.env"
    env_file.unlink()
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 2
    assert f"env file {env_file} not readable; see .env.example" in result.stderr
    assert argv == []


def test_train_exits_2_when_submitted_from_outside_the_repo_root(job, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in _SITE_VARS}
    env.update(SLURM_SUBMIT_DIR=str(elsewhere), MERMAIDSEG_ENV_FILE=str(tmp_path / "job.env"))
    result = subprocess.run(
        ["bash", str(job / "slurm" / "train.sbatch"), "RUN"],
        cwd=elsewhere,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert (
        f"submit from the repo root (slurm/common.sh not found under {elsewhere})" in result.stderr
    )


# ---------- configs ----------


@pytest.mark.parametrize("name", ["final", "final_hires"])
def test_decaying_configs_reach_zero_at_the_last_update(name):
    cfg = _training_config(name)
    scheduler = cfg["scheduler"]
    updates = cfg["epochs"] * cfg["iterations_per_train_epoch"]
    assert scheduler["type"] == "LinearLR" and scheduler["end_factor"] == 0.0
    assert scheduler["warmup_iters"] + scheduler["total_iters"] == updates


def test_trainval_config_is_rebuilt_from_the_base_config():
    rebuilt = build_final_data_config((_CONFIGS / "data_config.yaml").read_text(), header=HEADER)
    assert rebuilt == (_CONFIGS / "data_config_trainval.yaml").read_text()


# ---------- training launchers ----------


def test_final_runs_the_768_fine_tune_after_the_final_run(repo):
    _, jobs = _launch(repo, "final")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_final", "cbm_final_res768"]
    final, hires = runs["cbm_final"], runs["cbm_final_res768"]
    for job in (final, hires):
        assert _arg(job, "--data-config") == "configs/data_config_trainval.yaml"
    assert _arg(final, "--training-config") == "configs/training_config_cbm_final.yaml"
    assert _opt(final, "dependency") is None
    assert _opt(hires, "dependency") == "afterok:100001"
    assert _opt(hires, "gres") == "gpu:4"
    assert _arg(hires, "--init-weights") == "outputs/model_checkpoints/cbm_final/model_step300000"
    assert _arg(hires, "--resolution") == "768"
    assert _arg(hires, "--training-config") == "configs/training_config_cbm_final_hires.yaml"
    assert int(_arg(hires, "--batch-size")) * int(_arg(hires, "--grad-accum-steps")) == 32
    assert _arg(hires, "--batch-size") == "8"
    assert _arg(hires, "--grad-accum-steps") == "4"
    assert "--per-pixel-loss-weight" not in hires + final


def test_dry_run_emits_two_sbatch_lines_with_site_options(repo):
    _, jobs = _launch(repo, "final")
    assert len(jobs) == 2
    for job in jobs:
        assert _opt(job, "account") == "acct-x"
        assert _opt(job, "qos") == "qos-x"
        assert _opt(job, "partition") == "part-x"
        assert _opt(job, "output") == f"{repo.parent}/logs/slurm-%x-%j.out"
        assert _opt(job, "error") == f"{repo.parent}/logs/slurm-%x-%j.err"
        assert _opt(job, "export") is None
        assert "slurm/train.sbatch" in job


@pytest.mark.parametrize(
    "variable", ["SLURM_ACCOUNT", "SLURM_QOS", "SLURM_PARTITION", "MERMAIDSEG_LOG_DIR"]
)
def test_missing_site_variable_fails_naming_it(repo, variable):
    result = _run_launcher(repo, "final", unset=[variable])
    assert result.returncode != 0
    assert variable in result.stderr
    assert "would submit" not in result.stdout


def test_nonexistent_log_dir_fails_naming_it(repo):
    missing = repo.parent / "no-such-logs"
    result = _run_launcher(repo, "final", env_overrides={"MERMAIDSEG_LOG_DIR": str(missing)})
    assert result.returncode == 2
    assert "MERMAIDSEG_LOG_DIR" in result.stderr
    assert "would submit" not in result.stdout


def test_finished_run_is_reported_and_not_submitted(repo):
    done = repo / "outputs" / "model_checkpoints" / "cbm_final"
    done.mkdir()
    (done / ".done").touch()
    out, jobs = _launch(repo, "final")
    assert "done    : cbm_final" in out
    assert "cbm_final" not in _runs(jobs)


def test_selecting_one_run_submits_only_that_run(repo):
    _, jobs = _launch(repo, "final", "cbm_final")
    assert list(_runs(jobs)) == ["cbm_final"]


def test_768_batch_and_accum_must_multiply_to_32(repo):
    result = _run_launcher(
        repo, "final", env_overrides={"FINAL_768_BATCH": "16", "FINAL_768_ACCUM": "4"}
    )
    assert result.returncode == 2
    assert "not 32" in result.stderr
