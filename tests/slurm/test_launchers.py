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
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

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
    dry_run: bool = True,
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
        [
            "bash",
            str(repo / "slurm" / f"launch_{name}.sh"),
            *(["--dry-run"] if dry_run else []),
            *args,
        ],
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
    (bin_dir / "srun").write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$SRUN_ARGV_FILE"\nenv > "$SRUN_ARGV_FILE.env"\n'
    )
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


# Variables train.sbatch sets for the trainer; a caller's own values must not leak in.
_TRAINER_ENV_VARS = (
    "OMP_NUM_THREADS",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    "WANDB_RUN_ID",
    "WANDB_RUN_GROUP",
    "MERMAIDSEG_WANDB_ID_PREFIX",
)


def _run_train(
    job: Path, *args: str, env_overrides: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    tmp_path = job.parent
    argv_file = tmp_path / "srun_argv.txt"
    env = {
        k: v for k, v in os.environ.items() if k not in _SITE_VARS and k not in _TRAINER_ENV_VARS
    }
    env.update(
        PATH=f"{tmp_path / 'jobbin'}:{env['PATH']}",
        SLURM_SUBMIT_DIR=str(job),
        MERMAIDSEG_ENV_FILE=str(tmp_path / "job.env"),
        CUDA_VISIBLE_DEVICES="0",
        SLURM_CPUS_PER_TASK="32",
        SLURM_JOB_ID=f"test-{tmp_path.name}",
        SRUN_ARGV_FILE=str(argv_file),
    )
    env.update(env_overrides or {})
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


def _srun_env(job: Path) -> dict[str, str]:
    """The environment the fake srun was started with."""
    lines = (job.parent / "srun_argv.txt.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


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


def _checkpoint_dir(job: Path) -> Path:
    return job / "outputs" / "model_checkpoints" / "RUN"


def test_train_resume_skips_a_partial_checkpoint_even_when_it_is_the_newest(job):
    complete, partial = (
        _checkpoint_dir(job) / "model_step2000",
        _checkpoint_dir(job) / ("model_step4000.partial"),
    )
    complete.touch()
    partial.touch()
    os.utime(complete, (1_000_000, 1_000_000))
    os.utime(partial, (2_000_000, 2_000_000))
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert _arg(argv, "--checkpoint") == str(complete)


def test_train_with_only_a_partial_checkpoint_starts_from_scratch(job):
    (_checkpoint_dir(job) / "model_step4000.partial").touch()
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert "starting RUN from scratch" in result.stdout
    assert "--checkpoint" not in argv


def test_train_exits_1_when_the_init_weights_source_does_not_exist(job):
    result, argv = _run_train(job, "RUN", "--init-weights", "outputs/missing/model_step1")
    assert result.returncode == 1
    assert f"initial checkpoint {job}/outputs/missing/model_step1 does not exist" in result.stderr
    assert argv == []
    assert not (_checkpoint_dir(job) / ".done").exists()


def test_train_resolves_a_relative_init_weights_source_against_the_repo(job):
    source = job / "outputs" / "model_checkpoints" / "SRC" / "model_step1"
    source.parent.mkdir()
    source.touch()
    result, argv = _run_train(
        job, "RUN", "--init-weights", "outputs/model_checkpoints/SRC/model_step1"
    )
    assert result.returncode == 0, result.stderr
    assert _arg(argv, "--init-weights") == str(source)
    assert "--checkpoint" not in argv


def test_train_ignores_init_weights_once_the_run_has_a_checkpoint(job):
    own = _checkpoint_dir(job) / "model_step2000"
    own.touch()
    source = job / "init.pt"
    source.touch()
    result, argv = _run_train(job, "RUN", "--init-weights", str(source))
    assert result.returncode == 0, result.stderr
    assert _arg(argv, "--checkpoint") == str(own)
    assert "--init-weights" not in argv


def test_train_exits_0_without_training_when_the_run_is_already_done(job):
    (_checkpoint_dir(job) / ".done").touch()
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert "RUN already finished" in result.stdout
    assert argv == []


def test_train_groups_the_wandb_run_under_its_own_name_by_default(job):
    result, _ = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    env = _srun_env(job)
    assert env["WANDB_RUN_GROUP"] == "RUN"
    assert env["WANDB_RUN_ID"] == "r1-RUN"


def test_train_group_flag_sets_the_wandb_group_but_not_the_run_id(job):
    result, _ = _run_train(job, "RUN", "--group", "shared-group")
    assert result.returncode == 0, result.stderr
    env = _srun_env(job)
    assert env["WANDB_RUN_GROUP"] == "shared-group"
    assert env["WANDB_RUN_ID"] == "r1-RUN"


def test_train_wandb_id_prefix_from_the_env_file_starts_a_fresh_wandb_run(job):
    env_file = job.parent / "job.env"
    env_file.write_text(env_file.read_text() + "MERMAIDSEG_WANDB_ID_PREFIX=r7\n")
    result, _ = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert _srun_env(job)["WANDB_RUN_ID"] == "r7-RUN"


def test_train_rejects_an_unknown_option_with_usage(job):
    result, argv = _run_train(job, "RUN", "--bogus")
    assert result.returncode == 2
    assert "usage: sbatch slurm/train.sbatch" in result.stderr
    assert argv == []


def test_train_single_gpu_runs_without_torchrun_or_nccl_settings(job):
    result, argv = _run_train(job, "RUN")
    assert result.returncode == 0, result.stderr
    assert "torch.distributed.run" not in argv
    env = _srun_env(job)
    assert not [name for name in env if name.startswith("TORCH_NCCL") or name == "OMP_NUM_THREADS"]


def test_train_several_gpus_run_under_torchrun_with_the_workers_split_over_ranks(job):
    result, argv = _run_train(job, "RUN", env_overrides={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    assert result.returncode == 0, result.stderr
    python = str(job / ".venv" / "bin" / "python")
    assert argv[:6] == [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/train.py",
    ]
    assert _arg(argv, "--num-workers") == "7"


def test_train_several_gpus_export_the_nccl_and_omp_settings(job):
    result, _ = _run_train(job, "RUN", env_overrides={"CUDA_VISIBLE_DEVICES": "0,1"})
    assert result.returncode == 0, result.stderr
    env = _srun_env(job)
    assert {"OMP_NUM_THREADS", "TORCH_NCCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_DUMP_ON_TIMEOUT"} <= set(
        env
    )
    assert env["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == (
        f"{job}/logs/nccl_trace_RUN_test-{job.parent.name}_rank_"
    )


# ---------- configs ----------


@pytest.mark.parametrize("name", ["final", "final_hires"])
def test_decaying_configs_reach_zero_at_the_last_update(name):
    cfg = _training_config(name)
    scheduler = cfg["scheduler"]
    updates = cfg["epochs"] * cfg["iterations_per_train_epoch"]
    assert scheduler["type"] == "LinearLR" and scheduler["end_factor"] == 0.0
    assert scheduler["warmup_iters"] + scheduler["total_iters"] == updates


# ---------- training launchers ----------


def test_final_runs_the_768_fine_tune_after_the_final_run(repo):
    _, jobs = _launch(repo, "final")
    runs = _runs(jobs)
    assert list(runs) == ["cbm_final", "cbm_final_res768"]
    final, hires = runs["cbm_final"], runs["cbm_final_res768"]
    for job in (final, hires):
        assert _arg(job, "--data-config") == "configs/data_config.yaml"
    assert _arg(final, "--training-config") == "configs/training_config_cbm_final.yaml"
    assert _opt(final, "dependency") is None
    assert _opt(hires, "dependency") == "afterok:100001"
    assert _opt(hires, "gres") == "gpu:4"
    assert _arg(hires, "--init-weights") == "outputs/model_checkpoints/cbm_final/model_step300000"
    assert _arg(hires, "--resolution") == "768"
    assert _arg(hires, "--training-config") == "configs/training_config_cbm_final_hires.yaml"
    assert int(_arg(hires, "--batch-size")) * int(_arg(hires, "--grad-accum-steps")) == 32
    assert "--per-pixel-loss-weight" not in hires + final


def _record_calls(repo: Path, uv_exit: int = 0, uv_fails_on: str = "") -> Path:
    """Replace uv and sbatch with stubs that append their argv to one shared log.

    uv exits ``uv_exit`` for every call, or only for calls whose argv contains ``uv_fails_on``.
    """
    calls = repo.parent / "calls.txt"
    bin_dir = repo.parent / "bin"
    condition = f'[[ "$*" == *"{uv_fails_on}"* ]] && ' if uv_fails_on else ""
    (bin_dir / "uv").write_text(
        f'#!/bin/bash\necho "uv $*" >> "{calls}"\n{condition}exit {uv_exit}\nexit 0\n'
    )
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "sbatch $*" >> "{calls}"\necho 12345\n')
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return calls


def _launch_for_real(repo: Path, tmp_path_cache: Path) -> subprocess.CompletedProcess[str]:
    return _run_launcher(
        repo,
        "final",
        dry_run=False,
        env_overrides={
            "MERMAIDSEG_CACHE_DIR": str(tmp_path_cache),
            "MERMAIDSEG_SCRATCH": str(tmp_path_cache.parent / "scratch"),
        },
    )


_WARM_CALL = (
    "uv run --locked python scripts/warm_hf_cache.py "
    "--model-config configs/model_config_cbm_dpt_lora_vitl.yaml"
)


def test_final_syncs_then_warms_the_encoder_before_any_sbatch(repo):
    calls = _record_calls(repo)
    cache = repo.parent / "cache dir"

    result = _launch_for_real(repo, cache)

    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert lines[:2] == [
        f"uv run --locked python scripts/sync_data.py "
        f"--data-config configs/data_config.yaml --cache-dir {cache}",
        _WARM_CALL,
    ]
    assert [line.split()[0] for line in lines[2:]] == ["sbatch", "sbatch"]


def test_final_trains_both_jobs_on_the_model_config_it_warmed(repo):
    calls = _record_calls(repo)

    result = _launch_for_real(repo, repo.parent / "cache")

    assert result.returncode == 0, result.stderr
    sbatch_lines = [x for x in calls.read_text().splitlines() if x.startswith("sbatch")]
    assert len(sbatch_lines) == 2
    for line in sbatch_lines:
        assert "--model-config configs/model_config_cbm_dpt_lora_vitl.yaml" in line


def test_final_warms_the_hf_home_the_jobs_use(repo):
    bin_dir = repo.parent / "bin"
    seen = repo.parent / "hf_home.txt"
    (bin_dir / "uv").write_text(f'#!/bin/bash\necho "$HF_HOME" >> "{seen}"\n')
    (bin_dir / "uv").chmod(0o755)
    (bin_dir / "sbatch").write_text("#!/bin/bash\necho 12345\n")
    (bin_dir / "sbatch").chmod(0o755)

    result = _run_launcher(
        repo,
        "final",
        dry_run=False,
        env_overrides={
            "MERMAIDSEG_CACHE_DIR": str(repo.parent / "cache"),
            "MERMAIDSEG_SCRATCH": "/scratch-x",
        },
    )

    assert result.returncode == 0, result.stderr
    assert seen.read_text().splitlines()[-1] == "/scratch-x/hf-cache"


def test_final_submits_nothing_when_the_sync_fails(repo):
    calls = _record_calls(repo, uv_exit=3)

    result = _launch_for_real(repo, repo.parent / "cache")

    assert result.returncode != 0
    assert [line.split()[0] for line in calls.read_text().splitlines()] == ["uv"]


def test_final_submits_nothing_when_the_encoder_warm_fails(repo):
    calls = _record_calls(repo, uv_exit=3, uv_fails_on="warm_hf_cache")

    result = _launch_for_real(repo, repo.parent / "cache")

    assert result.returncode == 1
    assert [line.split()[0] for line in calls.read_text().splitlines()] == ["uv", "uv"]


def test_dry_run_neither_syncs_nor_warms(repo):
    calls = _record_calls(repo)

    result = _run_launcher(repo, "final")

    assert result.returncode == 0, result.stderr
    assert not calls.exists() or calls.read_text() == ""


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


@pytest.mark.parametrize("variable", ["SLURM_ACCOUNT", "MERMAIDSEG_LOG_DIR"])
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


# ---------- lib_launch.sh ----------

_HIRES = "cbm_final_res768"
_HIRES_SOURCE = "outputs/model_checkpoints/cbm_final/model_step300000"


def _age_checkpoint(repo: Path, minutes: float) -> Path:
    """Create the 768 run's source checkpoint, last written ``minutes`` ago."""
    path = repo / _HIRES_SOURCE
    path.parent.mkdir(parents=True)
    path.touch()
    stamp = time.time() - minutes * 60
    os.utime(path, (stamp, stamp))
    return path


def test_a_queued_run_is_reported_and_not_submitted_again(repo):
    out, jobs = _launch(repo, "final", queued=["cbm_final"])
    assert "queued  : cbm_final" in out
    assert "cbm_final" not in _runs(jobs)


def test_only_an_exact_job_name_counts_as_queued(repo):
    out, jobs = _launch(repo, "final", queued=["cbm_final_extra", "xcbm_final"])
    assert "queued  :" not in out
    assert "cbm_final" in _runs(jobs)


def test_a_queued_branch_run_is_reported_queued_not_waiting(repo):
    out, jobs = _launch(repo, "final", _HIRES, queued=[_HIRES])
    assert f"queued  : {_HIRES}" in out
    assert "waiting" not in out
    assert jobs == []


def test_a_branch_run_waits_for_a_source_checkpoint_that_does_not_exist(repo):
    out, jobs = _launch(repo, "final", _HIRES)
    assert f"waiting : {_HIRES} (needs {_HIRES_SOURCE})" in out
    assert jobs == []


def test_a_branch_run_waits_while_its_source_checkpoint_was_written_under_10_minutes_ago(repo):
    _age_checkpoint(repo, minutes=9)
    out, jobs = _launch(repo, "final", _HIRES)
    assert f"waiting : {_HIRES}" in out
    assert jobs == []


def test_a_branch_run_starts_once_its_source_checkpoint_is_over_10_minutes_old(repo):
    _age_checkpoint(repo, minutes=11)
    _, jobs = _launch(repo, "final", _HIRES)
    job = _runs(jobs)[_HIRES]
    assert _arg(job, "--init-weights") == _HIRES_SOURCE
    assert _opt(job, "dependency") is None


def test_an_unknown_launcher_option_exits_2_with_the_usage(repo):
    result = _run_launcher(repo, "final", "--no-such-option")
    assert result.returncode == 2
    assert "unknown option: --no-such-option" in result.stderr
    assert "usage: slurm/launch_final.sh" in result.stderr
    assert "would submit" not in result.stdout


def _install_numbering_sbatch(repo: Path) -> None:
    """An sbatch that prints ``<id>;<cluster>`` as ``--parsable`` does, ids from 5001."""
    sbatch = repo.parent / "bin" / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        'count_file="$(dirname "$0")/sbatch_count"\n'
        'n=$(($(cat "$count_file" 2>/dev/null || echo 0) + 1))\n'
        'echo "$n" > "$count_file"\n'
        'echo "$((5000 + n));cluster-a"\n'
    )


def test_the_job_id_of_a_real_submission_gates_the_dependent_run(repo):
    _install_numbering_sbatch(repo)
    (repo.parent / "bin" / "uv").write_text("#!/bin/bash\nexit 0\n")
    (repo.parent / "bin" / "uv").chmod(0o755)
    result = _launch_for_real(repo, repo.parent / "cache")
    assert result.returncode == 0, result.stderr
    submitted = re.findall(r"^  submitted (\d+): sbatch (.*)$", result.stdout, re.MULTILINE)
    assert [job_id for job_id, _ in submitted] == ["5001", "5002"]
    hires = shlex.split(submitted[1][1])
    assert _opt(hires, "dependency") == "afterok:5001"
    assert "submitted 2 job(s)" in result.stdout
