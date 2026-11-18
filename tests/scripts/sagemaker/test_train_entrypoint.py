"""SageMaker in-container training entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

import scripts.sagemaker.train_entrypoint as entrypoint
from mermaidseg.lineage import BUILD_GIT_SHA_ENV
from scripts.sagemaker.run_config import TrainConfig, load_run_config
from scripts.sagemaker.train_entrypoint import (
    OWNED_FLAGS,
    build_trainer_argv,
    find_run_yaml,
    num_workers,
    resolve_secrets,
    trainer_env,
)
from tests.slurm import test_launchers as launchers

repo = launchers.repo
job = launchers.job

_RUNS = Path(__file__).resolve().parents[3] / "sagemaker" / "runs"


def _train(init_weights: str | None = None, args: list[str] | None = None) -> TrainConfig:
    return TrainConfig(run_name="r", init_weights=init_weights, args=args or ["--batch-size", "8"])


def _without_owned(argv: list[str], *extra_flags: str) -> list[str]:
    """Trainer argv from scripts/train.py on, with owned and extra flag/value pairs removed."""
    tokens = argv[argv.index("scripts/train.py") :]
    kept: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
        elif token in (*OWNED_FLAGS, *extra_flags):
            skip = True
        else:
            kept.append(token)
    return kept


# ---------- same command as Slurm ----------


@pytest.mark.parametrize(("name", "ngpu"), [("cbm_final", 1), ("cbm_final_res768", 4)])
def test_trainer_argv_matches_the_slurm_launcher(repo, job, name, ngpu):
    _, jobs = launchers._launch(repo, "final")
    submitted = launchers._runs(jobs)[name]
    (job / "outputs" / "model_checkpoints" / name).mkdir(parents=True, exist_ok=True)
    slurm_init = launchers._arg(submitted, "--init-weights")
    if slurm_init is not None:
        (job / slurm_init).parent.mkdir(parents=True, exist_ok=True)
        (job / slurm_init).touch()
    result, slurm_argv = launchers._run_train(
        job, *submitted[submitted.index("slurm/train.sbatch") + 1 :]
    )
    assert result.returncode == 0, result.stderr

    config = load_run_config(_RUNS / f"{name}.yaml")
    ours = build_trainer_argv(config.train, ngpu=ngpu, cpus=48, python="py")

    # SageMaker alone swaps the tracker, so only it passes a logger config.
    assert _without_owned(ours, "--logger-config") == _without_owned(slurm_argv)
    # The Slurm harness runs 1 GPU and 32 CPUs; the values the substrate owns agree there.
    same_host = build_trainer_argv(config.train, ngpu=1, cpus=32, python="py")
    for flag in ("--nccl-timeout-min", "--num-workers"):
        assert launchers._arg(same_host, flag) == launchers._arg(slurm_argv, flag) is not None
    ours_init = launchers._arg(ours, "--init-weights")
    if slurm_init is None:
        assert ours_init is None
    else:
        assert ours_init is not None
        assert Path(ours_init).name == Path(slurm_init).name == "model_step300000"


# ---------- build_trainer_argv ----------


def test_single_gpu_runs_train_py_without_torchrun():
    assert build_trainer_argv(_train(), ngpu=1, cpus=32, python="py") == [
        "py",
        "scripts/train.py",
        "--run-name",
        "r",
        "--num-workers",
        "28",
        "--nccl-timeout-min",
        "30",
        "--preempt-checkpoint-every-steps",
        "0",
        "--output-dir",
        "/opt/ml/checkpoints",
        "--cache-dir",
        "/opt/ml/cache",
        "--batch-size",
        "8",
    ]


def test_several_gpus_run_under_torchrun_with_one_process_per_gpu():
    argv = build_trainer_argv(_train(), ngpu=4, cpus=48, python="py")
    assert argv[:6] == [
        "py",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/train.py",
    ]
    assert launchers._arg(argv, "--num-workers") == "11"


def test_no_gpu_raises():
    with pytest.raises(ValueError, match="GPU"):
        build_trainer_argv(_train(), ngpu=0, cpus=32, python="py")


def test_init_weights_point_into_the_init_channel():
    argv = build_trainer_argv(
        _train(init_weights="model_checkpoints/cbm_final/model_step300000"),
        ngpu=1,
        cpus=32,
        python="py",
    )
    assert launchers._arg(argv, "--init-weights") == "/opt/ml/input/data/init/model_step300000"
    assert argv.index("--init-weights") < argv.index("--batch-size")


@pytest.mark.parametrize(
    "flag",
    [
        "--run-name",
        "--num-workers",
        "--nccl-timeout-min",
        "--preempt-checkpoint-every-steps",
        "--output-dir",
        "--cache-dir",
        "--checkpoint",
        "--init-weights",
    ],
)
def test_owned_flag_in_train_args_raises(flag):
    with pytest.raises(ValueError, match=flag):
        build_trainer_argv(_train(args=[flag, "x"]), ngpu=1, cpus=32, python="py")


def test_owned_flag_in_equals_form_raises():
    with pytest.raises(ValueError, match="--output-dir"):
        build_trainer_argv(_train(args=["--output-dir=/x"]), ngpu=1, cpus=32, python="py")


# ---------- num_workers ----------


@pytest.mark.parametrize(("cpus", "ngpu", "expected"), [(32, 1, 28), (48, 4, 11), (5, 1, 1)])
def test_num_workers_splits_spare_cpus_over_ranks(cpus, ngpu, expected):
    assert num_workers(cpus, ngpu) == expected


@pytest.mark.parametrize(("cpus", "ngpu"), [(4, 1), (7, 4)])
def test_num_workers_below_one_raises(cpus, ngpu):
    with pytest.raises(ValueError):
        num_workers(cpus, ngpu)


# ---------- resolve_secrets ----------


class _FakeSecrets:
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets
        self.requested: list[str] = []

    def get_secret_value(self, SecretId: str) -> dict[str, str]:  # noqa: N803 - boto3 name
        self.requested.append(SecretId)
        if SecretId not in self.secrets:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue")
        return {"SecretString": self.secrets[SecretId]}


def test_resolve_secrets_sets_hf_token():
    environ = {"HF_TOKEN_SECRET_ID": "h-id"}
    resolve_secrets(environ, _FakeSecrets({"h-id": "hf-value"}))
    assert environ["HF_TOKEN"] == "hf-value"


def test_resolve_secrets_skips_unset_ids():
    environ = {"HF_TOKEN": "kept"}
    client = _FakeSecrets({})
    resolve_secrets(environ, client)
    assert environ == {"HF_TOKEN": "kept"}
    assert client.requested == []


def test_resolve_secrets_propagates_a_failed_fetch_naming_the_secret():
    with pytest.raises(ClientError) as exc_info:
        resolve_secrets({"HF_TOKEN_SECRET_ID": "missing"}, _FakeSecrets({}))
    assert "fetching HF_TOKEN_SECRET_ID=missing" in exc_info.value.__notes__


# ---------- trainer_env ----------


def test_trainer_env_single_gpu_adds_nothing():
    assert trainer_env(_train(), ngpu=1) == {}


def test_trainer_env_adds_nccl_settings_for_several_gpus():
    env = trainer_env(_train(), ngpu=4)
    assert set(env) == {
        "OMP_NUM_THREADS",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
    }
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert env["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] == "/opt/ml/checkpoints/nccl_trace/r_rank_"


# ---------- find_run_yaml ----------


@pytest.mark.parametrize("names", [[], ["a.yaml", "b.yml"]])
def test_find_run_yaml_needs_exactly_one(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text("")
    with pytest.raises(ValueError):
        find_run_yaml(tmp_path)


def test_find_run_yaml_returns_the_only_one(tmp_path):
    (tmp_path / "run.yml").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert find_run_yaml(tmp_path) == tmp_path / "run.yml"


# ---------- main ----------


_MAIN_WRITTEN_VARS = (
    "HF_TOKEN",
    "OMP_NUM_THREADS",
    "TORCH_NCCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
)


def _stub_main(tmp_path, monkeypatch, ngpu, run="cbm_final"):
    """Point main at a run YAML and stub every effect; return the execvp calls."""
    (tmp_path / "run.yaml").write_text((_RUNS / f"{run}.yaml").read_text())
    monkeypatch.setattr(entrypoint, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(entrypoint, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()  # SageMaker always creates the checkpoint dir
    monkeypatch.setattr(entrypoint, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("torch.cuda.device_count", lambda: ngpu)
    monkeypatch.setattr(os, "cpu_count", lambda: 48)
    monkeypatch.setattr(os, "chdir", lambda *a: None)
    # main writes these into the real environment; delenv makes teardown restore them.
    for var in _MAIN_WRITTEN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("boto3.client", lambda *a, **k: object())
    monkeypatch.setattr(entrypoint, "resolve_secrets", lambda *a: None)
    execs: list[list[str]] = []
    monkeypatch.setattr(os, "execvp", lambda file, argv: execs.append(argv) or 0)
    return execs


@pytest.mark.parametrize("sha", [None, ""], ids=["unset", "empty"])
def test_main_without_git_sha_raises_before_secrets_or_exec(tmp_path, monkeypatch, sha):
    execs = _stub_main(tmp_path, monkeypatch, ngpu=1)
    if sha is None:
        monkeypatch.delenv(BUILD_GIT_SHA_ENV, raising=False)
    else:
        monkeypatch.setenv(BUILD_GIT_SHA_ENV, sha)
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("no secrets fetch"))

    with pytest.raises(ValueError, match=BUILD_GIT_SHA_ENV):
        entrypoint.main([])

    assert execs == []


def test_main_creates_the_nccl_trace_dir_for_several_gpus(tmp_path, monkeypatch):
    _stub_main(tmp_path, monkeypatch, ngpu=4)
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")

    entrypoint.main([])

    assert (tmp_path / "out" / "nccl_trace").is_dir()


def test_main_creates_no_nccl_trace_dir_for_one_gpu(tmp_path, monkeypatch):
    _stub_main(tmp_path, monkeypatch, ngpu=1)
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")

    entrypoint.main([])

    assert not (tmp_path / "out" / "nccl_trace").exists()


def test_main_logs_disk_usage_of_cache_and_output_dirs_before_exec(tmp_path, monkeypatch, capsys):
    _stub_main(tmp_path, monkeypatch, ngpu=4)
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")
    gib = 1024**3
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: SimpleNamespace(total=100 * gib, used=60 * gib, free=40 * gib),
    )

    entrypoint.main([])

    lines = capsys.readouterr().out.splitlines()
    cache_line = next(
        i for i, ln in enumerate(lines) if "total=" in ln and str(tmp_path / "cache") in ln
    )
    out_line = next(
        i for i, ln in enumerate(lines) if "total=" in ln and str(tmp_path / "out") in ln
    )
    exec_line = next(i for i, ln in enumerate(lines) if "train.py" in ln)
    assert "total=100.0GiB free=40.0GiB" in lines[cache_line]
    assert "dev=" in lines[out_line]
    assert max(cache_line, out_line) < exec_line


def test_main_passes_the_init_channel_file_as_init_weights(tmp_path, monkeypatch):
    execs = _stub_main(tmp_path, monkeypatch, ngpu=4, run="cbm_final_res768")
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")
    init_dir = tmp_path / "init"
    init_dir.mkdir()
    (init_dir / "model_step300000").touch()
    monkeypatch.setattr(entrypoint, "INIT_DIR", init_dir)

    entrypoint.main([])

    assert len(execs) == 1
    argv = execs[0]
    assert argv[argv.index("--init-weights") + 1] == str(init_dir / "model_step300000")


def test_main_without_the_init_file_raises_listing_the_init_channel(tmp_path, monkeypatch):
    execs = _stub_main(tmp_path, monkeypatch, ngpu=4, run="cbm_final_res768")
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")
    init_dir = tmp_path / "init"
    (init_dir / "model_checkpoints").mkdir(parents=True)
    (init_dir / "other.pt").touch()
    monkeypatch.setattr(entrypoint, "INIT_DIR", init_dir)

    with pytest.raises(ValueError, match=r"model_step300000.*'model_checkpoints', 'other\.pt'"):
        entrypoint.main([])

    assert execs == []


def _wire_real_secrets(monkeypatch, client):
    monkeypatch.setattr(entrypoint, "resolve_secrets", resolve_secrets)
    monkeypatch.setattr("boto3.client", lambda *a, **k: client)
    monkeypatch.setenv("HF_TOKEN_SECRET_ID", "h-id")


def test_main_exports_the_fetched_secret_at_exec_without_leaking_it(tmp_path, monkeypatch, capsys):
    execs = _stub_main(tmp_path, monkeypatch, ngpu=1)
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")
    _wire_real_secrets(monkeypatch, _FakeSecrets({"h-id": "hf-sekret"}))
    seen: dict[str, str | None] = {}

    def fake_execvp(file, argv):
        execs.append(argv)
        seen["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        return 0

    monkeypatch.setattr(os, "execvp", fake_execvp)

    entrypoint.main([])

    assert seen == {"HF_TOKEN": "hf-sekret"}
    assert len(execs) == 1
    assert not any("hf-sekret" in arg for arg in execs[0])
    assert "hf-sekret" not in capsys.readouterr().out


def test_main_propagates_a_failed_secret_fetch_without_exec(tmp_path, monkeypatch):
    execs = _stub_main(tmp_path, monkeypatch, ngpu=1)
    monkeypatch.setenv(BUILD_GIT_SHA_ENV, "abc123")
    _wire_real_secrets(monkeypatch, _FakeSecrets({}))

    with pytest.raises(ClientError):
        entrypoint.main([])

    assert execs == []
