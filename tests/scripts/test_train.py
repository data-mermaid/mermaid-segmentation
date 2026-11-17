"""Training entrypoint parser."""

from __future__ import annotations

import pytest

import scripts.train as train_script
from mermaidseg.run_setup import resolve_repo_path
from scripts.train import build_parser
from tests.slurm import test_launchers as launchers

REQUIRED = ["--num-workers", "8"]

repo = launchers.repo
job = launchers.job


CONFIG_FLAGS = (
    "training_config",
    "model_config",
    "data_config",
    "concept_schema_config",
    "logger_config",
)


def test_config_defaults_point_at_existing_repo_files():
    args = build_parser().parse_args(REQUIRED)
    for flag in CONFIG_FLAGS:
        assert resolve_repo_path(getattr(args, flag)).is_file(), flag


def test_num_workers_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_nccl_timeout_minutes_parses_as_float():
    assert (
        build_parser().parse_args([*REQUIRED, "--nccl-timeout-min", "2.5"]).nccl_timeout_min == 2.5
    )


@pytest.mark.parametrize("value", ["0", "-5"])
def test_main_rejects_non_positive_nccl_timeout(monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["train.py", *REQUIRED, "--nccl-timeout-min", value])
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()


def test_main_rejects_zero_num_workers(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--num-workers", "0"])
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()


def test_main_rejects_checkpoint_with_init_weights(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", *REQUIRED, "--checkpoint", "x", "--init-weights", "y"]
    )
    monkeypatch.setattr(train_script, "run", lambda args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        train_script.main()


def test_every_final_launcher_job_parses(repo, job):
    _, jobs = launchers._launch(repo, "final")
    runs = launchers._runs(jobs)
    assert list(runs) == ["cbm_final", "cbm_final_res768"]
    for name, submitted in runs.items():
        (job / "outputs" / "model_checkpoints" / name).mkdir(parents=True, exist_ok=True)
        init_weights = launchers._arg(submitted, "--init-weights")
        if init_weights is not None:
            (job / init_weights).touch()
        sbatch_args = submitted[submitted.index("slurm/train.sbatch") + 1 :]
        result, argv = launchers._run_train(job, *sbatch_args)
        assert result.returncode == 0, result.stderr
        assert argv[1] == "scripts/train.py"
        args = build_parser().parse_args(argv[2:])
        assert args.run_name == name
        assert (args.init_weights is not None) == (init_weights is not None)
