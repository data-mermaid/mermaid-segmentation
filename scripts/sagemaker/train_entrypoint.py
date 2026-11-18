"""In-container entrypoint of a SageMaker TrainingJob, the counterpart of slurm/train.sbatch.

Reads the one run YAML from the ``config`` channel and replaces itself, from the repo root,
with the same trainer command the Slurm job runs:

    python scripts/train.py --run-name RUN ... <train.args>

with ``python -m torch.distributed.run`` in front when the instance has more than one GPU.
Secrets named by ``WANDB_API_KEY_SECRET_ID`` / ``HF_TOKEN_SECRET_ID`` are fetched from
Secrets Manager and exported as ``WANDB_API_KEY`` / ``HF_TOKEN``.

    python -m scripts.sagemaker.train_entrypoint
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from mermaidseg.lineage import BUILD_GIT_SHA_ENV
from scripts.sagemaker.run_config import TrainConfig, load_run_config

CONFIG_DIR = Path("/opt/ml/input/data/config")
INIT_DIR = Path("/opt/ml/input/data/init")
OUTPUT_DIR = Path("/opt/ml/checkpoints")
CACHE_DIR = Path("/opt/ml/cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SCRIPT = "scripts/train.py"

OWNED_FLAGS = (
    "--run-name",
    "--num-workers",
    "--nccl-timeout-min",
    "--preempt-checkpoint-every-steps",
    "--output-dir",
    "--cache-dir",
    "--checkpoint",
    "--init-weights",
)

# Secret-id variable -> the variable the trainer reads.
SECRET_ENV = {
    "WANDB_API_KEY_SECRET_ID": "WANDB_API_KEY",
    "HF_TOKEN_SECRET_ID": "HF_TOKEN",
}

# Cores kept for the main processes; the rest are split over the ranks (as train.sbatch).
_RESERVED_CPUS = 4


def find_run_yaml(config_dir: Path) -> Path:
    """Return the one run YAML in a directory.

    Args:
        config_dir: Directory holding the run YAML.

    Returns:
        The path of the only ``*.yaml`` or ``*.yml`` file directly in ``config_dir``.

    Raises:
        ValueError: If ``config_dir`` holds no run YAML, or more than one.
    """
    found = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one *.yaml or *.yml in {config_dir}, found {len(found)}: "
            f"{[path.name for path in found]}"
        )
    return found[0]


def resolve_secrets(environ: MutableMapping[str, str], client: Any) -> None:
    """Export each named Secrets Manager secret as the variable the trainer reads.

    Args:
        environ: Environment to read the secret ids from and write the values to.
        client: A boto3 Secrets Manager client.

    Raises:
        Exception: Whatever ``client.get_secret_value`` raises, with a note naming the id
            variable and the secret id; a run that needs a key must not start without it.
    """
    for id_var, var in SECRET_ENV.items():
        secret_id = environ.get(id_var)
        if secret_id:
            try:
                environ[var] = client.get_secret_value(SecretId=secret_id)["SecretString"]
            except Exception as exc:
                exc.add_note(f"fetching {id_var}={secret_id}")
                raise


def num_workers(cpus: int, ngpu: int) -> int:
    """Return the DataLoader workers per rank, as slurm/train.sbatch computes them.

    Args:
        cpus: CPUs on the instance.
        ngpu: GPUs on the instance, one rank each.

    Returns:
        ``(cpus - 4) // ngpu``.

    Raises:
        ValueError: If that is less than 1.
    """
    workers = (cpus - _RESERVED_CPUS) // ngpu
    if workers < 1:
        raise ValueError(f"{cpus} CPUs over {ngpu} GPU(s) leave {workers} workers per rank")
    return workers


def _owned_flags_in(args: Sequence[str]) -> list[str]:
    """Return the substrate-owned flags among ``args``, in either ``--flag`` or ``--flag=v`` form."""
    return [arg for arg in args if arg.split("=", 1)[0] in OWNED_FLAGS]


def build_trainer_argv(train: TrainConfig, ngpu: int, cpus: int, python: str) -> list[str]:
    """Build the trainer command for a run, as slurm/train.sbatch builds PIPELINE.

    Args:
        train: The run YAML's ``train`` block.
        ngpu: GPUs on the instance; more than one runs DDP under torchrun.
        cpus: CPUs on the instance.
        python: Python interpreter to run.

    Returns:
        The full argv, starting with ``python``.

    Raises:
        ValueError: If ``ngpu`` is less than 1, the CPUs leave no worker per rank, or
            ``train.args`` sets a flag the entrypoint owns.
    """
    if ngpu < 1:
        raise ValueError(f"training needs a GPU; found {ngpu}")
    owned = _owned_flags_in(train.args)
    if owned:
        raise ValueError(f"train.args must not set {owned}; the entrypoint sets them")
    pipeline = [
        TRAINER_SCRIPT,
        "--run-name",
        train.run_name,
        "--num-workers",
        str(num_workers(cpus, ngpu)),
        "--nccl-timeout-min",
        "30",
        "--preempt-checkpoint-every-steps",
        "0",
        "--output-dir",
        str(OUTPUT_DIR),
        "--cache-dir",
        str(CACHE_DIR),
    ]
    if train.init_weights is not None:
        pipeline += ["--init-weights", str(INIT_DIR / Path(train.init_weights).name)]
    pipeline += train.args
    if ngpu > 1:
        launcher = ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}"]
        return [python, *launcher, *pipeline]
    return [python, *pipeline]


def trainer_env(
    environ: Mapping[str, str], job_name: str | None, train: TrainConfig, ngpu: int
) -> dict[str, str]:
    """Return the variables to add to the trainer's environment.

    Args:
        environ: The current environment.
        job_name: The TrainingJob name (``MERMAIDSEG_JOB_NAME``).
        train: The run YAML's ``train`` block.
        ngpu: GPUs on the instance.

    Returns:
        The wandb run id, group and project, plus the NCCL settings for more than one GPU.

    Raises:
        ValueError: If ``job_name`` is missing or empty.
    """
    if not job_name:
        raise ValueError("MERMAIDSEG_JOB_NAME is not set")
    env = {
        # wandb run ids allow only alphanumerics, '_' and '-'.
        "WANDB_RUN_ID": re.sub(r"[^A-Za-z0-9_-]", "-", job_name),
        "WANDB_RUN_GROUP": train.group,
        "WANDB_PROJECT": environ.get("WANDB_PROJECT") or "mermaidseg",
    }
    if ngpu > 1:
        env["OMP_NUM_THREADS"] = "1"
        # On a timeout, the NCCL flight recorder logs each rank's last collectives.
        env["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "2000"
        env["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # Dumps land under the checkpoint channel so they outlive the container.
        env["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = str(
            OUTPUT_DIR / "nccl_trace" / f"{train.run_name}_rank_"
        )
    return env


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the entrypoint's command line, which takes no options."""
    parser = argparse.ArgumentParser(description="Run the trainer for a SageMaker TrainingJob.")
    return parser.parse_args(argv)


def _require_init_file(train: TrainConfig) -> None:
    """Raise ValueError when ``train.init_weights`` is set but absent from the init channel."""
    if train.init_weights is None:
        return
    # A single-object S3Prefix channel lands as INIT_DIR/<basename of the object>.
    expected = INIT_DIR / Path(train.init_weights).name
    if not expected.exists():
        found = sorted(path.name for path in INIT_DIR.iterdir()) if INIT_DIR.is_dir() else []
        raise ValueError(f"init weights {expected} not found; {INIT_DIR} holds {found}")


def _disk_report(path: Path) -> str:
    """One log line with a directory's capacity, free space, and filesystem device id."""
    usage = shutil.disk_usage(path)
    device = os.stat(path).st_dev
    gib = 1024**3
    return (
        f"[sagemaker] disk {path}: total={usage.total / gib:.1f}GiB "
        f"free={usage.free / gib:.1f}GiB dev={device}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trainer for the run YAML in the config channel.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        Nothing in practice: the process is replaced by the trainer.

    Raises:
        ValueError: If the build git SHA variable is empty or unset, or ``train.init_weights``
            is set and its file is not in the init channel.
    """
    _parse_args(argv)
    train = load_run_config(find_run_yaml(CONFIG_DIR)).train

    import torch

    ngpu = torch.cuda.device_count()
    command = build_trainer_argv(train, ngpu, os.cpu_count() or 0, sys.executable)

    if not os.environ.get(BUILD_GIT_SHA_ENV):
        raise ValueError(
            f"{BUILD_GIT_SHA_ENV} is not set: the image has no .git, so lineage would record "
            "an unknown or dirty tree and export would refuse the finished run"
        )
    _require_init_file(train)
    env = trainer_env(os.environ, os.environ.get("MERMAIDSEG_JOB_NAME"), train, ngpu)
    import boto3

    resolve_secrets(os.environ, boto3.client("secretsmanager"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ngpu > 1:
        Path(env["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"]).parent.mkdir(parents=True, exist_ok=True)
    # Differing dev ids show the cache is off the ML volume (e.g. the container's writable layer).
    for directory in (CACHE_DIR, OUTPUT_DIR):
        print(_disk_report(directory), flush=True)
    os.environ.update(env)
    os.chdir(REPO_ROOT)
    print(f"[sagemaker] {shlex.join(command)}", flush=True)
    # exec replaces this process, so tini (PID 1) delivers StopTrainingJob's SIGTERM
    # straight to the trainer (or torchrun).
    return os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
