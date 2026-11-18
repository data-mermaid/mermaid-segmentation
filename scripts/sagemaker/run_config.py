"""Schema of a SageMaker run YAML (sagemaker/runs/*.yaml).

A run YAML names the TrainingJob's resources (``job``) and the trainer run it executes
(``train``). Every key is required; unknown keys are errors.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError


class JobConfig(BaseModel):
    """Resources of the SageMaker TrainingJob.

    Attributes:
        name_prefix: Prefix of the TrainingJob name.
        image: Training image, as ``repository:tag``.
        instance_type: SageMaker instance type, e.g. ``ml.g6e.8xlarge``.
        volume_gb: Size of the ML storage volume in GB.
        max_runtime_hours: Hard limit on the job's runtime in hours.
    """

    model_config = ConfigDict(extra="forbid")

    name_prefix: str
    image: str
    instance_type: str
    volume_gb: PositiveInt
    max_runtime_hours: PositiveInt


class TrainConfig(BaseModel):
    """The trainer run the job executes.

    Attributes:
        run_name: Trainer run name (``--run-name``).
        group: wandb run group.
        init_weights: Starting weights for a fresh run, relative to the source job's
            checkpoint prefix, or None to start from scratch. The key is required.
        args: Arguments passed through to ``scripts/train.py`` (the Slurm PIPELINE_ARGS).
    """

    model_config = ConfigDict(extra="forbid")

    run_name: str
    group: str
    init_weights: str | None
    args: list[str]


class RunConfig(BaseModel):
    """A whole run YAML.

    Attributes:
        job: TrainingJob resources.
        train: Trainer run.
    """

    model_config = ConfigDict(extra="forbid")

    job: JobConfig
    train: TrainConfig


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run YAML.

    Args:
        path: Path of the run YAML.

    Returns:
        The validated run config.

    Raises:
        ValueError: If the file does not match the schema. The message starts with ``path``.
    """
    raw = yaml.safe_load(Path(path).read_text())
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
