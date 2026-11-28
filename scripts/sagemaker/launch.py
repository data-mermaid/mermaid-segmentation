"""Host-side launcher of a SageMaker TrainingJob for one run YAML (sagemaker/runs/*.yaml).

Validates the run YAML offline, then builds the CreateTrainingJob request, uploads the YAML
as the job's ``config`` channel and submits the job. It does not wait for the job.

Before it uploads or submits anything, a real launch pins ``job.image`` to its ECR digest and,
for a run with ``--init-from-job``, checks that the source job is ``Completed`` and that the
init checkpoint exists in the staging bucket. Any failure exits before an upload.

    python -m scripts.sagemaker.launch --run-config PATH [--init-from-job JOB]
        [--check | --dry-run]

``--check`` validates only and needs no environment. ``--dry-run`` also builds and prints the
request without calling AWS; it prints the image as a tag URI. Account values, including the
MLflow tracking server ARN (``SM_MLFLOW_TRACKING_ARN``, passed to the job as
``MLFLOW_TRACKING_URI``), come from the environment (see ``REQUIRED_ENV``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from mermaidseg.io import build_train_parser, check_train_args, resolve_repo_path
from scripts.sagemaker import train_entrypoint
from scripts.sagemaker.run_config import RunConfig, load_run_config

REGION = "us-east-1"

REQUIRED_ENV = (
    "SM_ROLE_ARN",
    "SM_STAGING_BUCKET",
    "SM_ECR_REGISTRY",
    "HF_TOKEN_SECRET_ID",
    "SM_MLFLOW_TRACKING_ARN",
)

# Any count that leaves workers per rank; validation only parses the resulting argv.
_VALIDATION_CPUS = 32

# Trainer options that name a config file in the repository.
_CONFIG_DESTS = (
    "training_config",
    "model_config",
    "data_config",
    "logger_config",
)

# SageMaker's TrainingJobName rule.
_JOB_NAME_RE = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")
_JOB_NAME_MAX = 63


def _is_job_name(name: str) -> bool:
    """Return whether ``name`` is a valid SageMaker TrainingJob name."""
    return len(name) <= _JOB_NAME_MAX and _JOB_NAME_RE.fullmatch(name) is not None


def make_run_id(prefix: str) -> str:
    """Return a new TrainingJob name: the prefix, a UTC timestamp and a random suffix.

    Args:
        prefix: The run YAML's ``job.name_prefix``.

    Returns:
        ``<prefix>-<YYYYmmddTHHMMSSZ>-<6 hex>``.

    Raises:
        ValueError: If the result is not a valid SageMaker job name.
    """
    # The random suffix keeps two launches in the same second from sharing a job name and
    # overwriting each other's uploaded config.
    run_id = f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"
    if not _is_job_name(run_id):
        raise ValueError(
            f"name_prefix {prefix!r} gives run id {run_id!r}, which is not a valid SageMaker job "
            f"name (<= {_JOB_NAME_MAX} chars of letters, digits and '-'); the prefix may be at "
            f"most {_JOB_NAME_MAX - (len(run_id) - len(prefix))} chars."
        )
    return run_id


def _check_init_from_job(cfg: RunConfig, init_from_job: str | None) -> None:
    """Raise ValueError unless ``train.init_weights`` and ``--init-from-job`` come together."""
    if (cfg.train.init_weights is None) != (init_from_job is None):
        raise ValueError(
            "train.init_weights and --init-from-job must be given together: "
            f"init_weights={cfg.train.init_weights!r}, --init-from-job={init_from_job!r}"
        )
    if init_from_job is not None and not _is_job_name(init_from_job):
        raise ValueError(f"--init-from-job {init_from_job!r} is not a valid SageMaker job name")


def _raise_parse_error(message: str) -> NoReturn:
    """Raise an argparse error as a ValueError, in place of printing usage and exiting."""
    raise ValueError(f"train.args: {message}")


def _parse_trainer_args(cfg: RunConfig) -> argparse.Namespace:
    """Parse the trainer command the job runs with the trainer's own parser.

    Raises:
        ValueError: If ``train.args`` sets an owned flag or the trainer parser rejects them.
    """
    argv = train_entrypoint.build_trainer_argv(
        cfg.train, ngpu=1, cpus=_VALIDATION_CPUS, python="python"
    )
    parser = build_train_parser()
    parser.error = _raise_parse_error  # type: ignore[method-assign]
    args = parser.parse_args(argv[argv.index(train_entrypoint.TRAINER_SCRIPT) + 1 :])
    check_train_args(parser, args)
    return args


def _missing_config_paths(namespace: argparse.Namespace) -> list[str]:
    """Return the trainer config paths that do not exist in the repository."""
    paths = [getattr(namespace, dest) for dest in _CONFIG_DESTS]
    return [path for path in paths if not resolve_repo_path(path).exists()]


def validate_run(cfg: RunConfig, init_from_job: str | None) -> None:
    """Check offline that a run YAML would start the trainer.

    Args:
        cfg: The run YAML.
        init_from_job: The TrainingJob whose checkpoints hold ``train.init_weights``.

    Raises:
        ValueError: If ``init_weights`` and ``init_from_job`` are not given together, the job
            name is invalid, the trainer rejects ``train.args``, or a config file is missing.
    """
    _check_init_from_job(cfg, init_from_job)
    missing = _missing_config_paths(_parse_trainer_args(cfg))
    if missing:
        raise ValueError(f"config files not found in the repository: {missing}")


def read_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the launch variables from an environment.

    Args:
        environ: The environment to read.

    Returns:
        Each variable of ``REQUIRED_ENV`` and its value.

    Raises:
        ValueError: Naming every variable of ``REQUIRED_ENV`` that is unset or empty.
    """
    missing = [var for var in REQUIRED_ENV if not environ.get(var)]
    if missing:
        raise ValueError(f"environment variables not set: {', '.join(missing)}")
    return {var: environ[var] for var in REQUIRED_ENV}


def _image_uri(registry: str, image: str) -> str:
    """Return the training image's full URI from its short ``repository:tag`` form."""
    if ".dkr.ecr." in image:
        raise ValueError(
            f"job.image {image!r} must be the short <repo>:<tag> form; "
            "SM_ECR_REGISTRY supplies the registry"
        )
    return f"{registry}/{image}"


def _pin_image(ecr: Any, registry: str, image: str) -> str:
    """Return the training image URI pinned to its ECR digest.

    Args:
        ecr: An ECR client.
        registry: The ``SM_ECR_REGISTRY`` value.
        image: ``job.image``; a ``repo@sha256:...`` value is used as given, with no ECR call.

    Raises:
        ValueError: If ``image`` is a full ECR URI or ECR cannot resolve its tag.
    """
    from botocore.exceptions import ClientError

    uri = _image_uri(registry, image)
    if "@" in image:
        return uri
    repo, _, tag = image.partition(":")
    try:
        details = ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise ValueError(f"cannot resolve job.image {image!r} in ECR: {code}") from exc
    digest = details["imageDetails"][0]["imageDigest"]
    print(f"Image: {repo}:{tag} -> {digest}")
    return f"{registry}/{repo}@{digest}"


def _init_checkpoint_key(init_from_job: str, init_weights: str) -> str:
    """Return the staging-bucket key of a source job's checkpoint."""
    return f"runs/{init_from_job}/checkpoints/{init_weights}"


def _check_init_source(
    sagemaker: Any, s3: Any, bucket: str, init_from_job: str, init_weights: str
) -> None:
    """Check that the source job completed and its init checkpoint is readable.

    Raises:
        ValueError: Naming the job and its status, or the checkpoint's s3:// URI and the S3
            error code, when a check fails.
    """
    from botocore.exceptions import ClientError

    try:
        status = sagemaker.describe_training_job(TrainingJobName=init_from_job)["TrainingJobStatus"]
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise ValueError(f"cannot describe --init-from-job {init_from_job!r}: {code}") from exc
    if status != "Completed":
        raise ValueError(f"--init-from-job {init_from_job!r} is {status}, not Completed")
    key = _init_checkpoint_key(init_from_job, init_weights)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise ValueError(f"init checkpoint s3://{bucket}/{key} is not readable: {code}") from exc


def _s3_channel(name: str, uri: str) -> dict[str, Any]:
    """Return a fully replicated S3Prefix input channel."""
    return {
        "ChannelName": name,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def build_request(
    cfg: RunConfig, job_name: str, env: Mapping[str, str], init_from_job: str | None
) -> dict[str, Any]:
    """Build the CreateTrainingJob keyword arguments for a run.

    Args:
        cfg: The run YAML.
        job_name: The TrainingJob name.
        env: The launch variables, as ``read_env`` returns them.
        init_from_job: The TrainingJob whose checkpoints hold ``train.init_weights``.

    Returns:
        Keyword arguments of boto3's ``create_training_job``.

    Raises:
        ValueError: If ``job.image`` is a full ECR URI.
    """
    runs = f"s3://{env['SM_STAGING_BUCKET']}/runs"
    channels = [_s3_channel("config", f"{runs}/{job_name}/config/")]
    if cfg.train.init_weights is not None:
        key = _init_checkpoint_key(str(init_from_job), cfg.train.init_weights)
        init_uri = f"s3://{env['SM_STAGING_BUCKET']}/{key}"
        channels.append(_s3_channel("init", init_uri))
    return {
        "TrainingJobName": job_name,
        "RoleArn": env["SM_ROLE_ARN"],
        "AlgorithmSpecification": {
            "TrainingImage": _image_uri(env["SM_ECR_REGISTRY"], cfg.job.image),
            "TrainingInputMode": "File",
        },
        "ResourceConfig": {
            "InstanceType": cfg.job.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": cfg.job.volume_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": cfg.job.max_runtime_hours * 3600},
        "OutputDataConfig": {"S3OutputPath": f"{runs}/{job_name}/output/"},
        "CheckpointConfig": {
            "S3Uri": f"{runs}/{job_name}/checkpoints/",
            "LocalPath": str(train_entrypoint.OUTPUT_DIR),
        },
        "InputDataConfig": channels,
        # Secret ids and non-secret settings only: the entrypoint fetches secret values,
        # so a secret value never goes here.
        # MLflow timeout and retries are below the client defaults (120 s, 7 retries) so an
        # unreachable tracking server fails fast instead of stalling a billed job.
        "Environment": {
            "HF_TOKEN_SECRET_ID": env["HF_TOKEN_SECRET_ID"],
            "MLFLOW_TRACKING_URI": env["SM_MLFLOW_TRACKING_ARN"],
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "30",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
        },
    }


def _console_url(job_name: str) -> str:
    """Return the SageMaker console URL of a TrainingJob."""
    return (
        f"https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}"
    )


def _cloudwatch_url(job_name: str) -> str:
    """Return the CloudWatch URL of a TrainingJob's log streams."""
    return (
        f"https://{REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={REGION}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FTrainingJobs$3FlogStreamNameFilter$3D{job_name}"
    )


def _submit(
    run_config: Path,
    request: Mapping[str, Any],
    cfg: RunConfig,
    env: Mapping[str, str],
    init_from_job: str | None,
) -> None:
    """Check the image and init source, upload the run YAML, then create the TrainingJob.

    Raises:
        ValueError: If the image or the init source check fails; nothing is uploaded then.
    """
    import boto3

    bucket = env["SM_STAGING_BUCKET"]
    job_name = request["TrainingJobName"]
    s3 = boto3.client("s3", region_name=REGION)
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    image = _pin_image(
        boto3.client("ecr", region_name=REGION), env["SM_ECR_REGISTRY"], cfg.job.image
    )
    if init_from_job is not None and cfg.train.init_weights is not None:
        _check_init_source(sagemaker, s3, bucket, init_from_job, cfg.train.init_weights)
    pinned = {
        **request,
        "AlgorithmSpecification": {**request["AlgorithmSpecification"], "TrainingImage": image},
    }
    s3.upload_file(str(run_config), bucket, f"runs/{job_name}/config/{run_config.name}")
    sagemaker.create_training_job(**pinned)
    print(f"Submitted TrainingJob {job_name}")
    print(f"Console:    {_console_url(job_name)}")
    print(f"CloudWatch: {_cloudwatch_url(job_name)}")


def _parse_cli(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the launcher's command line."""
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker TrainingJob for a run YAML.", allow_abbrev=False
    )
    parser.add_argument("--run-config", required=True, type=Path, help="path of the run YAML")
    parser.add_argument(
        "--init-from-job",
        default=None,
        help="TrainingJob whose checkpoints hold train.init_weights",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate the run YAML and stop")
    mode.add_argument(
        "--dry-run", action="store_true", help="print the CreateTrainingJob request and stop"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a run YAML and, unless ``--check`` or ``--dry-run``, submit its TrainingJob.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        0 on success. A validation or environment error exits with ``ERROR: <message>``.
    """
    args = _parse_cli(argv)
    try:
        cfg = load_run_config(args.run_config)
        validate_run(cfg, args.init_from_job)
        if args.check:
            print(f"OK: {args.run_config}")
            return 0
        env = read_env(os.environ)
        request = build_request(cfg, make_run_id(cfg.job.name_prefix), env, args.init_from_job)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    try:
        _submit(args.run_config, request, cfg, env, args.init_from_job)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
