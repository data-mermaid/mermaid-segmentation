"""Launch a mermaid-segmentation training run as a SageMaker TrainingJob.

SageMaker is the backup path; Slurm (slurm/engaging/) is the default. Both run the same
trainer, scripts/train.py, via the container entrypoint.

Reads a per-run YAML and submits CreateTrainingJob via sagemaker-core's
TrainingJob. The run YAML is the single source of truth: the same
file the launcher parses locally for the `job:` block is uploaded as
the container's `config` channel, so there is no second copy to drift
out of sync with it. The script is script-agnostic: it doesn't know
what the named `job.entrypoint` does. See
`mermaid-api/iac/sagemaker-launcher-convention.md` for the schema and
canonical ARNs.

Example
-------
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/cbm_vitl_example.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2

    # Submit and return immediately (no streaming logs):
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/cbm_vitl_example.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2 \\
        --no-wait

    # Validate the run YAML's trainer command offline and print the would-be job
    # parameters without submitting anything (no AWS call):
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/cbm_vitl_example.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
import sagemaker_train_entrypoint as entrypoint
import yaml

from mermaidseg.sagemaker.launcher_config import RunConfig, parse_run_config

ACCOUNT = "554812291621"
REGION = "us-east-1"
EXEC_ROLE = f"arn:aws:iam::{ACCOUNT}:role/dev-sm-execution-role"
STAGING_BUCKET = "dev-datamermaid-sm-data"
ECR_HOST = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
KNOWN_SHORT_REPOS = {"mermaid-segmentation-jobs"}

log = logging.getLogger("launch_training")


def expand_image_uri(image: str) -> str:
    if ".dkr.ecr." in image:
        return image
    if ":" not in image:
        raise ValueError(f"Image {image!r} must be `<repo>:<tag>` or a full ECR URI.")
    repo, _ = image.split(":", 1)
    if repo not in KNOWN_SHORT_REPOS:
        raise ValueError(f"Unknown short-form repo {repo!r}. Known: {sorted(KNOWN_SHORT_REPOS)}.")
    return f"{ECR_HOST}/{image}"


# SageMaker's TrainingJobName rule; the run id is the job name.
_JOB_NAME_RE = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")
_JOB_NAME_MAX = 63


def make_run_id(prefix: str) -> str:
    # The random suffix keeps two launches in the same second from sharing a job name and
    # overwriting each other's uploaded config.
    run_id = f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"
    if len(run_id) > _JOB_NAME_MAX or not _JOB_NAME_RE.fullmatch(run_id):
        raise ValueError(
            f"name_prefix {prefix!r} gives run id {run_id!r}, which is not a valid SageMaker job "
            f"name (<= {_JOB_NAME_MAX} chars of letters, digits and '-'); the prefix may be at "
            f"most {_JOB_NAME_MAX - (len(run_id) - len(prefix))} chars."
        )
    return run_id


# Env keys the launcher sets itself; a run YAML `job.env` may not override them.
RESERVED_ENV_KEYS = ("AWS_DEFAULT_REGION", "CONTAINER_ENTRYPOINT_SCRIPT", "MERMAIDSEG_RUN_ID")


def check_job_env(job_env: dict) -> None:
    """Raise ValueError when `job_env` sets a key the launcher reserves."""
    clashes = sorted(set(job_env) & set(RESERVED_ENV_KEYS))
    if clashes:
        raise ValueError(
            f"job.env must not set {', '.join(clashes)}: the launcher sets "
            f"{', '.join(RESERVED_ENV_KEYS)}"
        )


def _upload_run_config(run_config: Path, run_id: str, s3_client) -> str:
    """Upload the run YAML under `runs/<run_id>/config/` and return that prefix's S3 URI."""
    key_prefix = f"runs/{run_id}/config"
    s3_client.upload_file(str(run_config), STAGING_BUCKET, f"{key_prefix}/{run_config.name}")
    return f"s3://{STAGING_BUCKET}/{key_prefix}/"


def build_training_job_kwargs(
    *,
    cfg: RunConfig,
    run_id: str,
    staging_bucket: str,
    mlflow_uri: str | None,
    role: str,
    hf_token_secret_id: str | None = None,
    wandb_secret_id: str | None = None,
) -> dict:
    """Build the keyword arguments for `TrainingJob.create`."""
    # Imported lazily so the module imports without the optional `sagemaker` extra.
    from sagemaker.core.resources import Tag
    from sagemaker.core.shapes import (
        AlgorithmSpecification,
        Channel,
        CheckpointConfig,
        DataSource,
        OutputDataConfig,
        ResourceConfig,
        S3DataSource,
        StoppingCondition,
    )

    job = cfg.job
    check_job_env(job.env)
    env = {
        "AWS_DEFAULT_REGION": REGION,
        "CONTAINER_ENTRYPOINT_SCRIPT": job.entrypoint,
        # Stable per-job id, preserved across a Managed-Spot restart; the container entrypoint
        # derives WANDB_RUN_ID from it so a restart resumes the same wandb run.
        "MERMAIDSEG_RUN_ID": run_id,
        **job.env,
    }
    if mlflow_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_uri
    # Secret names only: the container entrypoint fetches the values from Secrets Manager, so
    # DescribeTrainingJob never exposes them. The execution role needs secretsmanager:GetSecretValue.
    if hf_token_secret_id:
        env["HF_TOKEN_SECRET_ID"] = hf_token_secret_id
    if wandb_secret_id:
        env["WANDB_API_KEY_SECRET_ID"] = wandb_secret_id

    def channel(name: str, s3_uri: str, input_mode: str) -> Channel:
        return Channel(
            channel_name=name,
            input_mode=input_mode,
            data_source=DataSource(
                s3_data_source=S3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri=s3_uri,
                    s3_data_distribution_type="FullyReplicated",
                )
            ),
        )

    channels = [channel("config", f"s3://{staging_bucket}/runs/{run_id}/config/", "File")]
    if cfg.training:
        channels += [
            channel(name, ch.s3_uri, ch.input_mode) for name, ch in cfg.training.channels.items()
        ]

    max_run = job.max_runtime_hours * 3600
    kwargs = {
        "training_job_name": run_id,
        "role_arn": role,
        "algorithm_specification": AlgorithmSpecification(
            training_image=expand_image_uri(job.image), training_input_mode="File"
        ),
        "resource_config": ResourceConfig(
            instance_type=job.instance_type,
            instance_count=job.instance_count,
            volume_size_in_gb=job.volume_gb,
        ),
        "stopping_condition": StoppingCondition(max_runtime_in_seconds=max_run),
        "output_data_config": OutputDataConfig(
            s3_output_path=f"s3://{staging_bucket}/runs/{run_id}/output/"
        ),
        "input_data_config": channels,
        "environment": env,
        "tags": [Tag(key=k, value=v) for k, v in job.tags.items()],
    }
    if job.use_spot:
        kwargs["enable_managed_spot_training"] = True
        kwargs["stopping_condition"] = StoppingCondition(
            max_runtime_in_seconds=max_run, max_wait_time_in_seconds=max_run + 3600
        )
        # SageMaker restores this prefix to /opt/ml/checkpoints after a spot interruption; the
        # entrypoint resumes the trainer from it.
        kwargs["checkpoint_config"] = CheckpointConfig(
            s3_uri=f"s3://{staging_bucket}/runs/{run_id}/checkpoints/",
            local_path="/opt/ml/checkpoints",
        )
    return kwargs


def _strings(value) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


def validate_trainer_config(seg: dict | None) -> list[str]:
    """Offline check of the run YAML `config:` block the container entrypoint will consume."""
    if seg is None:
        return ["Run YAML missing top-level `config:` block."]
    # The container reads the uploaded YAML unexpanded, so a variable would reach the trainer
    # literally.
    variables = [s for s in _strings(seg) if "$" in s]
    if variables:
        return [f"`$` is not allowed in the `config:` block (not expanded): {variables}"]
    try:
        entrypoint.build_trainer_argv(seg)
    except SystemExit as exc:
        return [str(exc)]
    return [f"Config file not found: {p}" for p in entrypoint.missing_config_files(seg)]


def _trainer_errors(seg: dict | None, cfg: RunConfig) -> list[str]:
    """Offline errors for a launch: the `config:` block checks plus, for spot, output-dir ownership."""
    errors = validate_trainer_config(seg)
    if not errors and cfg.job.use_spot:
        try:
            entrypoint.check_output_dir_owned(seg or {}, cfg.job.env)
        except SystemExit as exc:
            errors.append(str(exc))
    return errors


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _cloudwatch_url(run_id: str) -> str:
    return (
        f"https://{REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={REGION}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{run_id}"
    )


def main(argv=None):
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Optional; the CBM trainer logs to wandb by default. Defaults to MLFLOW_TRACKING_URI.",
    )
    parser.add_argument("--role-arn", default=EXEC_ROLE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--hf-token-secret-id",
        default=os.environ.get("HF_TOKEN_SECRET_ID"),
        help="Secrets Manager name/ARN of the HuggingFace token (gated models); the container "
        "fetches it as HF_TOKEN. Defaults to the HF_TOKEN_SECRET_ID environment variable.",
    )
    parser.add_argument(
        "--wandb-secret-id",
        default=os.environ.get("WANDB_API_KEY_SECRET_ID"),
        help="Secrets Manager name/ARN of the Weights & Biases key (the trainer logs to wandb); "
        "the container fetches it as WANDB_API_KEY. Defaults to WANDB_API_KEY_SECRET_ID.",
    )
    args = parser.parse_args(argv)

    cfg = parse_run_config(args.run_config.read_text(), kind="training", strict=False)

    try:
        run_id = make_run_id(cfg.job.name_prefix)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    cw_url = _cloudwatch_url(run_id)

    check_job_env(cfg.job.env)

    # Checked before any upload or submit, so a bad run YAML fails here and not on a GPU instance.
    seg = (yaml.safe_load(args.run_config.read_text()) or {}).get("config")
    errors = _trainer_errors(seg, cfg)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN -- not submitting")
        print("=" * 60)
        print(f"run_id:        {run_id}")
        print(f"role:          {args.role_arn}")
        print(f"image:         {expand_image_uri(cfg.job.image)}")
        print(f"entrypoint:    {cfg.job.entrypoint}")
        print(f"instance:      {cfg.job.instance_type} (x{cfg.job.instance_count})")
        print(f"volume_gb:     {cfg.job.volume_gb}")
        print(f"max_runtime:   {cfg.job.max_runtime_hours}h")
        print(f"output:        s3://{STAGING_BUCKET}/runs/{run_id}/output/")
        if cfg.job.use_spot:
            print(f"checkpoints:   s3://{STAGING_BUCKET}/runs/{run_id}/checkpoints/ (artifacts)")
        print(f"CloudWatch:    {cw_url}")
        print("-" * 60)
        print(f"trainer config validate: {'OK' if not errors else 'INVALID'}")
        for error in errors:
            print(f"  ERROR: {error}")
        if errors:
            sys.exit(1)
        return

    if errors:
        for error in errors:
            log.error("Invalid run YAML: %s", error)
        sys.exit(1)

    # Imported lazily so the module and its pure-logic helpers (build_training_job_kwargs,
    # expand_image_uri, CLI parsing) import without the optional `sagemaker` extra; only actual
    # job submission needs it. Keeps these unit tests runnable in CI without the extra.
    from sagemaker.core.resources import TrainingJob

    _upload_run_config(args.run_config.resolve(), run_id, boto3.client("s3", region_name=REGION))

    kwargs = build_training_job_kwargs(
        cfg=cfg,
        run_id=run_id,
        staging_bucket=STAGING_BUCKET,
        mlflow_uri=args.mlflow_tracking_uri,
        role=args.role_arn,
        hf_token_secret_id=args.hf_token_secret_id,
        wandb_secret_id=args.wandb_secret_id,
    )

    log.info("Run ID:       %s", run_id)
    log.info("CloudWatch:   %s", cw_url)
    if args.mlflow_tracking_uri:
        log.info("MLflow:       %s", args.mlflow_tracking_uri)

    log.info("Submitting TrainingJob...")
    job = TrainingJob.create(**kwargs, session=boto3.Session(region_name=REGION), region=REGION)
    if job is None:
        raise RuntimeError(f"TrainingJob.create returned no job for {run_id}")
    if not args.no_wait:
        # sagemaker-core's log client ignores the passed session and reads the ambient region.
        os.environ["AWS_DEFAULT_REGION"] = REGION
        job.wait(logs=True)
    if args.no_wait:
        log.info("Job submitted (--no-wait); run_id: %s", run_id)
        log.info("CloudWatch: %s", cw_url)
        return
    log.info("Job %s reached terminal state.", run_id)
    log.info("CloudWatch: %s", cw_url)


if __name__ == "__main__":
    main()
