"""Launch a mermaid-segmentation training run as a SageMaker TrainingJob.

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
        --run-config sagemaker/runs/my-run.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2

    # Submit and return immediately (no streaming logs):
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/my-run.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2 \\
        --no-wait

    # Validate the run YAML (mermaidseg.experiment) and print the would-be job
    # parameters without submitting anything:
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/my-run.yaml \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3

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


def make_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


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
    mlflow_uri: str,
    role: str,
    hf_token: str | None = None,
) -> dict:
    """Build the keyword arguments for `TrainingJob.create`."""
    # Imported lazily so the module imports without the optional `sagemaker` extra.
    from sagemaker.core.resources import Tag
    from sagemaker.core.shapes import (
        AlgorithmSpecification,
        Channel,
        DataSource,
        OutputDataConfig,
        ResourceConfig,
        S3DataSource,
        StoppingCondition,
    )

    job = cfg.job
    env = {
        "MLFLOW_TRACKING_URI": mlflow_uri,
        "AWS_DEFAULT_REGION": REGION,
        "CONTAINER_ENTRYPOINT_SCRIPT": job.entrypoint,
        # Stable per-job id, preserved across a Managed-Spot restart, so the training container can
        # find and resume its prior MLflow run (see Logger._find_run_by_resume_key). Harmless on
        # on-demand runs (it just tags the run).
        "MERMAIDSEG_RUN_ID": run_id,
        **job.env,
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token

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
    return kwargs


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--role-arn", default=EXEC_ROLE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token to pass as HF_TOKEN env var to the container (for gated models). "
        "Defaults to the HF_TOKEN environment variable.",
    )
    args = parser.parse_args(argv)

    cfg = parse_run_config(args.run_config.read_text(), kind="training", strict=False)

    run_id = make_run_id(cfg.job.name_prefix)
    cw_url = _cloudwatch_url(run_id)

    if args.dry_run:
        # Imported lazily (mermaidseg.experiment pulls in torch at module level) so
        # --run-config parsing and CLI arg handling stay import-light otherwise.
        from mermaidseg.experiment import Experiment

        report = Experiment.validate(args.run_config)
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
        print(f"CloudWatch:    {cw_url}")
        print("-" * 60)
        print(f"experiment validate: {'OK' if report.ok else 'INVALID'}")
        for error in report.errors:
            print(f"  ERROR: {error}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        if not report.ok:
            sys.exit(1)
        return

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
        hf_token=args.hf_token,
    )

    log.info("Run ID:       %s", run_id)
    log.info("CloudWatch:   %s", cw_url)
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
