"""Launch a mermaid-segmentation training run as a SageMaker TrainingJob.

Reads a per-run YAML and submits CreateTrainingJob via the SageMaker
SDK's Estimator. The script is script-agnostic: it doesn't know what
the named `job.entrypoint` does. See
`mermaid-api/iac/sagemaker-launcher-convention.md` for the schema and
canonical ARNs.

Example
-------
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/my-run.yaml \\
        --config-dir sagemaker/configs/my-run/ \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2

    # Submit and return immediately (no streaming logs):
    uv run python scripts/launch_training.py \\
        --run-config sagemaker/runs/my-run.yaml \\
        --config-dir sagemaker/configs/my-run/ \\
        --mlflow-tracking-uri arn:aws:sagemaker:us-east-1:554812291621:mlflow-app/app-EJVJ6AVFDWW2 \\
        --no-wait
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session

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


def build_estimator_kwargs(
    *,
    cfg: RunConfig,
    run_id: str,
    staging_bucket: str,
    mlflow_uri: str,
    sm_session,
    role: str,
    hf_token: str | None = None,
) -> dict:
    job = cfg.job
    env = {
        "MLFLOW_TRACKING_URI": mlflow_uri,
        "AWS_DEFAULT_REGION": REGION,
        "CONTAINER_ENTRYPOINT_SCRIPT": job.entrypoint,
        **job.env,
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
    kwargs = {
        "image_uri": expand_image_uri(job.image),
        "role": role,
        "instance_count": job.instance_count,
        "instance_type": job.instance_type,
        "volume_size": job.volume_gb,
        "max_run": job.max_runtime_hours * 3600,
        "output_path": f"s3://{staging_bucket}/runs/{run_id}/output/",
        "environment": env,
        "sagemaker_session": sm_session,
        "base_job_name": job.name_prefix,
        "tags": [{"Key": k, "Value": v} for k, v in job.tags.items()],
    }
    if job.use_spot:
        kwargs["use_spot_instances"] = True
        kwargs["max_wait"] = job.max_runtime_hours * 3600 + 3600
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
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--role-arn", default=EXEC_ROLE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token to pass as HF_TOKEN env var to the container (for gated models).",
    )
    args = parser.parse_args(argv)

    cfg = parse_run_config(args.run_config.read_text(), kind="training", strict=False)
    if not args.config_dir.is_dir():
        log.error("Config dir does not exist: %s", args.config_dir)
        sys.exit(2)

    run_id = make_run_id(cfg.job.name_prefix)
    cw_url = _cloudwatch_url(run_id)

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
        print(f"CloudWatch:    {cw_url}")
        return

    boto_session = boto3.Session(region_name=REGION)
    sm_session = Session(boto_session=boto_session)
    key_prefix = f"runs/{run_id}/config"
    sm_session.upload_data(
        path=str(args.config_dir.resolve()), bucket=STAGING_BUCKET, key_prefix=key_prefix
    )

    kwargs = build_estimator_kwargs(
        cfg=cfg,
        run_id=run_id,
        staging_bucket=STAGING_BUCKET,
        mlflow_uri=args.mlflow_tracking_uri,
        sm_session=sm_session,
        role=args.role_arn,
        hf_token=args.hf_token,
    )

    log.info("Run ID:       %s", run_id)
    log.info("CloudWatch:   %s", cw_url)
    log.info("MLflow:       %s", args.mlflow_tracking_uri)

    estimator = Estimator(**kwargs)
    inputs = {
        "config": TrainingInput(s3_data=f"s3://{STAGING_BUCKET}/{key_prefix}/", input_mode="File"),
    }
    if cfg.training:
        for name, ch in cfg.training.channels.items():
            inputs[name] = TrainingInput(s3_data=ch.s3_uri, input_mode=ch.input_mode)

    log.info("Submitting TrainingJob...")
    estimator.fit(
        inputs=inputs,
        wait=not args.no_wait,
        logs=("All" if not args.no_wait else None),
        job_name=run_id,
    )
    if args.no_wait:
        log.info("Job submitted (--no-wait); run_id: %s", run_id)
        log.info("CloudWatch: %s", cw_url)
        return
    log.info("Job %s reached terminal state.", run_id)
    log.info("CloudWatch: %s", cw_url)


if __name__ == "__main__":
    main()
