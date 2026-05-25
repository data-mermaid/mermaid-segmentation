"""Launch SageMaker ProcessingJob(s) for mermaid-segmentation.

Reads a per-run YAML and submits one or more CreateProcessingJob calls
via raw boto3. With `processing.shard:` set, fans out N parallel
ProcessingJobs each running the container against a sharded subset of
items. See
`mermaid-api/iac/sagemaker-launcher-convention.md` for the schema and
canonical ARNs.

Example
-------
    uv run python scripts/launch_processing.py \\
        --run-config sagemaker/runs/my-eval.yaml \\
        --config-dir sagemaker/configs/my-eval/
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

from mermaidseg.sagemaker.launcher_config import RunConfig, parse_run_config

ACCOUNT = "554812291621"
REGION = "us-east-1"
EXEC_ROLE = f"arn:aws:iam::{ACCOUNT}:role/dev-sm-execution-role"
STAGING_BUCKET = "dev-datamermaid-sm-data"
ECR_HOST = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com"
KNOWN_SHORT_REPOS = {"mermaid-segmentation-jobs"}
TERMINAL_STATES = {"Completed", "Failed", "Stopped"}
DEFAULT_POLL_INTERVAL_S = 60

log = logging.getLogger("launch_processing")


def expand_image_uri(image: str) -> str:
    if ".dkr.ecr." in image:
        return image
    if ":" not in image:
        raise ValueError(f"Image {image!r} must be `<repo>:<tag>` or a full ECR URI.")
    repo, _ = image.split(":", 1)
    if repo not in KNOWN_SHORT_REPOS:
        raise ValueError(
            f"Unknown short-form repo {repo!r}. Known: {sorted(KNOWN_SHORT_REPOS)}.")
    return f"{ECR_HOST}/{image}"


def make_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def chunk_items(items: list[str], n_workers: int) -> list[list[str]]:
    if n_workers <= 0:
        raise ValueError(f"n_workers must be positive, got {n_workers}")
    chunks: list[list[str]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    return [c for c in chunks if c]


def load_items(csv_path: Path, column: str | None) -> list[str]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if column is None:
            column = reader.fieldnames[0]
        elif column not in reader.fieldnames:
            raise ValueError(f"Column {column!r} not in {csv_path}: {reader.fieldnames}")
        return [row[column] for row in reader if row[column]]


def build_processing_request(
    *, cfg: RunConfig, run_id: str, worker_idx: int, worker_items: list[str] | None,
) -> dict:
    job = cfg.job
    proc = cfg.processing
    container_args = list(proc.container_args)
    if worker_items is not None:
        assert proc.shard is not None
        container_args.extend([proc.shard.per_worker_arg, ",".join(worker_items)])
    return {
        "ProcessingJobName": f"{run_id}-{worker_idx}",
        "RoleArn": EXEC_ROLE,
        "AppSpecification": {
            "ImageUri": expand_image_uri(job.image),
            "ContainerArguments": container_args,
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": job.instance_type,
                "VolumeSizeInGB": job.volume_gb,
            },
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": job.max_runtime_hours * 3600},
        "Environment": {
            "AWS_DEFAULT_REGION": REGION,
            "CONTAINER_ENTRYPOINT_SCRIPT": job.entrypoint,
            **job.env,
        },
        "Tags": (
            [{"Key": "Project", "Value": "mermaid-segmentation"},
             {"Key": "RunId", "Value": run_id},
             {"Key": "WorkerIdx", "Value": str(worker_idx)}]
            + [{"Key": k, "Value": v} for k, v in job.tags.items()]
        ),
    }


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _make_sagemaker_client():
    from botocore.config import Config
    return boto3.client(
        "sagemaker",
        config=Config(region_name=REGION, retries={"mode": "adaptive", "max_attempts": 10}),
    )


def _wait_for_completion(client, job_names, poll_interval_s=DEFAULT_POLL_INTERVAL_S):
    status = {n: "Pending" for n in job_names}
    while True:
        for name in job_names:
            if status[name] in TERMINAL_STATES:
                continue
            resp = client.describe_processing_job(ProcessingJobName=name)
            status[name] = resp["ProcessingJobStatus"]
        unfinished = [n for n, s in status.items() if s not in TERMINAL_STATES]
        if not unfinished:
            return status
        log.info("Polling: %d/%d still running; sleeping %ds",
                 len(unfinished), len(job_names), poll_interval_s)
        time.sleep(poll_interval_s)


def _upload_config_dir(config_dir: Path, run_id: str, s3_client):
    key_prefix = f"runs/{run_id}/config"
    for p in sorted(config_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(config_dir)
        s3_client.upload_file(str(p), STAGING_BUCKET, f"{key_prefix}/{rel}")
    return f"s3://{STAGING_BUCKET}/{key_prefix}/"


def main(argv=None):
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args(argv)

    cfg = parse_run_config(args.run_config.read_text(), kind="processing", strict=True)
    if cfg.processing is None:
        log.error("YAML must include a `processing:` block.")
        sys.exit(2)
    if not args.config_dir.is_dir():
        log.error("Config dir does not exist: %s", args.config_dir)
        sys.exit(2)

    run_id = make_run_id(cfg.job.name_prefix)
    if cfg.processing.shard:
        items = load_items(
            args.config_dir / cfg.processing.shard.items_from,
            cfg.processing.shard.items_column)
        chunks = chunk_items(items, cfg.processing.shard.workers)
        requests = [
            build_processing_request(cfg=cfg, run_id=run_id, worker_idx=i, worker_items=c)
            for i, c in enumerate(chunks)
        ]
    else:
        requests = [build_processing_request(
            cfg=cfg, run_id=run_id, worker_idx=0, worker_items=None)]

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN -- not submitting")
        print("=" * 60)
        print(f"run_id:    {run_id}")
        print(f"workers:   {len(requests)}")
        print(f"image:     {expand_image_uri(cfg.job.image)}")
        for i, r in enumerate(requests):
            print(f"  [{i}] {r['AppSpecification']['ContainerArguments']}")
        return

    s3_client = boto3.client("s3", region_name=REGION)
    _upload_config_dir(args.config_dir.resolve(), run_id, s3_client)
    sm_client = _make_sagemaker_client()
    job_names = []
    for req in requests:
        sm_client.create_processing_job(**req)
        job_names.append(req["ProcessingJobName"])
        log.info("Submitted %s", req["ProcessingJobName"])

    cw_url = (
        f"https://{REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={REGION}#logsV2:log-groups/log-group/"
        f"$252Faws$252Fsagemaker$252FProcessingJobs")
    log.info("CloudWatch: %s", cw_url)

    if args.no_wait:
        return
    statuses = _wait_for_completion(sm_client, job_names)
    failures = [n for n, s in statuses.items() if s != "Completed"]
    if failures:
        log.error("%d job(s) did not complete: %s", len(failures), failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
