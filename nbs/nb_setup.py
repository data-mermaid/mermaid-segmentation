"""Notebook setup utilities for SageMaker JupyterLab spaces.

Import at the top of every pipeline notebook after setting TF env vars and
MLFLOW_TRACKING_URI. Validates the environment early so failures surface at
cell-run time rather than mid-training.

Typical notebook header (TF vars must come before any mermaidseg import)::

    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "arn:aws:sagemaker:us-east-1:ACCOUNT:mlflow-app/APP-ID"

    from nbs.nb_setup import check_env, check_aws_session, check_mlflow_version
    check_env()
    check_aws_session()
    check_mlflow_version()
"""

import os
import warnings

import boto3
import botocore.exceptions
import mlflow

from mermaidseg.logger import mlflow_connect, resume_run  # noqa: F401 — re-exported


def check_env() -> None:
    """Validate required environment variables using os.getenv().

    Checks MLFLOW_TRACKING_URI, AWS_PROFILE, and HF_TOKEN and prints their
    resolved values. Raises ValueError for unset or broken MLFLOW_TRACKING_URI.

    Raises:
        ValueError: If MLFLOW_TRACKING_URI is unset or contains the
                    ``{region}`` placeholder literal from the notebook template.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is not set. Add to the cell above:\n"
            '    os.environ["MLFLOW_TRACKING_URI"] = "arn:aws:sagemaker:us-east-1:ACCOUNT:mlflow-app/APP-ID"'
        )
    if "{region}" in uri:
        raise ValueError(
            f"MLFLOW_TRACKING_URI contains a literal '{{region}}' placeholder: {uri!r}\nReplace with the actual region (e.g. 'us-east-1')."
        )

    print(f"MLFLOW_TRACKING_URI : {uri}")
    print(f"AWS_PROFILE         : {os.getenv('AWS_PROFILE', '(not set — using default)')}")
    print(f"HF_TOKEN            : {'set' if os.getenv('HF_TOKEN') else '(not set)'}")


def check_aws_session() -> None:
    """Confirm the active IAM identity via STS.

    On SageMaker the execution role (dev-sm-execution-role) auto-refreshes, so this is informational
    — it confirms the correct role is active. After a role policy change, just re-run this cell; no
    space restart needed.
    """
    try:
        identity = boto3.client("sts").get_caller_identity()
        print(f"AWS account : {identity['Account']}")
        print(f"IAM ARN     : {identity['Arn']}")
    except botocore.exceptions.NoCredentialsError:
        print(
            "AWS credentials not found. On SageMaker this should not happen — check the execution role."
        )
    except botocore.exceptions.ClientError as e:
        print(f"AWS session error: {e}")


def check_mlflow_version() -> None:
    """Warn if the installed MLflow version is below 3.x.

    The Serverless MLflow App runs 3.2.0+; mismatched versions break the
    'Models (Experimental)' tab and produce different S3 artifact paths.
    Fix: run ``uv sync --extra training`` (or ``uv sync --all-extras``) from the project directory.
    """
    version = mlflow.__version__
    if tuple(int(x) for x in version.split(".")[:2]) < (3, 0):
        warnings.warn(
            f"MLflow {version} is older than 3.x. The Serverless App runs 3.x; "
            "artifact paths and the 'Models (Experimental)' tab may behave differently.\n"
            "Fix: uv sync --extra training  (or: uv sync --all-extras)",
            stacklevel=2,
        )
    else:
        print(f"mlflow {version} OK")


def check_gpu() -> None:
    """Print available CUDA devices and memory.

    Useful at the top of training notebooks to confirm the expected GPU is visible before starting a
    long run.
    """
    import torch

    if not torch.cuda.is_available():
        print("No GPU — running on CPU")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {props.name}  {total // 1024**3} GB total  {free // 1024**3} GB free")


def print_space_info() -> None:
    """Print SageMaker Studio space metadata from the instance metadata file.

    Prints domain ID, space name, and region. If not running on SageMaker Studio (e.g. local dev),
    prints a notice instead.
    """
    import json
    from pathlib import Path

    meta = Path("/opt/ml/metadata/resource-metadata.json")
    if not meta.exists():
        print("Not running on SageMaker Studio")
        return
    info = json.loads(meta.read_text())
    region = info.get("DomainRegion") or os.getenv("AWS_DEFAULT_REGION", "?")
    print(f"Domain   : {info.get('DomainId')}")
    print(f"Space    : {info.get('SpaceName')}")
    print(f"Region   : {region}")


def reconnect_mlflow(run_id: str | None = None) -> None:
    """Reconnect to MLflow after a kernel restart.

    Re-establishes the tracking URI connection and optionally resumes an
    existing run. Use after re-running the notebook setup cells.

    Args:
        run_id: MLflow run ID to resume. If None, just reconnects and prints
                the tracking URI. Obtain the run_id from the cell output after
                Logger init, or from the MLflow UI.

    Example::

        # After kernel restart, re-run setup cells then:
        from nbs.nb_setup import reconnect_mlflow
        from mermaidseg.logger import resume_run

        reconnect_mlflow()                         # verify connection
        active_run = resume_run("paste-run-id")    # re-attach to run
    """
    duration = mlflow_connect()
    print(f"Connected to MLflow in {duration.total_seconds():.1f}s")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    if run_id:
        resume_run(run_id)
        print(f"Resumed run: {mlflow.active_run().info.run_id}")
