# GPU image for mermaid-segmentation SageMaker TrainingJobs (the backup substrate to Slurm; both
# run scripts/train.py). Built by .github/workflows/sagemaker-image.yml from a committed tree;
# local builds (docker/jobs/local_smoke.sh) are smoke-only and never pushed.
#
# Base: nvidia/cuda 12.6 + cuDNN runtime. Python 3.12 and every package come from uv.lock via
# `uv sync --locked --extra mlflow`, the same torch stack Slurm runs (torch from the lock
# bundles its CUDA 12.6 libraries; Slurm installs the `wandb` extra instead), so
# Python is installed by uv rather than taken from the base image.
#
# See docker/jobs/README.md.

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d827f6436b3cb7468fd8acbdc8c1d41261614e579ae49afe6141da51133
FROM ${BASE_IMAGE}

# Pinned uv. The digest is bumped by Dependabot; update the tag together with the uv used to
# refresh uv.lock.
# The `version` in the setup-uv steps of .github/workflows/*.yml moves with this tag.
COPY --from=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563 /uv /uvx /usr/local/bin/

# Secrets (HF_TOKEN) are never baked into the image or set in the job Environment:
# the entrypoint fetches them at run time from Secrets Manager (see docker/jobs/README.md).

# No git: the image has no .git and the commit arrives as MERMAIDSEG_GIT_SHA.
# libglib2.0-0 is kept as a conservative runtime dependency of the OpenCV wheel.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# UV_PYTHON matches .python-version.
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PATH="/opt/ml/code/.venv/bin:${PATH}"

# Dependencies first (cached until the lock changes). `--locked` fails if uv.lock is stale.
COPY pyproject.toml uv.lock .python-version ./
# The lock lists the coralnet-scraper, segdata-etl and segdata-spec workspace members; uv reads their
# pyprojects to validate the lock. The members themselves are not installed.
COPY packages/coralnet-scraper/pyproject.toml ./packages/coralnet-scraper/pyproject.toml
COPY packages/segdata-etl/pyproject.toml ./packages/segdata-etl/pyproject.toml
COPY packages/segdata-spec/pyproject.toml ./packages/segdata-spec/pyproject.toml
# segdata-spec comes in through the `training` extra (the `training` default group, which `--no-dev`
# keeps, and the `mlflow` extra both include it). `--no-install-workspace` skips it and the root
# project, so this layer holds only third-party packages and stays cached until the lock changes.
RUN uv sync --locked --no-dev --extra mlflow --no-install-workspace

# Project itself (editable so scripts/ stays importable) and its segdata-spec workspace dependency.
COPY packages/segdata-spec/src ./packages/segdata-spec/src
COPY mermaidseg ./mermaidseg
COPY scripts ./scripts
COPY configs ./configs
RUN uv sync --locked --no-dev --extra mlflow

# The image has no .git; CI passes the commit so lineage records the mermaidseg SHA. No default:
# a build without GIT_SHA yields an image whose entrypoint refuses to start a run.
ARG GIT_SHA
ENV MERMAIDSEG_GIT_SHA=$GIT_SHA

COPY docker/jobs/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    AWS_DEFAULT_REGION=us-east-1

# tini is PID 1 so StopTrainingJob's SIGTERM reaches the trainer (a PID-1 process with no handler
# ignores SIGTERM).
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/ml/code/entrypoint.sh"]
