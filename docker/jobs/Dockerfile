# GPU image for mermaid-segmentation SageMaker TrainingJobs. Built from the mermaid-segmentation/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:training-<tag> -f docker/jobs/Dockerfile .
#
# The launcher selects the entrypoint script via CONTAINER_ENTRYPOINT_SCRIPT; the default,
# scripts/sagemaker_train_entrypoint.py, runs nbs/Concept_Bottleneck_Pipeline_vitL_lora_dpt_all.py.
#
# Base: nvidia/cuda 12.6 + cuDNN runtime. Python 3.12 and every package come from uv.lock via
# `uv sync --locked`, the same stack Slurm runs (torch 2.7.1 from the lock bundles its CUDA 12.6
# libraries). No official pytorch/pytorch 2.7.1 image ships Python 3.12, so Python is installed by
# uv rather than taken from the base image.
#
# See docker/jobs/README.md for build/push recipe and tagging conventions.

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Pinned uv. Update together with the uv used to refresh uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

# Secrets (HF_TOKEN, WANDB_API_KEY) are never baked into the image or set in the job Environment:
# the entrypoint fetches them at run time from Secrets Manager (see docker/jobs/README.md).

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# UV_PYTHON matches .python-version.
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PATH="/opt/ml/code/.venv/bin:${PATH}"

# Dependencies first (cached until the lock changes): base deps, the `sagemaker` extra and the
# `research` group (wandb, mlflow, albumentations, ...). `--locked` fails if uv.lock is stale.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra sagemaker --group research --no-install-project

# Project itself (editable so scripts/ stays importable).
COPY mermaidseg ./mermaidseg
COPY scripts ./scripts
COPY configs ./configs
COPY nbs ./nbs
RUN uv sync --locked --no-dev --extra sagemaker --group research

# The image has no .git; the build passes the commit so sidecars record the mermaidseg SHA.
ARG GIT_SHA=unknown
ENV MERMAIDSEG_GIT_SHA=$GIT_SHA

COPY docker/jobs/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    AWS_DEFAULT_REGION=us-east-1

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
