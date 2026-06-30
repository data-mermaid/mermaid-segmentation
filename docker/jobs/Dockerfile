# GPU image for mermaid-segmentation SageMaker jobs (TrainingJob +
# ProcessingJob). Built from the mermaid-segmentation/ directory:
#   docker buildx build --platform linux/amd64 \
#       -t <ECR_URI>:training-<tag> -f docker/jobs/Dockerfile .
#
# Same image runs both Training and Processing — the launcher selects
# which entrypoint script to invoke via CONTAINER_ENTRYPOINT_SCRIPT.
#
# Base: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime (Python 3.11, torch 2.4.1)
# Matches pyproject.toml requires-python>=3.11, torch>=2.4,<2.8.
#
# See docker/jobs/CLAUDE.md for build/push recipe and tagging conventions.

ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Project itself (editable so scripts/ stays importable).
COPY pyproject.toml README.md ./
COPY mermaidseg ./mermaidseg
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY docker/jobs/entrypoint.sh /opt/ml/code/entrypoint.sh
RUN chmod +x /opt/ml/code/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_DEFAULT_REGION=us-east-1

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
