#!/usr/bin/env bash
# Local smoke test for the mermaid-segmentation jobs image.
# Builds the image, runs the entrypoint against a tiny config-dir,
# asserts the in-container entrypoint loads without crashing.
#
# Usage: bash docker/jobs/local_smoke.sh [training|processing]
# Default: training.

set -euo pipefail

KIND="${1:-training}"
case "$KIND" in training|processing) ;; *) echo "Usage: $0 [training|processing]" >&2; exit 2 ;; esac

cd "$(dirname "$0")/../.."

IMAGE="mermaid-segmentation-jobs:${KIND}-smoke-local"

echo "[smoke] Building ${IMAGE}..."
docker buildx build --platform linux/amd64 --load \
    -t "${IMAGE}" -f docker/jobs/Dockerfile .

if [ "$KIND" = "training" ]; then
    echo "[smoke] Running training entrypoint (expect 'Loaded run YAML')..."
    docker run --rm \
        -v "$(pwd)/sagemaker/configs/example:/opt/ml/input/data/config:ro" \
        -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
        "${IMAGE}" 2>&1 | head -20 | tee /tmp/seg_smoke.log
    grep -q "Loaded run YAML" /tmp/seg_smoke.log || { echo "[smoke] FAILED"; exit 1; }
else
    echo "[smoke] Running processing entrypoint --help..."
    docker run --rm \
        -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_processing_entrypoint.py \
        "${IMAGE}" --task=eval --help | head -5
fi

echo "[smoke] OK"
