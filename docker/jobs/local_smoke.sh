#!/usr/bin/env bash
# Local smoke test for the mermaid-segmentation jobs image. Smoke-only: the image is built
# WITHOUT the GIT_SHA build arg (so its entrypoint would refuse a real run) and is never pushed.
# Real images come only from .github/workflows/sagemaker-image.yml.
#
# Checks (1) the trainer's import chain loads in the image and (2) every sagemaker/runs/*.yaml
# passes `python -m scripts.sagemaker.launch --check` inside the image. No AWS, data or GPU.
#
# Usage: bash docker/jobs/local_smoke.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE="mermaid-segmentation-jobs:training-smoke-local"

echo "[smoke] Building ${IMAGE}..."
docker buildx build --platform linux/amd64 --load -t "${IMAGE}" -f docker/jobs/Dockerfile .

# 1. The trainer's import chain loads in *this* image. This catches version skew baked into the
#    image (a transformers/torch pair that resolves on a laptop but not in the container).
echo "[smoke] Checking the trainer import chain..."
LOG="$(mktemp)"
if ! docker run --rm --workdir /opt/ml/code --entrypoint python "${IMAGE}" -c "
import mermaidseg.training
import mermaidseg.run_setup
import mermaidseg.inference
import scripts.sagemaker.train_entrypoint
print('[smoke] import chain OK')
" > "$LOG" 2>&1; then
    echo "[smoke] FAILED - trainer import chain is broken in this image:"
    cat "$LOG"
    exit 1
fi
cat "$LOG"

# 2. Each committed run config validates offline with the launcher inside the image.
for run in sagemaker/runs/*.yaml; do
    name="$(basename "$run")"
    extra=()
    if [ "$name" = "cbm_final_res768.yaml" ]; then
        extra=(--init-from-job smoke-placeholder-job)
    fi
    echo "[smoke] launch --check ${name}..."
    if ! docker run --rm --workdir /opt/ml/code --entrypoint python \
        -v "$PWD/sagemaker/runs:/opt/ml/runs:ro" \
        "${IMAGE}" -m scripts.sagemaker.launch --check \
        --run-config "/opt/ml/runs/${name}" ${extra[@]+"${extra[@]}"} > "$LOG" 2>&1; then
        echo "[smoke] FAILED - launch --check rejected ${name}:"
        cat "$LOG"
        exit 1
    fi
done

echo "[smoke] OK"
