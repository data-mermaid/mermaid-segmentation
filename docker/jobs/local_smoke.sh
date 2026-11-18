#!/usr/bin/env bash
# Local smoke test for the mermaid-segmentation jobs image. Smoke-only: the image is built
# WITHOUT the GIT_SHA build arg (so its entrypoint would refuse a real run) and is never pushed.
# Real images come only from .github/workflows/sagemaker-image.yml.
#
# Checks (1) the trainer's import chain loads in the image, (1b) the image's real ENTRYPOINT
# (tini, then entrypoint.sh, then the trainer entrypoint) starts and refuses a run without a config
# channel, and (2) every sagemaker/runs/*.yaml passes `python -m scripts.sagemaker.launch --check`
# inside the image. No AWS, data or GPU.
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

# 1b. The image's real ENTRYPOINT is what SageMaker runs, so smoke it too. tini must exist at the
#     path the Dockerfile names.
echo "[smoke] Checking tini at its ENTRYPOINT path..."
if ! docker run --rm --entrypoint /usr/bin/tini "${IMAGE}" --version > "$LOG" 2>&1; then
    echo "[smoke] FAILED - /usr/bin/tini is missing or broken in this image:"
    cat "$LOG"
    exit 1
fi
cat "$LOG"

# With no GIT_SHA and no config channel, the real ENTRYPOINT must exit non-zero. The `if` keeps
# `set -e` from aborting here, and the error must be the one train_entrypoint raises first.
echo "[smoke] Checking the real ENTRYPOINT refuses a run with no config channel..."
if OUT="$(docker run --rm "${IMAGE}" train 2>&1)"; then
    echo "[smoke] FAILED - the real ENTRYPOINT started a run with no config channel:"
    echo "${OUT}"
    exit 1
fi
if ! grep -F -q "expected exactly one *.yaml or *.yml" <<< "${OUT}"; then
    echo "[smoke] FAILED - the real ENTRYPOINT failed, but not with the expected error:"
    echo "${OUT}"
    exit 1
fi
echo "[smoke] real ENTRYPOINT refused the run, as expected"

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
