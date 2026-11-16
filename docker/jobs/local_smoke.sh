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
    # Two checks, both offline (no AWS creds / data / GPU needed):
    #
    # 1. The training-code import chain actually loads in *this* image. This is the check
    #    that catches version-skew bugs baked into the image (e.g. a transformers/torch pair
    #    that resolves fine on a dev laptop but not inside the container's platform/Python
    #    version) — the failure mode that a "did the entrypoint start" check can't see,
    #    because it happens deep in mermaidseg.experiment's import chain, independent of any
    #    run YAML. Mirror the heaviest real import path (training entrypoint -> Experiment ->
    #    eval -> MetaModel -> models -> peft/transformers) rather than a shallow one.
    echo "[smoke] Checking the training import chain..."
    docker run --rm --entrypoint python "${IMAGE}" -c "
from mermaidseg.experiment import Experiment
from mermaidseg.model.models import LinearLoRADINOv3, LinearDualLoRADINOv3, LinearDINOv3
print('[smoke] import chain OK')
" || { echo "[smoke] FAILED — training-code import chain is broken in this image"; exit 1; }

    # 2. The in-container entrypoint parses a run YAML and gets as far as invoking
    #    scripts/train.py — a config-plumbing check (e.g. the run-YAML-discovery bug), not a
    #    substitute for #1. Capture the full, untruncated log and check docker run's actual
    #    exit code (not a truncated head(20) grep, which can report OK even when a crash
    #    happens moments later in output it never looked at).
    echo "[smoke] Running training entrypoint end to end..."
    LOG="$(mktemp)"
    if docker run --rm \
        -v "$(pwd)/sagemaker/configs/example:/opt/ml/input/data/config:ro" \
        -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
        "${IMAGE}" > "$LOG" 2>&1; then
        :
    else
        rc=$?
        # A failure past config-parsing is expected here (no AWS creds/data in a bare local
        # run) as long as it got past "Loaded run YAML" first — that's what #1 already covers
        # for the import chain, and full data access needs real credentials this script
        # doesn't have. Fail only if it never even got that far.
        grep -q "Loaded run YAML" "$LOG" || {
            echo "[smoke] FAILED (exit $rc) — entrypoint never parsed the run YAML:"
            cat "$LOG"
            exit 1
        }
    fi
else
    echo "[smoke] Running processing entrypoint --help..."
    docker run --rm \
        -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_processing_entrypoint.py \
        "${IMAGE}" --task=eval --help | head -5
fi

echo "[smoke] OK"
