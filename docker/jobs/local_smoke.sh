#!/usr/bin/env bash
# Local smoke test for the mermaid-segmentation jobs image.
# Builds the image, checks the trainer's import chain, then runs the entrypoint against a
# run YAML with SMOKE_TEST=1 (parse + trainer-command build, no training).
#
# Usage: bash docker/jobs/local_smoke.sh [training]

set -euo pipefail

KIND="${1:-training}"
[ "$KIND" = "training" ] || { echo "Usage: $0 [training]" >&2; exit 2; }

cd "$(dirname "$0")/../.."

IMAGE="mermaid-segmentation-jobs:${KIND}-smoke-local"

echo "[smoke] Building ${IMAGE}..."
docker buildx build --platform linux/amd64 --load \
    --build-arg GIT_SHA="$(git rev-parse HEAD)$(test -z "$(git status --porcelain -- mermaidseg scripts configs pyproject.toml uv.lock)" || echo -dirty)" \
    -t "${IMAGE}" -f docker/jobs/Dockerfile .

# 1. The trainer's import chain loads in *this* image. This catches version skew baked into the
#    image (a transformers/torch pair that resolves on a laptop but not in the container).
echo "[smoke] Checking the trainer import chain..."
docker run --rm --workdir /opt/ml/code --entrypoint python "${IMAGE}" -c "
import mermaidseg.run_setup
from mermaidseg.datasets import build_datasets, setup_local_cache
from mermaidseg.model.meta import MetaModel
from mermaidseg.logger import Logger
print('[smoke] import chain OK')
" || { echo "[smoke] FAILED — trainer import chain is broken in this image"; exit 1; }

# 2. The in-container entrypoint finds the run YAML and builds the trainer command. SMOKE_TEST
#    stops it before the trainer starts, so no AWS creds, data or GPU are needed.
echo "[smoke] Running training entrypoint (SMOKE_TEST)..."
CONFIG_DIR="$(mktemp -d)"
cp sagemaker/runs/cbm_vitl_example.yaml "${CONFIG_DIR}/"
LOG="$(mktemp)"
if ! docker run --rm \
    -v "${CONFIG_DIR}:/opt/ml/input/data/config:ro" \
    -e SMOKE_TEST=1 \
    -e CONTAINER_ENTRYPOINT_SCRIPT=scripts/sagemaker_train_entrypoint.py \
    "${IMAGE}" > "$LOG" 2>&1; then
    echo "[smoke] FAILED — entrypoint exited non-zero:"
    cat "$LOG"
    exit 1
fi
grep -q "Smoke test OK" "$LOG" || { echo "[smoke] FAILED — no smoke marker in log:"; cat "$LOG"; exit 1; }

echo "[smoke] OK"
