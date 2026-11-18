#!/usr/bin/env bash
# SageMaker container entrypoint shim: runs the training entrypoint from the repo root.
set -euo pipefail
cd /opt/ml/code

# TrainingJob containers receive `train` as the first arg by convention. Drop it.
if [ "${1:-}" = "train" ]; then
    shift
fi

exec python -m scripts.sagemaker.train_entrypoint "$@"
