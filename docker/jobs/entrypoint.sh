#!/usr/bin/env bash
# SageMaker container entrypoint shim. Dispatches to the script named
# by CONTAINER_ENTRYPOINT_SCRIPT (set by the launcher from the YAML's
# job.entrypoint). The same image serves both TrainingJob and
# ProcessingJob; the env var picks the right entrypoint script.
set -euo pipefail
cd /opt/ml/code

# TrainingJob containers receive `train` as the first arg by convention.
# ProcessingJob containers don't. Drop `train` only if present.
if [ "${1:-}" = "train" ]; then
    shift
fi

SCRIPT="${CONTAINER_ENTRYPOINT_SCRIPT:-scripts/sagemaker_train_entrypoint.py}"
exec python -u "${SCRIPT}" "$@"
