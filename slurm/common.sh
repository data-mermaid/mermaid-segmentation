#!/bin/bash
# Environment shared by every batch job. Source it; do not run it.
#
# Everything runs from the single shared checkout $SCRATCH/mermaid-segmentation
# with its uv environment. $HOME and every cache are relocated onto scratch so
# nothing touches the real home directory.

export SCRATCH=/data/vision/beery/scratch/sauder
export REPO="$SCRATCH/mermaid-segmentation"
export HOME=$SCRATCH/home
export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
export XDG_DATA_HOME=$HOME/.local/share
export UV_CACHE_DIR=$SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$SCRATCH/uv-python
export UV_LINK_MODE=copy
export HF_HOME=$SCRATCH/hf-cache
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export PATH=$HOME/.local/bin:$PATH

# Node-local TMPDIR. DataLoader workers hand batches over Unix sockets in
# multiprocessing's pymp-* directory under TMPDIR, which must not be on NFS.
LOCAL_TMP="/tmp/${USER:-$(id -un)}-${SLURM_JOB_ID:-$$}"
export TMPDIR="$LOCAL_TMP"
mkdir -p "$TMPDIR" "$REPO/logs" "$REPO/wandb"
trap 'rm -rf "$LOCAL_TMP"' EXIT

# NFS copy of the AFS env file (a batch job has no AFS token). It holds the
# non-secret repo defaults plus WANDB_API_KEY / WANDB_ENTITY.
set -a
# shellcheck disable=SC1090
source "${MERMAIDSEG_ENV_FILE:-$SCRATCH/mermaid.env}"
set +a

# No job needs AWS: data comes from the pre-warmed local cache, and a cache miss
# fails loudly instead of reaching for S3.
unset AWS_PROFILE AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION || true
export MERMAIDSEG_S3_OFFLINE=1

export WANDB_DIR="$REPO/wandb"
export WANDB_PROJECT=${WANDB_PROJECT:-mermaidseg}

cd "$REPO"
uv sync --locked
PY="$REPO/.venv/bin/python"
export PY
