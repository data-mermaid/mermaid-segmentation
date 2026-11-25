#!/bin/bash
# Environment shared by every batch job. Source it; do not run it.
#
# Everything runs from the checkout that holds this file, with its uv environment.
# The env file (default $REPO/.env, override with MERMAIDSEG_ENV_FILE) supplies
# MERMAIDSEG_SCRATCH, MERMAIDSEG_CACHE_DIR, WANDB_API_KEY and the other site values.
# $HOME and every cache are relocated onto scratch so nothing touches the real home
# directory.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO

# A batch job has no AFS token, so MERMAIDSEG_ENV_FILE may point at an NFS copy.
ENV_FILE="${MERMAIDSEG_ENV_FILE:-$REPO/.env}"
if [[ ! -r "$ENV_FILE" ]]; then
    echo "env file $ENV_FILE not readable; see .env.example" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

SCRATCH="${MERMAIDSEG_SCRATCH:?set MERMAIDSEG_SCRATCH in .env}"
export SCRATCH
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

export WANDB_DIR="$REPO/wandb"

cd "$REPO"
uv sync --locked --extra wandb
PY="$REPO/.venv/bin/python"
export PY
