#!/bin/bash
#SBATCH --job-name=mermaid-eval
#SBATCH --account=vision-beery
#SBATCH --qos=vision-beery-free
#SBATCH --requeue
#SBATCH --partition=vision-beery-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.out
#SBATCH --error=/data/vision/beery/scratch/sauder/mermaid-segmentation/logs/slurm-%x-%j.err

# Run the unified validation suite (scripts/evaluate.py) on one checkpoint at one
# prediction resolution.
#
#   sbatch [sbatch options] slurm/eval.sbatch CKPT MODEL_CONFIG RES OUT [EVAL_ARGS...]
#
# CKPT          model_step* / model_epoch* checkpoint (relative to the repo root or absolute)
# MODEL_CONFIG  model config of the checkpoint's encoder
# RES           prediction resolution (256, 384, 512, 768 or 1024)
# OUT           output directory for this (checkpoint, resolution)
# EVAL_ARGS     extra scripts/evaluate.py arguments (e.g. --evals coralnet pacific)
#
# OUT/.done is written only after a clean exit; slurm/launch_evals.sh skips
# outputs that have it. Pass parameters as arguments, not --export=ALL,VAR=...
# (a site prolog cancels such jobs).

set -euo pipefail

usage="usage: sbatch slurm/eval.sbatch CKPT MODEL_CONFIG RES OUT [EVAL_ARGS...]"
CKPT="${1:?$usage}"
MODEL_CONFIG="${2:?$usage}"
RES="${3:?$usage}"
OUT="${4:?$usage}"
shift 4
EVAL_ARGS=("$@")

# 512 at batch 8 is the known ViT-L fit on an L40S; larger resolutions step the
# batch down with the token grid.
case "$RES" in
    256 | 384 | 512) EVAL_BATCH=8 ;;
    768) EVAL_BATCH=2 ;;
    1024) EVAL_BATCH=1 ;;
    *)
        echo "RES must be one of 256 384 512 768 1024, got: $RES" >&2
        exit 2
        ;;
esac

source /data/vision/beery/scratch/sauder/mermaid-segmentation/slurm/common.sh

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID:-} ckpt=$CKPT model=$MODEL_CONFIG res=$RES out=$OUT batch=$EVAL_BATCH args=${EVAL_ARGS[*]:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Every eval rolls CoralNet / Pacific labels up with one benthic hierarchy: the
# first eval fetches it from the MERMAID API and later evals reuse that copy.
HIERARCHY="$REPO/eval_out/benthic_hierarchy.json"
HIERARCHY_ARGS=()
[[ -f "$HIERARCHY" ]] && HIERARCHY_ARGS=(--benthic-hierarchy-json "$HIERARCHY")

# summary.json is rewritten as the suite runs, so only .done marks a finished eval.
mkdir -p "$OUT"
rm -f "$OUT/.done"
srun "$PY" scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --model-config "$MODEL_CONFIG" \
    --resolution "$RES" \
    --output-dir "$OUT" \
    --batch-size "$EVAL_BATCH" \
    --num-workers 16 \
    --amp \
    --strict \
    ${HIERARCHY_ARGS[@]+"${HIERARCHY_ARGS[@]}"} \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}

if [[ ! -f "$HIERARCHY" && -f "$OUT/benthic_hierarchy.json" ]]; then
    cp -n "$OUT/benthic_hierarchy.json" "$HIERARCHY" || true
fi
touch "$OUT/.done"
echo "[batch] eval complete -> $OUT/.done"
