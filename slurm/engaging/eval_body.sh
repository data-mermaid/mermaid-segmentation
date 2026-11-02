# Sourced by Engaging eval sbatch scripts after CKPT, RES, OUT, EVAL_ARGS,
# MODEL_CONFIG, and EVAL_BATCH are set. A preempted eval is requeued by Slurm
# and starts this script over; .done is written only after a clean exit.

: "${CKPT:?}"
: "${RES:?}"
: "${OUT:?}"
: "${MODEL_CONFIG:?}"
: "${EVAL_BATCH:?}"
: "${REPO:?}"
EVAL_ARGS="${EVAL_ARGS:---evals all}"

echo "[batch] node=$(hostname) job=${SLURM_JOB_ID:-} ckpt=$CKPT res=$RES out=$OUT model=$MODEL_CONFIG batch=$EVAL_BATCH evals=$EVAL_ARGS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
engaging_require_gpu

cd "$REPO"
uv sync --locked
rm -f "$OUT/.done"
# shellcheck disable=SC2086
srun uv run python scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --model-config "$MODEL_CONFIG" \
    --data-config configs/data_config_dedup.yaml \
    --id2label demo/id2label.json \
    --concept-id2name nbs/concept_id2name.json \
    --output-dir "$OUT" \
    $EVAL_ARGS \
    --batch-size "$EVAL_BATCH" \
    --num-workers 8 \
    --amp

touch "$OUT/.done"
echo "[batch] eval complete -> $OUT/.done"
