#!/bin/bash
# Final model, trained on the final-run mixture (configs/data_config.yaml):
#
#   cbm_final         300k iterations at 256px, 4k warmup then linear decay to 0,
#                     batch 32, KL 0.5.
#   cbm_final_res768  50k iterations at 768px from cbm_final's model_step300000
#                     weights, 4 GPUs (DDP), batch 8 x accum 4 (aggregated 32),
#                     4k warmup then linear decay to 0, KL 0.5.
#
# cbm_final_res768 starts after cbm_final's job ends successfully (afterok) when
# both are submitted together, or as soon as model_step300000 exists otherwise.
# If 768 runs out of memory, override FINAL_768_BATCH / FINAL_768_ACCUM (product 32).
#
#   slurm/launch_final.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_final.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

# Training reads only the local cache, so fill it here, where AWS credentials and the
# HF hub are reachable, before any job is submitted: the data cache, then the encoder in
# the HF_HOME the jobs use (slurm/common.sh). A dry run submits nothing and skips both.
MODEL_CONFIG="configs/model_config_cbm_dpt_lora_vitl.yaml"
if ((DRY_RUN)); then
    echo "dry run: skipping the cache sync and the encoder download"
else
    : "${MERMAIDSEG_CACHE_DIR:?set MERMAIDSEG_CACHE_DIR in .env}"
    : "${MERMAIDSEG_SCRATCH:?set MERMAIDSEG_SCRATCH in .env}"
    if ! uv run --locked python scripts/sync_data.py --data-config configs/data_config.yaml \
        --cache-dir "$MERMAIDSEG_CACHE_DIR"; then
        echo "cache sync failed; nothing submitted" >&2
        exit 1
    fi
    if ! HF_HOME="$MERMAIDSEG_SCRATCH/hf-cache" \
        uv run --locked python scripts/warm_hf_cache.py --model-config "$MODEL_CONFIG"; then
        echo "encoder download failed; nothing submitted" >&2
        exit 1
    fi
fi

BATCH_768="${FINAL_768_BATCH:-8}"
ACCUM_768="${FINAL_768_ACCUM:-4}"
if ((BATCH_768 * ACCUM_768 != 32)); then
    echo "768: batch $BATCH_768 x accum $ACCUM_768 is not 32" >&2
    exit 2
fi
COMMON=(--data-config configs/data_config.yaml --model-config "$MODEL_CONFIG")

launch_run cbm_final "" -- \
    --group cbm_final -- \
    --training-config configs/training_config_cbm_final.yaml \
    "${COMMON[@]}"
dep=""
[[ -n "$LAST_JOB_ID" ]] && dep="afterok:$LAST_JOB_ID"

launch_branch cbm_final_res768 "$CKPT_ROOT/cbm_final/model_step300000" --init-weights "$dep" \
    --gres=gpu:4 --mem=192G -- \
    --group cbm_final -- \
    --training-config configs/training_config_cbm_final_hires.yaml \
    --resolution 768 \
    "${COMMON[@]}" \
    --batch-size "$BATCH_768" \
    --grad-accum-steps "$ACCUM_768"
launcher_summary
