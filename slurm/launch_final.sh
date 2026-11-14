#!/bin/bash
# Final model, trained on the train + val splits (configs/data_config_trainval.yaml):
#
#   cbm_final         300k iterations at 256px, 4k warmup then linear decay to 0,
#                     batch 32, pixel/image 0.5/0.5, KL 0.5.
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

BATCH_768="${FINAL_768_BATCH:-8}"
ACCUM_768="${FINAL_768_ACCUM:-4}"
if ((BATCH_768 * ACCUM_768 != 32)); then
    echo "768: batch $BATCH_768 x accum $ACCUM_768 is not 32" >&2
    exit 2
fi
COMMON=(--data-config ../configs/data_config_trainval.yaml
    --per-pixel-loss-weight 0.5
    --per-image-loss-weight 0.5)

launch_run cbm_final "" -- \
    --group cbm_final -- \
    --training-config ../configs/training_config_cbm_final.yaml \
    "${COMMON[@]}"
dep=""
[[ -n "$LAST_JOB_ID" ]] && dep="afterok:$LAST_JOB_ID"

launch_branch cbm_final_res768 "$CKPT_ROOT/cbm_final/model_step300000" --init-weights "$dep" \
    --gres=gpu:4 --mem=192G -- \
    --group cbm_final -- \
    --training-config ../configs/training_config_cbm_final_hires.yaml \
    --resolution 768 \
    "${COMMON[@]}" \
    --batch-size "$BATCH_768" \
    --grad-accum-steps "$ACCUM_768"
launcher_summary
