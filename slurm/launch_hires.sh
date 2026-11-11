#!/bin/bash
# High-resolution fine-tuning: cbm_hires_res<R> for R in 256 384 512 768.
# 25k iterations from the weights of cbm_duration_cd275000 (275k peak + 25k
# cooldown; fresh optimizer), 4k warmup then linear decay to 0, pixel/image
# 0.5/0.5, aggregated batch 32, train split, no taxonomy regularizer.
#
#   256  1 GPU    batch 32 x accum 1
#   384  1 GPU    batch 16 x accum 2
#   512  4 GPUs   batch 32 x accum 1  (DDP, 8 per GPU)
#   768  4 GPUs   batch  8 x accum 4  (DDP, 2 per GPU)
#
# --batch-size is the global batch per micro-step. If a resolution runs out of
# memory, override it with HIRES_BATCH_<R> and HIRES_ACCUM_<R> (product 32).
# A run is submitted once the cooldown checkpoint exists.
#
#   slurm/launch_hires.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_hires.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

SRC="$CKPT_ROOT/cbm_duration_cd275000/model_step25000"
RESOLUTIONS=(256 384 512 768)
GPUS=(1 1 4 4)
BATCH=("${HIRES_BATCH_256:-32}" "${HIRES_BATCH_384:-16}" "${HIRES_BATCH_512:-32}" "${HIRES_BATCH_768:-8}")
ACCUM=("${HIRES_ACCUM_256:-1}" "${HIRES_ACCUM_384:-2}" "${HIRES_ACCUM_512:-1}" "${HIRES_ACCUM_768:-4}")

for i in "${!RESOLUTIONS[@]}"; do
    res=${RESOLUTIONS[$i]}
    if ((BATCH[i] * ACCUM[i] != 32)); then
        echo "res $res: batch ${BATCH[$i]} x accum ${ACCUM[$i]} is not 32" >&2
        exit 2
    fi
    sbatch_opts=()
    if ((GPUS[i] > 1)); then
        sbatch_opts=(--gres="gpu:${GPUS[$i]}" --mem=192G)
    fi
    launch_branch "cbm_hires_res${res}" "$SRC" --init-weights "" ${sbatch_opts[@]+"${sbatch_opts[@]}"} -- \
        --group cbm_hires -- \
        --training-config ../configs/training_config_cbm_hires.yaml \
        --resolution "$res" \
        --per-pixel-loss-weight 0.5 \
        --per-image-loss-weight 0.5 \
        --batch-size "${BATCH[$i]}" \
        --grad-accum-steps "${ACCUM[$i]}"
done
launcher_summary
