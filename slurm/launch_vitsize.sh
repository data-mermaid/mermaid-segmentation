#!/bin/bash
# Encoder-size experiment: cbm_vitsize_{vits,vitb,vithplus} with the DINOv3
# ViT-S, ViT-B and ViT-H+ LoRA encoders. Same recipe as the loss-weight sweep:
# 100k iterations, 4k warmup then linear decay to 0, batch 32, pixel/image
# 0.5/0.5, train split, no taxonomy regularizer. The ViT-L point is
# cbm_lossw_pix0.5_img0.5 from slurm/launch_lossw.sh.
#
#   slurm/launch_vitsize.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_vitsize.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

for size in vits vitb vithplus; do
    launch_run "cbm_vitsize_${size}" "" -- \
        --group cbm_vitsize -- \
        --training-config ../configs/training_config_cbm_100k.yaml \
        --model-config "../configs/model_config_cbm_dpt_lora_${size}.yaml" \
        --per-pixel-loss-weight 0.5 \
        --per-image-loss-weight 0.5
done
launcher_summary
