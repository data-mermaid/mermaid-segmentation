#!/bin/bash
# Per-pixel / per-image loss-weight sweep: 7 runs cbm_lossw_pix<P>_img<I>.
# 100k iterations, 4k warmup then linear decay to 0, batch 32, train split
# (configs/data_config.yaml), no taxonomy regularizer.
# cbm_lossw_pix0.5_img0.5 is also the ViT-L point of slurm/launch_vitsize.sh.
#
#   slurm/launch_lossw.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_lossw.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

PIX=(1.0 0.9 0.75 0.5 0.25 0.1 0.0)
IMG=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)

for i in "${!PIX[@]}"; do
    launch_run "cbm_lossw_pix${PIX[$i]}_img${IMG[$i]}" "" -- \
        --group cbm_lossw -- \
        --training-config ../configs/training_config_cbm_100k.yaml \
        --per-pixel-loss-weight "${PIX[$i]}" \
        --per-image-loss-weight "${IMG[$i]}"
done
launcher_summary
