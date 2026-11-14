#!/bin/bash
# Taxonomy-regularizing losses, each at weight 0.5 from the first step (no ramp;
# the LR schedule handles the start): cbm_tax_kl, cbm_tax_partial, cbm_tax_tree.
# 100k iterations, 4k warmup then linear decay to 0, batch 32, pixel/image 0.5/0.5,
# train split. The unregularized baseline is cbm_lossw_pix0.5_img0.5.
#
#   slurm/launch_taxloss.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_taxloss.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

LOSSES=(kl partial tree)
FLAGS=(--kl-weight --partial-weight --distance-weight)

for i in "${!LOSSES[@]}"; do
    launch_run "cbm_tax_${LOSSES[$i]}" "" -- \
        --group cbm_tax -- \
        --training-config ../configs/training_config_cbm_100k.yaml \
        --per-pixel-loss-weight 0.5 \
        --per-image-loss-weight 0.5 \
        "${FLAGS[$i]}" 0.5
done
launcher_summary
