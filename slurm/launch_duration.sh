#!/bin/bash
# Training-duration experiment, pixel/image 0.5/0.5, batch 32, train split, no
# taxonomy regularizer.
#
#   cbm_duration_peak       1M iterations: 4k warmup, then constant LR. model_epoch{E}
#                           is written after (E+1)*25k iterations.
#   cbm_duration_cd<STEP>   25k-iteration linear cooldown to 0 from the peak run's
#                           checkpoint at STEP (model optimizer state kept, fresh
#                           schedule, data seed 1004).
#
# The peak run does not fit into one 72h job. Rerun this launcher after each peak job
# ends: it resubmits the peak run, which resumes from its newest checkpoint, and
# submits the cooldowns whose peak checkpoint exists by then. cbm_duration_cd275000
# is the starting point of slurm/launch_hires.sh.
#
#   slurm/launch_duration.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_duration.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

PEAK=cbm_duration_peak
# Must match iterations_per_train_epoch in configs/training_config_cbm_duration_peak.yaml.
ITERS_PER_EPOCH=25000
STEPS=(25000 75000 125000 175000 275000 375000 475000 575000 675000 775000 875000 975000)
LOSS_ARGS=(--per-pixel-loss-weight 0.5 --per-image-loss-weight 0.5)

launch_run "$PEAK" "" -- \
    --group cbm_duration -- \
    --training-config ../configs/training_config_cbm_duration_peak.yaml \
    "${LOSS_ARGS[@]}"

for step in "${STEPS[@]}"; do
    epoch=$((step / ITERS_PER_EPOCH - 1))
    launch_branch "cbm_duration_cd${step}" "$CKPT_ROOT/$PEAK/model_epoch${epoch}" --init-checkpoint "" -- \
        --group cbm_duration -- \
        --training-config ../configs/training_config_cbm_duration_cooldown.yaml \
        "${LOSS_ARGS[@]}" \
        --data-seed 1004
done
launcher_summary
