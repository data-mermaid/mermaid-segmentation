#!/bin/bash
# Data experiments, one run per built profile (slurm/launch_build_profiles.sh):
#
#   cbm_dsize_exp1_drop{0,10,20,50}      exp1  CoralNet train images, IID drop
#   cbm_dsize_exp{2,3,4}_drop{10,20,50}  exp2  whole CoralNet sources
#                                        exp3  every meta-dataset train image, IID drop
#                                        exp4  CoralNet sources and other datasets as units
#   cbm_sparse                           Coralscapes V2 and UCSD mosaics sparsified to
#                                        25 points per image
#
# Same recipe as the loss-weight sweep: 100k iterations, 4k warmup then linear decay
# to 0, batch 32, pixel/image 0.5/0.5, no taxonomy regularizer. A run whose profile
# is not built yet is reported and skipped.
#
#   slurm/launch_data_experiments.sh [--dry-run] [RUN...]

USAGE="usage: slurm/launch_data_experiments.sh [--dry-run] [RUN...]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

PROFILES=(exp1_drop0 exp1_drop10 exp1_drop20 exp1_drop50
    exp2_drop10 exp2_drop20 exp2_drop50
    exp3_drop10 exp3_drop20 exp3_drop50
    exp4_drop10 exp4_drop20 exp4_drop50)

RUNS=()
CONFIGS=()
for profile in "${PROFILES[@]}"; do
    RUNS+=("cbm_dsize_${profile}")
    CONFIGS+=("$DS_ROOT/profiles/$profile/data_config.yaml")
done
RUNS+=(cbm_sparse)
CONFIGS+=("$SPARSE_ROOT/profile/data_config.yaml")

for i in "${!RUNS[@]}"; do
    run=${RUNS[$i]}
    config=${CONFIGS[$i]}
    wanted "$run" || continue
    if [[ ! -f "$config" ]] && ! run_finished "$run" && ! is_queued "$run"; then
        echo "waiting : $run (profile not built: $config)"
        continue
    fi
    launch_run "$run" "" -- \
        --group "${run%_drop*}" -- \
        --training-config ../configs/training_config_cbm_100k.yaml \
        --data-config "$config" \
        --per-pixel-loss-weight 0.5 \
        --per-image-loss-weight 0.5
done
launcher_summary
