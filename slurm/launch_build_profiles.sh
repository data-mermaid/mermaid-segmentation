#!/bin/bash
# Build the training profiles of the data experiments (CPU jobs):
#
#   datasize  $DS_ROOT/profiles/<exp>_drop<f>/data_config.yaml
#             exp1 drop 0/10/20/50, exp2-exp4 drop 10/20/50
#   sparse    $SPARSE_ROOT/profile/data_config.yaml
#
# A kind whose profiles all exist, or whose build job is queued, is skipped.
# Then run slurm/launch_data_experiments.sh.
#
#   slurm/launch_build_profiles.sh [--dry-run] [datasize] [sparse]

USAGE="usage: slurm/launch_build_profiles.sh [--dry-run] [datasize] [sparse]"
source "$(dirname "${BASH_SOURCE[0]}")/lib_launch.sh"
parse_launcher_args "$@"

DATASIZE_PROFILES=(exp1_drop0 exp1_drop10 exp1_drop20 exp1_drop50
    exp2_drop10 exp2_drop20 exp2_drop50
    exp3_drop10 exp3_drop20 exp3_drop50
    exp4_drop10 exp4_drop20 exp4_drop50)

if wanted datasize; then
    job=mermaid-build-datasize
    missing=()
    for profile in "${DATASIZE_PROFILES[@]}"; do
        [[ -f "$DS_ROOT/profiles/$profile/data_config.yaml" ]] || missing+=("${profile%%_*}")
    done
    experiments=()
    while IFS= read -r exp; do
        [[ -n "$exp" ]] && experiments+=("$exp")
    done < <(printf '%s\n' ${missing[@]+"${missing[@]}"} | sort -u)
    if ((${#experiments[@]} == 0)); then
        echo "done    : datasize profiles"
    elif is_queued "$job"; then
        echo "queued  : $job"
    else
        echo "launch  : $job (${experiments[*]})"
        sbatch_submit --job-name="$job" slurm/build_profiles.sbatch datasize "${experiments[@]}"
    fi
fi

if wanted sparse; then
    job=mermaid-build-sparse
    if [[ -f "$SPARSE_ROOT/profile/data_config.yaml" ]]; then
        echo "done    : sparse profile"
    elif is_queued "$job"; then
        echo "queued  : $job"
    else
        echo "launch  : $job"
        sbatch_submit --job-name="$job" slurm/build_profiles.sbatch sparse
    fi
fi
launcher_summary
