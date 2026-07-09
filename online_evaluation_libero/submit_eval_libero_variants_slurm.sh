#!/bin/bash
#SBATCH --job-name=eval_libero_variants
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%A_%a.err
#SBATCH --constraint="L40S|A100_80GB"
# Submit examples:
#   sbatch --array=0-5 online_evaluation_libero/submit_eval_libero_variants_slurm.sh
#   VARIANT=plus TASK_CHUNK=0 sbatch --array=0-23 online_evaluation_libero/submit_eval_libero_variants_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"

variant="${VARIANT:-pro}"
preset="${PRESET:-3dfavla}"
seed="${SEED:-0}"

declare -a presets=(3dfavla 3dfavla_motion 3dfavla_motion_subset)
declare -a variants=(pro plus)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_id="$SLURM_ARRAY_TASK_ID"
    if [[ -n "${TASK_CHUNK:-}" ]]; then
        preset="${presets[$((task_id % ${#presets[@]}))]}"
        variant="${variants[$((task_id / ${#presets[@]}))]}"
    else
        preset="${presets[$task_id]}"
        variant="pro"
    fi
fi

export PRESET="$preset"
export SEED="$seed"

if [[ "$variant" == "plus" ]]; then
    if [[ -n "${TASK_CHUNK:-}" ]]; then
        chunk_size="${TASK_CHUNK_SIZE:-100}"
        suite="${SUITES:-libero_spatial}"
        total_tasks="${TASK_TOTAL:-2402}"
        chunk_start=$((TASK_CHUNK * chunk_size))
        chunk_end=$((chunk_start + chunk_size))
        if [[ "$chunk_end" -gt "$total_tasks" ]]; then
            chunk_end="$total_tasks"
        fi
        export SUITES="$suite"
        export TASK_START="$chunk_start"
        export TASK_END="$chunk_end"
    fi
    bash "$SCRIPT_DIR/eval_libero_plus.sh"
else
    bash "$SCRIPT_DIR/eval_libero_pro.sh"
fi
