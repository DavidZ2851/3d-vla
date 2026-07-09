#!/bin/bash
#SBATCH --job-name=rw2_joint_eval_grid
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/%u/logs/%x_%A_%a.err

# Batch launcher: evaluate both joint checkpoints on all 5 training tasks under
# clean and randomized configs, using RoboTwin's normal unseen-seed schedule.
#
# Array layout (20 jobs = 2 models x 5 tasks x 2 splits):
#   model_idx = array_id / 10        (0 -> 3d_joint, 1 -> 2d_joint)
#   task_idx  = (array_id / 2) % 5
#   split_idx = array_id % 2         (0 -> clean, 1 -> randomized)
#
# Usage:
#   sbatch online_evaluation_robotwin/submit_eval_joint_unseen_seeds_array_slurm.sh
#   SEED=1 sbatch --array=0-19 online_evaluation_robotwin/submit_eval_joint_unseen_seeds_array_slurm.sh
#   sbatch --array=0-9 online_evaluation_robotwin/submit_eval_joint_unseen_seeds_array_slurm.sh  # 3d only
#   sbatch --array=10-19 online_evaluation_robotwin/submit_eval_joint_unseen_seeds_array_slurm.sh # 2d only

set -euo pipefail

# NOTE: sbatch runs a spooled *copy* of this file (under /var/spool/slurmd/...),
# so ${BASH_SOURCE[0]} cannot be used to locate the sibling wrapper. Use an
# absolute repo path instead.
REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
SCRIPT_DIR="${REPO_ROOT}/online_evaluation_robotwin"

TASKS=(
  beat_block_hammer
  pick_dual_bottles
  move_can_pot
  handover_mic
  stack_bowls_two
)
SPLITS=(clean randomized)
MODELS=(3d_joint 2d_joint)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this wrapper expects a SLURM array task id." >&2
  exit 1
fi

idx="${SLURM_ARRAY_TASK_ID}"
split_idx=$(( idx % 2 ))
task_idx=$(( (idx / 2) % 5 ))
model_idx=$(( idx / 10 ))

if [[ "${model_idx}" -ge "${#MODELS[@]}" ]]; then
  echo "ERROR: array task ${idx} maps to unsupported model_idx=${model_idx}." >&2
  exit 1
fi

export MODEL="${MODELS[${model_idx}]}"
export TASK="${TASKS[${task_idx}]}"
export SPLIT="${SPLITS[${split_idx}]}"

echo -e "\033[95m[array] id=${idx} -> model=${MODEL} task=${TASK} split=${SPLIT}\033[0m"

exec bash "${SCRIPT_DIR}/submit_eval_joint_unseen_seeds_slurm.sh"
