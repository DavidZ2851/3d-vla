#!/bin/bash
#SBATCH --job-name=rw2_15task_eval_grid
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/%u/logs/%x_%A_%a.err
#SBATCH --constraint="L40S"

# Batch launcher: evaluate the 15-task RoboTwin2 motion-latent ee checkpoint on
# all 15 tasks under clean and randomized configs, using RoboTwin's normal
# unseen-seed schedule. Each array element is one (task, split) pair and is
# forwarded to submit_eval_50_slurm.sh (which validates against the 15-task set
# and sets ALLOW_NON_ZARR_TASKS=True).
#
# Array layout (30 jobs = 15 tasks x 2 splits):
#   task_idx  = array_id / 2
#   split_idx = array_id % 2   (0 -> clean, 1 -> randomized)
#
# Usage:
#   sbatch online_evaluation_robotwin/submit_eval_50_array_slurm.sh
#   SEED=1 sbatch online_evaluation_robotwin/submit_eval_50_array_slurm.sh
#   CHECKPOINT=/path/<iter>.pth sbatch online_evaluation_robotwin/submit_eval_50_array_slurm.sh
#   sbatch --array=0-1 online_evaluation_robotwin/submit_eval_50_array_slurm.sh  # first task only

set -euo pipefail

# NOTE: sbatch runs a spooled *copy* of this file (under /var/spool/slurmd/...),
# so ${BASH_SOURCE[0]} cannot be used to locate the sibling wrapper. Use an
# absolute repo path instead.
REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
SCRIPT_DIR="${REPO_ROOT}/online_evaluation_robotwin"

# Must match ROBOTWIN2_TASKS15 in submit_eval_50_slurm.sh.
TASKS=(
  beat_block_hammer
  pick_dual_bottles
  move_can_pot
  handover_mic
  stack_bowls_two
  place_phone_stand
  click_alarmclock
  grab_roller
  place_a2b_left
  place_bread_skillet
  scan_object
  handover_block
  open_laptop
  blocks_ranking_rgb
  hanging_mug
)
SPLITS=(clean randomized)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this wrapper expects a SLURM array task id." >&2
  exit 1
fi

idx="${SLURM_ARRAY_TASK_ID}"
split_idx=$(( idx % 2 ))
task_idx=$(( idx / 2 ))

if [[ "${task_idx}" -ge "${#TASKS[@]}" ]]; then
  echo "ERROR: array task ${idx} maps to unsupported task_idx=${task_idx}." >&2
  exit 1
fi

export TASK="${TASKS[${task_idx}]}"
export SPLIT="${SPLITS[${split_idx}]}"

echo -e "\033[95m[array] id=${idx} -> task=${TASK} split=${SPLIT}\033[0m"

exec bash "${SCRIPT_DIR}/submit_eval_50_slurm.sh"
