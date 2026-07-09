#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"

# Online evaluation of a RoboTwin2 6-task (action16/raw6) motion-latent
# checkpoint using RoboTwin's normal evaluation seed schedule. The SEED value is
# passed to script/eval_policy.py, which starts from 100000 * (1 + SEED) and,
# with expert_check enabled, skips seeds the scripted expert cannot solve.
#
# Usage:
#   sbatch online_evaluation_robotwin/submit_eval_trainset50_slurm.sh
#   # override anything from the CLI, e.g.:
#   TASK=handover_mic SPLIT=randomized SEED=1 CHECKPOINT=/path/<iter>.pth \
#     sbatch online_evaluation_robotwin/submit_eval_trainset50_slurm.sh
#
# ------------------------- CONFIGURE HERE -----------------------------------
# Task: any of the 15-task RoboTwin2 set (5 original + 10 appended). See
# ROBOTWIN2_TASKS15 below.
TASK="${TASK:-beat_block_hammer}"
# Which eval config to use: clean or randomized
SPLIT="${SPLIT:-clean}"
# RoboTwin eval seed index. eval_policy.py starts from 100000 * (1 + SEED).
SEED="${SEED:-0}"
# How many normal eval episodes to evaluate
NUM_EPISODES="${NUM_EPISODES:-50}"
# Checkpoint to evaluate (default: the 15-task motion-latent ee model)
CHECKPOINT="${CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/robotwin2/3dfavla_motion_15tasks/420k.pth}"
# Motion expert checkpoint (15-task motion expert paired with the 15-task policy)
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/robotwin2_florence2_15tasks/checkpoint_step_40000.pt}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
EXPERT_CHECK="${EXPERT_CHECK:-True}"
# Root for this wrapper's summaries/logs. Each submission gets a fresh child
# directory below this root; existing result directories are never reused.
# Leave empty to use the absolute default computed below (an absolute path is
# required: eval_robotwin.sh cd's into ROBOTWIN_ROOT before writing logs, so a
# relative SUMMARY_ROOT would resolve to a non-existent dir and tee would fail).
SUMMARY_ROOT="${SUMMARY_ROOT:-}"
# ----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
CONDA_BASE="${CONDA_BASE:-/home/ziyinx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-flowervla}"

# --- validate task is one of the 15-task RoboTwin2 set ----------------------
# Original 5 + 10 appended tasks (matches the robotwin2 15-task train split).
ROBOTWIN2_TASKS15=(
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
ok=0
for t in "${ROBOTWIN2_TASKS15[@]}"; do [[ "${t}" == "${TASK}" ]] && ok=1; done
if [[ "${ok}" != "1" ]]; then
  echo "ERROR: TASK='${TASK}' is not one of the 15 RoboTwin2 task names:" >&2
  printf '  %s\n' "${ROBOTWIN2_TASKS15[@]}" >&2
  exit 1
fi

# --- validate split ---------------------------------------------------------
case "${SPLIT}" in
  clean|randomized) ;;
  *) echo "ERROR: SPLIT must be 'clean' or 'randomized', got '${SPLIT}'." >&2; exit 1 ;;
esac

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

N="${NUM_EPISODES}"
START_SEED=$(( 100000 * (1 + SEED) ))

echo -e "\033[95m[eval-normal] task=${TASK} split=${SPLIT} seed=${SEED} start_seed=${START_SEED}\033[0m"
echo -e "\033[95m[eval-normal] running ${N} normal eval episodes with expert_check=${EXPERT_CHECK}\033[0m"
echo -e "\033[95m[eval-normal] checkpoint: ${CHECKPOINT}\033[0m"
echo -e "\033[95m[eval-normal] motion expert checkpoint: ${MOTION_EXPERT_CHECKPOINT}\033[0m"

# --- env: activate conda ----------------------------------------------------
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CKPT_ALIAS="$(basename "$(dirname "${CHECKPOINT}")")"
RUN_TAG="${SLURM_JOB_ID:-manual_$$}"
RUN_ID="${RUN_ID:-normal_seed${SEED}_${TASK}_${SPLIT}_$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${REPO_ROOT}/eval_logs/robotwin2_normal/${CKPT_ALIAS}}"
REQUESTED_SUMMARY_DIR="${SUMMARY_DIR:-${SUMMARY_ROOT}/${RUN_ID}}"

# Reserve a fresh output directory. eval_robotwin.sh writes summary files with
# shell redirection, so reusing a directory would overwrite previous results.
SUMMARY_DIR="${REQUESTED_SUMMARY_DIR}"
SUMMARY_PARENT="$(dirname "${REQUESTED_SUMMARY_DIR}")"
mkdir -p "${SUMMARY_PARENT}"
if mkdir "${SUMMARY_DIR}" 2>/dev/null; then
  :
elif [[ -e "${SUMMARY_DIR}" ]]; then
  SUMMARY_DIR=""
else
  echo "ERROR: could not create summary dir: ${REQUESTED_SUMMARY_DIR}" >&2
  exit 1
fi

if [[ -z "${SUMMARY_DIR}" ]]; then
  for suffix in $(seq -w 1 999); do
    CANDIDATE="${REQUESTED_SUMMARY_DIR}_${suffix}"
    if mkdir "${CANDIDATE}" 2>/dev/null; then
      SUMMARY_DIR="${CANDIDATE}"
      break
    fi
  done
  if [[ -z "${SUMMARY_DIR}" ]]; then
    echo "ERROR: could not find an unused summary dir under ${REQUESTED_SUMMARY_DIR}_NNN" >&2
    exit 1
  fi
fi

echo -e "\033[95m[eval-normal] summary dir: ${SUMMARY_DIR}\033[0m"

# Ensure this wrapper cannot accidentally inherit train-set replay / overfit
# seed overrides from the caller's environment.
unset RW2_EVAL_SEED_LIST
unset RW2_OVERFIT_SEED

# ALLOW_NON_ZARR_TASKS=True: several of the 15 tasks are outside eval_robotwin.sh's
# 9-task action15_stride2 zarr whitelist; we validate against ROBOTWIN2_TASKS15 above.
CHECKPOINT="${CHECKPOINT}" \
GPU_ID=0 \
TEST_NUM="${N}" \
EXPERT_CHECK="${EXPERT_CHECK}" \
SAVE_VIDEO="${SAVE_VIDEO}" \
SUMMARY_DIR="${SUMMARY_DIR}" \
ALLOW_NON_ZARR_TASKS=True \
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT}" \
bash "${REPO_ROOT}/online_evaluation_robotwin/eval_robotwin_flower_actor_3d_motion_latent.sh" \
  "${SPLIT}" 0 "${SEED}" "${TASK}" "${SAVE_VIDEO}"

echo -e "\033[32m[eval-normal] done. summary dir: ${SUMMARY_DIR}\033[0m"
