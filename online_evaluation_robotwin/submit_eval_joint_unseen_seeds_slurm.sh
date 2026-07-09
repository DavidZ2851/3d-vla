#!/bin/bash
#SBATCH --job-name=2d_bottles_easy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --exclude=babel-m5-32

# Online evaluation of RoboTwin2 joint (qpos) checkpoints on unseen seeds.
# Uses RoboTwin's normal eval seed schedule: eval_policy.py starts from
# 100000 * (1 + SEED) and, with expert_check enabled, skips seeds the scripted
# expert cannot solve. This is NOT train-set seed replay (RW2_EVAL_SEED_LIST is
# explicitly unset).
#
# Supports:
#   flower_actor_3d_motion_latent_joint  (MODEL=3d_joint)
#   flower_actor_2d_joint              (MODEL=2d_joint)
#
# Usage:
#   sbatch online_evaluation_robotwin/submit_eval_joint_unseen_seeds_slurm.sh
#   MODEL=2d_joint SPLIT=clean TASK=handover_mic SEED=0 \
#     sbatch online_evaluation_robotwin/submit_eval_joint_unseen_seeds_slurm.sh
#   sbatch --array=0-19 online_evaluation_robotwin/submit_eval_joint_unseen_seeds_array_slurm.sh
#
# ------------------------- CONFIGURE HERE -----------------------------------
# Model variant: 3d_joint | 2d_joint
MODEL="${MODEL:-2d_joint}"
# Task: one of the 5-task joint training set
#   beat_block_hammer | pick_dual_bottles | move_can_pot | handover_mic | stack_bowls_two
TASK="${TASK:-pick_dual_bottles}"
# Which eval config to use: clean or randomized
SPLIT="${SPLIT:-clean}"
# RoboTwin eval seed index. eval_policy.py starts from 100000 * (1 + SEED).
SEED="${SEED:-0}"
# How many normal eval episodes to evaluate
NUM_EPISODES="${NUM_EPISODES:-50}"
# Checkpoints (override per model or pass CHECKPOINT directly)
CHECKPOINT_3D_JOINT="${CHECKPOINT_3D_JOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/robotwin2/3d_joint_5tasks/200k.pth}"
CHECKPOINT_2D_JOINT="${CHECKPOINT_2D_JOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/robotwin2/2d_joint_5tasks/200k.pth}"
CHECKPOINT="${CHECKPOINT:-}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
EXPERT_CHECK="${EXPERT_CHECK:-True}"
SUMMARY_ROOT="${SUMMARY_ROOT:-}"
# ----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
CONDA_BASE="${CONDA_BASE:-/home/ziyinx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-flowervla}"

JOINT5=(beat_block_hammer pick_dual_bottles move_can_pot handover_mic stack_bowls_two)
ok=0
for t in "${JOINT5[@]}"; do [[ "${t}" == "${TASK}" ]] && ok=1; done
if [[ "${ok}" != "1" ]]; then
  echo "ERROR: TASK='${TASK}' is not one of the 5-task joint training names:" >&2
  printf '  %s\n' "${JOINT5[@]}" >&2
  exit 1
fi

case "${SPLIT}" in
  clean|randomized) ;;
  *) echo "ERROR: SPLIT must be 'clean' or 'randomized', got '${SPLIT}'." >&2; exit 1 ;;
esac

case "${MODEL}" in
  3d_joint|3d|flower_actor_3d_motion_latent_joint)
    MODEL="3d_joint"
    CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_3D_JOINT}}"
    EVAL_SCRIPT="${REPO_ROOT}/online_evaluation_robotwin/eval_robotwin_flower_actor_3d_motion_latent_joint.sh"
    ;;
  2d_joint|2d|flower_actor_2d_joint)
    MODEL="2d_joint"
    CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_2D_JOINT}}"
    EVAL_SCRIPT="${REPO_ROOT}/online_evaluation_robotwin/eval_robotwin_flower_actor_2d_joint.sh"
    ;;
  *)
    echo "ERROR: MODEL must be '3d_joint' or '2d_joint', got '${MODEL}'." >&2
    exit 1
    ;;
esac

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

N="${NUM_EPISODES}"
START_SEED=$(( 100000 * (1 + SEED) ))

echo -e "\033[95m[eval-joint-unseen] model=${MODEL} task=${TASK} split=${SPLIT} seed=${SEED} start_seed=${START_SEED}\033[0m"
echo -e "\033[95m[eval-joint-unseen] running ${N} normal eval episodes with expert_check=${EXPERT_CHECK}\033[0m"
echo -e "\033[95m[eval-joint-unseen] checkpoint: ${CHECKPOINT}\033[0m"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CKPT_ALIAS="$(basename "$(dirname "${CHECKPOINT}")")"
RUN_TAG="${SLURM_JOB_ID:-manual_$$}"
RUN_ID="${RUN_ID:-unseen_seed${SEED}_${MODEL}_${TASK}_${SPLIT}_$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${REPO_ROOT}/eval_logs/robotwin2_joint_unseen/${CKPT_ALIAS}}"
REQUESTED_SUMMARY_DIR="${SUMMARY_DIR:-${SUMMARY_ROOT}/${RUN_ID}}"

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

echo -e "\033[95m[eval-joint-unseen] summary dir: ${SUMMARY_DIR}\033[0m"

unset RW2_EVAL_SEED_LIST
unset RW2_OVERFIT_SEED

CHECKPOINT="${CHECKPOINT}" \
GPU_ID=0 \
TEST_NUM="${N}" \
EXPERT_CHECK="${EXPERT_CHECK}" \
SAVE_VIDEO="${SAVE_VIDEO}" \
SUMMARY_DIR="${SUMMARY_DIR}" \
bash "${EVAL_SCRIPT}" \
  "${SPLIT}" 0 "${SEED}" "${TASK}" "${SAVE_VIDEO}"

echo -e "\033[32m[eval-joint-unseen] done. summary dir: ${SUMMARY_DIR}\033[0m"
