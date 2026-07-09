#!/usr/bin/env bash
set -euo pipefail

# Roll out ONE RoboTwin2 task with flower_actor_3d_motion_latent and record the
# static head-camera RGB + metric depth + camera matrices, so you can later pick
# a frame and run the motion-expert spaghetti prediction. This is a thin wrapper
# around the normal eval that enables the opt-in RGBD recorder
# (online_evaluation_robotwin/rollout_recorder.py) and restricts to one task.
#
# Needs the RoboTwin simulator repo (ROBOTWIN_ROOT) + the eval conda env, same as
# eval_robotwin.sh. Example:
#   CHECKPOINT=/path/to/last.pth TASK=beat_block_hammer \
#     bash online_evaluation_robotwin/rollout_robotwin2_motion_latent_video.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TASK="${TASK:-move_can_pot}"
SPLIT="${SPLIT:-randomized}"               # clean|randomized|<task_config>
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-50}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/vis_outputs/robotwin2_task_rollouts/${TASK}}"

# Recorder controls (consumed by rollout_recorder.RoboTwinRolloutRecorder).
export ROBOTWIN_ROLLOUT_DUMP_DIR="${OUT_DIR}"
export ROBOTWIN_ROLLOUT_CAMERA="${CAMERA:-head_camera}"
export ROBOTWIN_ROLLOUT_FPS="${FPS:-20}"
export ROBOTWIN_ROLLOUT_MAX_EPISODES="${ROLLOUT_EPISODES}"

# Only roll out the requested episodes; never save the inner eval video.
export TEST_NUM="${TEST_NUM:-${ROLLOUT_EPISODES}}"

mkdir -p "${OUT_DIR}"
echo "[rw2-rollout] task=${TASK} split=${SPLIT} seed=${SEED} episodes=${ROLLOUT_EPISODES}"
echo "[rw2-rollout] dumping head-camera RGBD to ${OUT_DIR}"

# Positional args to eval_robotwin.sh: <split> <gpu> <seed> <task_selector> <save_video>
exec bash "${SCRIPT_DIR}/eval_robotwin_flower_actor_3d_motion_latent_action16_raw6.sh" \
    "${SPLIT}" "${GPU_ID}" "${SEED}" "${TASK}" False
