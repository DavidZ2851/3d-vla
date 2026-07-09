#!/bin/bash
set -euo pipefail

# Predict + render the MotionExpert spaghetti trajectory for one frame of a saved
# RoboTwin2 rollout (recorded by online_evaluation_robotwin/rollout_recorder.py).
# Edit these, or override from the command line:
#   FRAME_INDEX=40 ROLLOUT_DIR=vis_outputs/robotwin2_task_rollouts/beat_block_hammer \
#     bash scripts/motion_expert/visualize_robotwin2_rollout_motion_prediction.sh
ROLLOUT_DIR="${ROLLOUT_DIR:-vis_outputs/robotwin2_task_rollouts/beat_block_hammer}"
FRAME_INDEX="${FRAME_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROLLOUT_DIR}/motion_prediction}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/motion_expert/configs/robotwin2_florence2_viz.yml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/robotwin2_florence2/checkpoint_step_20000.pt}"

GRID_SIZE="${GRID_SIZE:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
FPS="${FPS:-4}"
MAX_MOVING_POINTS="${MAX_MOVING_POINTS:-100}"
MOTION_THRESHOLD="${MOTION_THRESHOLD:-0.005}"
PCD_STRIDE="${PCD_STRIDE:-3}"
SCENE_ALPHA="${SCENE_ALPHA:-0.92}"
POINT_SIZE="${POINT_SIZE:-2.0}"
PCD_MAX_POINTS="${PCD_MAX_POINTS:-25000}"

cmd=(
    "${PYTHON_BIN}" -m motion_expert.visualize_robotwin2_rollout_motion_prediction
    --rollout-dir "${ROLLOUT_DIR}"
    --frame-index "${FRAME_INDEX}"
    --output-dir "${OUTPUT_DIR}"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --grid-size "${GRID_SIZE}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --fps "${FPS}"
    --max-moving-points "${MAX_MOVING_POINTS}"
    --motion-threshold "${MOTION_THRESHOLD}"
    --pcd-stride "${PCD_STRIDE}"
    --scene-alpha "${SCENE_ALPHA}"
    --point-size "${POINT_SIZE}"
    --pcd-max-points "${PCD_MAX_POINTS}"
)

if [ "${INSTRUCTION:-}" != "" ]; then
    cmd+=(--instruction "${INSTRUCTION}")
fi
if [ "${SUMMARY:-}" != "" ]; then
    cmd+=(--summary "${SUMMARY}")
fi
if [ "${DEVICE:-}" != "" ]; then
    cmd+=(--device "${DEVICE}")
fi

exec "${cmd[@]}"
