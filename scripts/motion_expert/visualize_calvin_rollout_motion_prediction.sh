#!/bin/bash
set -euo pipefail

# Edit these, or override from the command line:
#   FRAME_INDEX=20 ROLLOUT_DIR=vis_outputs/calvin_task_rollouts/move_slider_left \
#     bash scripts/motion_expert/visualize_calvin_rollout_motion_prediction.sh
ROLLOUT_DIR="${ROLLOUT_DIR:-vis_outputs/calvin_task_rollouts/push_red_block_right}"
FRAME_INDEX="${FRAME_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROLLOUT_DIR}/motion_prediction}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_viz.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt}"

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
    "${PYTHON_BIN}" -m motion_expert.visualize_calvin_rollout_motion_prediction
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
if [ "${DEVICE:-}" != "" ]; then
    cmd+=(--device "${DEVICE}")
fi

exec "${cmd[@]}"
