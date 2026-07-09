#!/usr/bin/env bash
#
# Online RoboTwin2 evaluation for the joint-angle (qpos) model
# `flower_actor_3d_motion_latent_joint`.
#
# The policy predicts anchor-relative arm-joint deltas (6 per arm) + a binary
# gripper command and is executed through RoboTwin's action_type="qpos" (14-dim
# [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]). Requires
# data_type.qpos=true in the task config (the aloha_vla_*_tcp_analogical configs
# already enable it).
#
# Joint head dims (joint_action_dim=6, joint_proprio_dim=7, gripper_dim=1) come
# from build_eval_args defaults and match the training script
# scripts/robotwin2/train_flower_actor_3d_motion_latent_joint.sh.
#
# Usage:
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin_flower_actor_3d_motion_latent_joint.sh
#   CHECKPOINT=/path/to/last.pth bash .../eval_robotwin_flower_actor_3d_motion_latent_joint.sh clean 0 0 beat_block_hammer True

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_TYPE=flower_actor_3d_motion_latent_joint
export BACKBONE=${BACKBONE:-florence2}
export NUM_HISTORY=${NUM_HISTORY:-1}
export PREDICTION_LEN=${PREDICTION_LEN:-16}
export EMBEDDING_DIM=${EMBEDDING_DIM:-800}
export NUM_SHARED_ATTN_LAYERS=${NUM_SHARED_ATTN_LAYERS:-4}
export ROTATION_FORMAT=${ROTATION_FORMAT:-quat_wxyz}
export POLICY_ACTION_MODE=${POLICY_ACTION_MODE:-relative}
export RELATIVE_ACTION=${RELATIVE_ACTION:-True}
export RELATIVE_ROTATION=${RELATIVE_ROTATION:-True}

# Joint head (must match scripts/robotwin2/train_flower_actor_3d_motion_latent_joint.sh).
export JOINT_ACTION_DIM=${JOINT_ACTION_DIM:-6}
export JOINT_PROPRIO_DIM=${JOINT_PROPRIO_DIM:-7}
export GRIPPER_DIM=${GRIPPER_DIM:-1}
export JOINT_NUM_LAYERS=${JOINT_NUM_LAYERS:-12}
export JOINT_EMBEDDING_DIM=${JOINT_EMBEDDING_DIM:-800}
export JOINT_NUM_ATTN_HEADS=${JOINT_NUM_ATTN_HEADS:-16}
# Trained 3D joint checkpoint uses 2 camera-id slots (not the 3 used by 2D).
export NUM_CAMERA_IDS=${NUM_CAMERA_IDS:-2}

# Joint position control.
export ROBOTWIN_ACTION_TYPE=${ROBOTWIN_ACTION_TYPE:-qpos}

# Motion expert weights are baked into the joint checkpoint; config is still required.
export MOTION_EXPERT_CONFIG=${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/robotwin2_florence2.yaml}
export MOTION_EXPERT_CHECKPOINT=${MOTION_EXPERT_CHECKPOINT:-None}
export MOTION_SOLVER_STEPS=${MOTION_SOLVER_STEPS:-10}
export MOTION_STEPS_TO_RUN=${MOTION_STEPS_TO_RUN:-1}
export MOTION_FINETUNE=${MOTION_FINETUNE:-true}
export MOTION_SCENE_FUSION=${MOTION_SCENE_FUSION:-true}
export MOTION_ACTION_FUSION=${MOTION_ACTION_FUSION:-false}

bash "${SCRIPT_DIR}/eval_robotwin.sh" "$@"
