#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MODEL_TYPE=flower_actor_3d_motion_latent
export BACKBONE=${BACKBONE:-florence2}
export NUM_HISTORY=${NUM_HISTORY:-1}
export PREDICTION_LEN=${PREDICTION_LEN:-16}
export EMBEDDING_DIM=${EMBEDDING_DIM:-912}
export ROTATION_FORMAT=${ROTATION_FORMAT:-quat_wxyz}
export POLICY_ACTION_MODE=${POLICY_ACTION_MODE:-relative}
export RELATIVE_ACTION=${RELATIVE_ACTION:-True}
export RELATIVE_ROTATION=${RELATIVE_ROTATION:-True}

export MOTION_EXPERT_CONFIG=${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/robotwin2_florence2.yaml}
# No motion-expert checkpoint at eval: the policy checkpoint already carries the
# (finetuned) motion_expert.* weights, and load_pretrained_weights overwrites
# anything pre-loaded here. Only the config is needed, to build the module.
export MOTION_EXPERT_CHECKPOINT=${MOTION_EXPERT_CHECKPOINT:-None}
export MOTION_SOLVER_STEPS=${MOTION_SOLVER_STEPS:-10}
export MOTION_STEPS_TO_RUN=${MOTION_STEPS_TO_RUN:-1}
export MOTION_FINETUNE=${MOTION_FINETUNE:-true}
export MOTION_SCENE_FUSION=${MOTION_SCENE_FUSION:-true}
export MOTION_ACTION_FUSION=${MOTION_ACTION_FUSION:-false}

bash "${SCRIPT_DIR}/eval_robotwin.sh" "$@"
