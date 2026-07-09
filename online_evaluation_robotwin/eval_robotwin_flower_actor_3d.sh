#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_TYPE=flower_actor_3d
export BACKBONE=${BACKBONE:-florence2}
export NUM_HISTORY=${NUM_HISTORY:-2}
export PREDICTION_LEN=${PREDICTION_LEN:-16}
export EMBEDDING_DIM=${EMBEDDING_DIM:-1024}
export ROTATION_FORMAT=${ROTATION_FORMAT:-quat_wxyz}
export POLICY_ACTION_MODE=${POLICY_ACTION_MODE:-relative}
export RELATIVE_ACTION=${RELATIVE_ACTION:-True}
export RELATIVE_ROTATION=${RELATIVE_ROTATION:-True}

bash "${SCRIPT_DIR}/eval_robotwin.sh" "$@"
