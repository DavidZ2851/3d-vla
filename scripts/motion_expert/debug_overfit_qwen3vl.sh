#!/bin/bash

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_qwen3vl_overfit.yaml}

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}

cd "$REPO_ROOT"
python3 -m motion_expert.train --config "$CONFIG_PATH" "$@"
