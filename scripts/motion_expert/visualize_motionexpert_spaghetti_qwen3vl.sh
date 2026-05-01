#!/bin/bash
# Visualize MotionExpert validation trajectories (spaghetti GT/pred) from a trained checkpoint.
# This runs validate-only mode, so it will not perform training.
# The configured run now saves both spaghetti plots and per-sample RGB images.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_qwen3vl_viz.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/calvin_qwen3vl/checkpoint_step_18000.pt}

export WANDB_MODE=${WANDB_MODE:-disabled}
export QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}

cd "$REPO_ROOT"
python3 -m motion_expert.train \
  --config "$CONFIG_PATH" \
  --resume "$CHECKPOINT_PATH" \
  --validate-only \
  "$@"

echo "Spaghetti and RGB visualizations saved under the configured train.spaghetti_viz_dir"
