#!/bin/bash
# Visualize MotionExpert validation trajectories (spaghetti GT/pred) from a trained checkpoint.
# This runs validate-only mode, so it will not perform training.
# The configured run now saves both spaghetti plots and per-sample RGB images.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2_viz.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/calvin_florence2/checkpoint_step_16000.pt}

export WANDB_MODE=${WANDB_MODE:-disabled}
export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}

cd "$REPO_ROOT"
python3 -m motion_expert.train \
  --config "$CONFIG_PATH" \
  --resume "$CHECKPOINT_PATH" \
  --validate-only \
  "$@"

echo "Spaghetti and RGB visualizations saved under the configured train.spaghetti_viz_dir"
