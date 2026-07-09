#!/bin/bash
# Visualize RoboTwin2 MotionExpert validation trajectories from a Florence-2 checkpoint.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/robotwin2_florence2_viz.yml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$REPO_ROOT/train_logs/motion_expert/robotwin2_florence2/checkpoint_step_20000.pt}
FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594

export WANDB_MODE=${WANDB_MODE:-disabled}
export MOTION_EXPERT_FORCE_SDPA_ATTENTION=${MOTION_EXPERT_FORCE_SDPA_ATTENTION:-0}
export MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN=${MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN:-0}
export MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC=${MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC:-1}
if [ -d "$FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$FLORENCE2_CACHE_SNAPSHOT}
else
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
fi

cd "$REPO_ROOT"
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

python3 -m motion_expert.train \
  --config "$CONFIG_PATH" \
  --resume "$CHECKPOINT_PATH" \
  --validate-only \
  "$@"

echo "Spaghetti and RGB visualizations saved under the configured train.spaghetti_viz_dir"
