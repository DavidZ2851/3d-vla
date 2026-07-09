#!/bin/bash
# Run MotionExpert prediction from a LIBERO pointworld package and render pointcloud videos.
# Run in the flowervla conda environment.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
OUTPUT_DIR=${OUTPUT_DIR:-/data/user_data/ziyinx/vis_outputs/libero_pointworld}
PACKAGE=${PACKAGE:-$OUTPUT_DIR/sample_rgbd_tracks.npz}
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/libero_florence2.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$REPO_ROOT/train_logs/motion_expert/libero_florence2/B32_18k.pt}

export WANDB_MODE=${WANDB_MODE:-disabled}
export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}

cd "$REPO_ROOT"

python3 -m motion_expert.visualize_libero_pointworld \
  --stage predict \
  --package "$PACKAGE" \
  --output-dir "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --device "${DEVICE:-cuda}" \
  "$@"
