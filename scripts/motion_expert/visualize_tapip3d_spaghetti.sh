#!/bin/bash
# TAPIP3D-style point-cloud spaghetti visualization on calvin / libero / robotwin2.
# DEFAULT = GROUND-TRUTH only over the real 20x20 query point cloud (no model needed):
# for each benchmark it samples NUM_ROLLOUTS rollouts from the ordered training
# zarr and saves gt/ (query point cloud + GT spaghetti) and video/ (rollout
# videos). Each rollout uses 3 timesteps to represent the process. The cloud is
# the union of the real colored query points from those sampled timesteps, on a
# white background.
#
# Add --pred to ALSO run the trained MotionExpert and render predictions
# (requires the Florence2 checkpoint + flowervla env).
#
# Usage:
#   scripts/motion_expert/visualize_tapip3d_spaghetti.sh                 # all 3, GT-only query cloud
#   DATASETS="calvin" scripts/motion_expert/visualize_tapip3d_spaghetti.sh
#   NUM_ROLLOUTS=3 NO_VIDEO=1 DATASETS="libero" scripts/...sh            # quick smoke test
# Extra flags are forwarded, e.g.:
#   scripts/motion_expert/visualize_tapip3d_spaghetti.sh --pred          # add predictions
#   scripts/motion_expert/visualize_tapip3d_spaghetti.sh --dense         # old interpolated surface cloud
#   scripts/motion_expert/visualize_tapip3d_spaghetti.sh --color-by time

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
cd "$REPO_ROOT"

# flowervla has the Florence2-compatible transformers (4.40.2) + flash_attn.
PYTHON_BIN=${PYTHON_BIN:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python}
DATASETS=${DATASETS:-"calvin libero robotwin2"}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-20}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
SEED=${SEED:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user_data/ziyinx/vis_outputs/motion_tapip3d_white}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-disabled}
export MOTION_EXPERT_FORCE_SDPA_ATTENTION=${MOTION_EXPERT_FORCE_SDPA_ATTENTION:-0}
export MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN=${MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN:-0}
export MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC=${MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC:-1}
FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594
if [ -d "$FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$FLORENCE2_CACHE_SNAPSHOT}
fi

VIDEO_FLAG=""
if [ "${NO_VIDEO:-0}" = "1" ]; then
  VIDEO_FLAG="--no-video"
fi

for ds in $DATASETS; do
  echo "=========================================================="
  echo "[tapip3d-viz] dataset=$ds rollouts=$NUM_ROLLOUTS"
  echo "=========================================================="
  "$PYTHON_BIN" -m motion_expert.visualize_tapip3d_spaghetti \
    --dataset "$ds" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_ROOT/$ds" \
    $VIDEO_FLAG \
    "$@"
done

echo "[tapip3d-viz] finished. Outputs under $OUTPUT_ROOT/<dataset>/{gt,video}/ (add --pred for pred/)"
