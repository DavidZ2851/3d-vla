#!/bin/bash
# Raw RGBD -> SpaTracker2 -> TAPIP3D-style white-background 3D spaghetti.
#
# Defaults:
#   - benchmarks: calvin, libero-10, robotwin2 pick_dual_bottles
#   - first selected rollout per benchmark
#   - start/mid/end query windows per rollout
#   - separate spaghetti PNG for each start/mid/end window
#   - 15 SpaTracker2 frames per window at raw stride 3
#   - TAPIP3D scene point cloud density: 256x192 depth unprojection
#
# Quick loader check:
#   DRY_RUN=1 scripts/motion_expert/visualize_raw_spatracker_tapip3d.sh
#
# Run one benchmark:
#   BENCHMARKS="libero" NUM_ROLLOUTS=1 scripts/motion_expert/visualize_raw_spatracker_tapip3d.sh

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python}
BENCHMARKS=${BENCHMARKS:-"calvin libero robotwin2"}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user_data/ziyinx/vis_outputs/raw_spatracker_tapip3d_first_windows}
DEVICE=${DEVICE:-cuda}

DRY_RUN_FLAG=""
if [ "${DRY_RUN:-0}" = "1" ]; then
  DRY_RUN_FLAG="--dry-run"
fi

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" -m motion_expert.visualize_raw_spatracker_tapip3d \
  --benchmarks $BENCHMARKS \
  --num-rollouts "$NUM_ROLLOUTS" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  $DRY_RUN_FLAG \
  "$@"

echo "[raw-spatracker-viz] outputs under $OUTPUT_ROOT"
