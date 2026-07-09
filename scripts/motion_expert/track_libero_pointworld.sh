#!/bin/bash
# Build a LIBERO RGB-D + SpaTracker GT point-track package for pointworld visualization.
# Run in the traceforge conda environment.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
LIBERO_ROOT=${LIBERO_ROOT:-/data/group_data/katefgroup/datasets/ziyinx/libero}
OUTPUT_DIR=${OUTPUT_DIR:-/data/user_data/ziyinx/vis_outputs/libero_pointworld}
PACKAGE=${PACKAGE:-$OUTPUT_DIR/sample_rgbd_tracks.npz}

cd "$REPO_ROOT"

python3 -m motion_expert.visualize_libero_pointworld \
  --stage track \
  --libero-root "$LIBERO_ROOT" \
  --episode-index "${EPISODE_INDEX:-0}" \
  --demo-index "${DEMO_INDEX:-0}" \
  --camera-name "${CAMERA_NAME:-agentview}" \
  --track-step "${TRACK_STEP:-3}" \
  --query-index "${QUERY_INDEX:-0}" \
  --query-stride-raw "${QUERY_STRIDE_RAW:-45}" \
  --device "${DEVICE:-cuda}" \
  --package "$PACKAGE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
