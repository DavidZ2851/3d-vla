#!/bin/bash
#SBATCH --job-name=calvin_motion_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err

# Precompute frame-specific normalization statistics for the motion expert.
#
# Optional: training regenerates missing stats on rank 0, but that burns ~40 min
# of a 4-GPU allocation doing CPU work. Run this first instead.
#
# All frames use the SAME percentile estimator over the SAME full 727k rows, so
# the arms differ only in coordinate frame. Note the zarr chunk cache is kept
# small: both passes touch every chunk exactly once, so caching only inflates
# RSS (a 16 GB cache gets the process killed on a login node).

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
PY=/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python
TRAIN_ZARR=${TRAIN_ZARR:-/data/group_data/katefgroup-ssd/world_model/ziyinx/calvin_motion_zarr/train_rechunked8.zarr}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/datasets}
# Space-separated frames to compute. 'world' is included so every arm shares
# one estimator; it should reproduce the existing NPZ baseline to ~1%.
FRAMES=${FRAMES:-"calvin_static calvin_base_yaw45"}
MEM_LIMIT_GB=${MEM_LIMIT_GB:-2.0}

cd "$REPO_ROOT"

for frame in ${FRAMES}; do
  case "$frame" in
    world)            suffix="_world_from_zarr" ;;
    calvin_static)    suffix="_cam" ;;
    calvin_base_yaw45) suffix="_yaw45" ;;
    *)                suffix="_${frame}" ;;
  esac
  out="${OUT_DIR}/calvin_motion_normalization_stats${suffix}.json"
  if [ -f "$out" ]; then
    echo "[SKIP] $frame -> $out already exists"
    continue
  fi
  echo "[INFO] $(date) computing frame=${frame} -> ${out}"
  "$PY" -m motion_expert.compute_calvin_normalization_stats \
    --zarr "$TRAIN_ZARR" \
    --camera-frame "$frame" \
    --mem-limit-gb "$MEM_LIMIT_GB" \
    --output-json "$out" > /dev/null
  echo "[INFO] $(date) done ${frame}"
  "$PY" -c "
import json,sys
s=json.load(open('$out'))
print('   frame:', s['camera_frame'], ' rows:', s['query_centered_xyz']['count'])
for k in ('query_centered_xyz','traj_delta_xyz'):
    print(f'   {k:20s} scale', [round(v,6) for v in s[k]['scale']])
"
done

echo "[INFO] $(date) all frames done"
