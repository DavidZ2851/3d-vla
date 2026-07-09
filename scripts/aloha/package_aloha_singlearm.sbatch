#!/bin/bash
#SBATCH --job-name=aloha_sa_zarr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#
# Package the processed single-arm ALOHA LeRobot dataset into the zarr that
# training flower_actor_3d consumes, on a CPU node.
#
#   1) aloha_singlearm_to_zarr.py : LeRobot parquet + videos -> train.zarr / val.zarr
#                                   (+ instructions.json, camera_calibration.json,
#                                    xyz_delta_stats.json sidecars)
#
# The converter ALREADY writes row-chunk-8 stores (--zarr-chunk 8), so train.zarr /
# val.zarr are directly trainer-ready and the trainer defaults to those names. A
# separate "rechunked8" pass only adds (a) row shuffling and (b) a compressor swap;
# it rewrites every image array a second time, so it is OFF by default. Turn it on
# (DO_RECHUNK=1) only if you want the rows shuffled across episodes:
#
#   2) rechunk.py (shuffle)       : train.zarr -> train_rechunked8.zarr   [DO_RECHUNK=1]
#   3) rechunk.py (no shuffle)    : val.zarr   -> val_rechunked8.zarr     [DO_RECHUNK=1]
#
# rechunk uses the 3dfa-vla-data-processing copy with --data_type motion, because
# the aloha zarr stores `rel_action` (not `action`); the amc copy assumes `action`.
#
# Submit (fast path, convert only):
#   sbatch scripts/aloha/package_aloha_singlearm.sbatch
# Also build the shuffled rechunked8 stores:
#   DO_RECHUNK=1 sbatch scripts/aloha/package_aloha_singlearm.sbatch
# Override anything via env, e.g.:
#   SRC=/path OUT=/path CHUNK=8 VAL_PER_FOLDER=0 sbatch scripts/aloha/package_aloha_singlearm.sbatch

set -euo pipefail
mkdir -p /data/user_data/${USER}/logs

# ---- config (override via environment) ----------------------------------- #
AMC=${AMC:-/data/user_data/ziyinx/analogical_manipulation_clean}
RECHUNK=${RECHUNK:-/data/user_data/ziyinx/3dfa-vla-data-processing/data_processing/rechunk.py}
CONDA_SH=${CONDA_SH:-/data/user_data/ziyinx/miniconda3_more/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-flowervla}

SRC=${SRC:-/data/user_data/ziyinx/datasets/aloha-singlearm-processed}
OUT=${OUT:-/data/user_data/ziyinx/datasets/zarr_datasets/aloha_singlearm_zarr}
CALIB=${CALIB:-/data/user_data/ziyinx/lerobot/lerobot/scripts/aloha_calibration/calibration_multiview.json}
HORIZON=${HORIZON:-15}
VAL_PER_FOLDER=${VAL_PER_FOLDER:-1}   # held-out val episodes per task (0 = all train)
CHUNK=${CHUNK:-8}                     # row-chunk size (converter writes this directly)
OVERWRITE=${OVERWRITE:-1}             # 1 -> rebuild train.zarr/val.zarr from scratch
DO_RECHUNK=${DO_RECHUNK:-0}           # 1 -> also write shuffled *_rechunked8.zarr (extra copy)

# ---- env ----------------------------------------------------------------- #
# shellcheck disable=SC1090
PYTHON=${PYTHON:-python3}
cd "${AMC}/data_processing"

echo "=========================================================="
echo "SRC        = ${SRC}"
echo "OUT        = ${OUT}"
echo "CALIB      = ${CALIB}"
echo "HORIZON    = ${HORIZON}   VAL_PER_FOLDER = ${VAL_PER_FOLDER}   CHUNK = ${CHUNK}"
echo "python     = $(which ${PYTHON})  (env ${CONDA_ENV})"
echo "=========================================================="

# ---- 1) LeRobot -> train.zarr / val.zarr --------------------------------- #
CONVERT_ARGS=(--root "${SRC}" --out "${OUT}" --calib "${CALIB}"
              --horizon "${HORIZON}" --val-per-folder "${VAL_PER_FOLDER}"
              --zarr-chunk "${CHUNK}")
if [[ "${OVERWRITE}" == "1" ]]; then CONVERT_ARGS+=(--overwrite); fi

echo ">>> [1] aloha_singlearm_to_zarr.py (writes row-chunk-${CHUNK} train.zarr/val.zarr)"
"${PYTHON}" aloha_singlearm_to_zarr.py "${CONVERT_ARGS[@]}"

TRAIN_OUT="${OUT}/train.zarr"
EVAL_OUT="${OUT}/val.zarr"

if [[ "${DO_RECHUNK}" == "1" ]]; then
    # ---- 2) shuffle + rechunk train -------------------------------------- #
    echo ">>> [2] rechunk train.zarr -> train_rechunked${CHUNK}.zarr (shuffled)"
    "${PYTHON}" "${RECHUNK}" \
        --src "${OUT}/train.zarr" \
        --tgt "${OUT}/train_rechunked${CHUNK}.zarr" \
        --chunk_size "${CHUNK}" \
        --shuffle true \
        --data_type motion
    TRAIN_OUT="${OUT}/train_rechunked${CHUNK}.zarr"

    # ---- 3) rechunk val (no shuffle) ------------------------------------- #
    if [[ -d "${OUT}/val.zarr" ]]; then
        echo ">>> [3] rechunk val.zarr -> val_rechunked${CHUNK}.zarr"
        "${PYTHON}" "${RECHUNK}" \
            --src "${OUT}/val.zarr" \
            --tgt "${OUT}/val_rechunked${CHUNK}.zarr" \
            --chunk_size "${CHUNK}" \
            --shuffle false \
            --data_type motion
        EVAL_OUT="${OUT}/val_rechunked${CHUNK}.zarr"
    fi
else
    echo ">>> rechunk skipped (DO_RECHUNK=0); converter already wrote row-chunk-${CHUNK} stores"
fi

echo "=========================================================="
echo "DONE. Trainer inputs:"
echo "  TRAIN_DATA_DIR=${TRAIN_OUT}"
echo "  EVAL_DATA_DIR =${EVAL_OUT}"
echo "  *_INSTRUCTIONS=${OUT}/instructions.json"
echo "=========================================================="
