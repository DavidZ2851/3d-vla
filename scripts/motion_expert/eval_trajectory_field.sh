#!/bin/bash
#SBATCH --job-name=motion_field_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/%u/logs/%x_%A_%a.err
#SBATCH --constraint="L40S|A100_40GB|A100_80GB"
#SBATCH --array=0-3
#
# Scores the decoded MotionExpert trajectory field against the SpatialTrackerV2
# labels held in each benchmark's validation zarr.
#
#   0  calvin_h15    H=15 expert, CALVIN val
#   1  calvin_h32    H=32 expert, SAME CALVIN val, truncated to the first 15 steps
#   2  libero        H=15 expert, LIBERO val
#   3  robotwin2_5t  H=15 expert, RoboTwin2.0 val restricted to the 5 policy tasks
#
# Tasks 0 and 1 deliberately share one val store: the H=32 zarr was consumed by
# move_calvin_h32_val_to_train_rechunked.py, and scoring both experts on the same
# rows is the apples-to-apples comparison anyway.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
PYTHON_BIN=${PYTHON_BIN:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python}
FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594
OUT_DIR=${OUT_DIR:-$REPO_ROOT/eval_logs/motion_field}

CALVIN_VAL=/data/group_data/katefgroup-ssd/world_model/ziyinx/calvin_motion_zarr/val_rechunked8.zarr
CALVIN_VAL_INSTR=$REPO_ROOT/instructions/calvin/val_motion_instructions_full.json
LIBERO_VAL=/data/group_data/katefgroup/VLA/zarr_datasets/libero_motion_zarr/val_rechunked8.zarr
RW2_ROOT=/data/group_data/katefgroup/VLA/zarr_datasets/robotwin2_motion_zarr_50tasks
RW2_TASKS=beat_block_hammer,pick_dual_bottles,move_can_pot,handover_mic,stack_bowls_two

HORIZON=${HORIZON:-15}
STEPS=${STEPS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-0}
MAX_ROWS_ARG=()
if [ -n "${MAX_ROWS:-}" ]; then
  MAX_ROWS_ARG=(--max-rows "$MAX_ROWS")
fi

export TOKENIZERS_PARALLELISM=false
export MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC=1
if [ -d "$FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$FLORENCE2_CACHE_SNAPSHOT}
else
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
fi

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

IDX=${SLURM_ARRAY_TASK_ID:-${1:-0}}
EXTRA=()
case "$IDX" in
  0)
    TAG=calvin_h15
    CONFIG=$REPO_ROOT/motion_expert/configs/calvin_florence2.yaml
    CKPT=$REPO_ROOT/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt
    EXTRA=(--val-zarr "$CALVIN_VAL" --val-instructions "$CALVIN_VAL_INSTR")
    ;;
  1)
    TAG=calvin_h32
    CONFIG=$REPO_ROOT/motion_expert/configs/calvin_florence2_h32.yaml
    CKPT=$REPO_ROOT/train_logs/motion_expert/baseline_calvin_florence2_h32/checkpoint_step_20000.pt
    EXTRA=(--val-zarr "$CALVIN_VAL" --val-instructions "$CALVIN_VAL_INSTR")
    ;;
  2)
    TAG=libero
    CONFIG=$REPO_ROOT/motion_expert/configs/libero_florence2.yaml
    CKPT=$REPO_ROOT/train_logs/motion_expert/libero_florence2/B32_18k.pt
    EXTRA=(--val-zarr "$LIBERO_VAL")
    ;;
  3)
    TAG=robotwin2_5tasks
    CONFIG=$REPO_ROOT/motion_expert/configs/robotwin2_florence2.yaml
    CKPT=$REPO_ROOT/train_logs/motion_expert/robotwin2_florence2_5tasks/checkpoint_step_20000.pt
    EXTRA=(
      --val-zarr "$RW2_ROOT/val_rechunked8.zarr"
      --split-manifest "$RW2_ROOT/split_manifest.json"
      --tasks "$RW2_TASKS"
    )
    ;;
  *)
    echo "Unknown array index $IDX" >&2
    exit 1
    ;;
esac

echo "[motion-field-eval] tag=$TAG config=$CONFIG ckpt=$CKPT horizon=$HORIZON steps=$STEPS"

"$PYTHON_BIN" -m motion_expert.eval_trajectory_field \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --horizon "$HORIZON" \
  --num-inference-steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 6 \
  --seed "$SEED" \
  --tag "$TAG" \
  --output "$OUT_DIR/${TAG}_h${HORIZON}_s${STEPS}_seed${SEED}.json" \
  --dump-npz "$OUT_DIR/${TAG}_h${HORIZON}_s${STEPS}_seed${SEED}.npz" \
  "${EXTRA[@]}" "${MAX_ROWS_ARG[@]}"
