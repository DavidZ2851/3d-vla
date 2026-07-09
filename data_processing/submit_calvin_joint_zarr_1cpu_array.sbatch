#!/bin/bash
#SBATCH --job-name=calvin_joint_zarr
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/user_data/%u/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/%u/logs/%x_%A_%a.err

# 10-shard CALVIN joint-action zarr producer (one CPU array task per shard).
# Mirrors the convention in 3dfa-vla-data-processing/data_processing/calvin_traj_2d_uv_10cpu_array.sbatch.
#
# After all 10 shards finish, merge into single train.zarr / val.zarr with:
#   python data_processing/merge_calvin_joint_shards.py --tgt $TGT --num_shards 10
#
# Submit:  sbatch data_processing/submit_calvin_joint_zarr_1cpu_array.sbatch

set -euo pipefail
mkdir -p /data/user_data/${USER}/logs

CODE_ROOT=${CODE_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}
PYTHON=${PYTHON:-python3}
ROOT=${ROOT:-/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D}
TGT=${TGT:-/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr_joint}
NUM_SHARDS=${NUM_SHARDS:-10}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
SPLITS=${SPLITS:-train val}
VAL_MAX_ROWS=${VAL_MAX_ROWS:-5000}

# get_env() in the producer uses hydra.initialize(".") with config paths relative
# to the repo root (online_evaluation_calvin/configs/...), so run from CODE_ROOT.
cd "${CODE_ROOT}"

echo "[INFO] CODE_ROOT=${CODE_ROOT}"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] TGT=${TGT}"
echo "[INFO] shard=${SHARD_ID}/${NUM_SHARDS}"
echo "[INFO] splits=${SPLITS}  val_max_rows=${VAL_MAX_ROWS}"

# Only shard 0 dumps the instruction jsons (avoid 10x duplicate writes / races).
STORE_INSTR=0
if [[ "${SHARD_ID}" -eq 0 ]]; then
  STORE_INSTR=1
fi

"${PYTHON}" data_processing/calvin_to_zarr_joint.py \
  --root "${ROOT}" \
  --tgt "${TGT}" \
  --splits ${SPLITS} \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --val_max_rows "${VAL_MAX_ROWS}" \
  --store_instructions "${STORE_INSTR}"

echo "[INFO] done shard ${SHARD_ID}/${NUM_SHARDS}"
