#!/bin/bash
#SBATCH --job-name=merge_calvin_joint
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

# CPU-only job: merge the per-shard calvin_zarr_joint into a single train.zarr,
# rechunked to chunk_size=8 along dim 0 in the same pass (no separate rechunk.py).
# Only the train split exists (no val shards).

cd /data/user_data/ziyinx/analogical_manipulation_clean

/usr/bin/python3 data_processing/merge_calvin_joint_shards.py \
  --tgt /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr_joint \
  --out /data/user_data/ziyinx/datasets/calvin_zarr_joint \
  --splits train \
  --num_shards 10 \
  --chunk_size 8
