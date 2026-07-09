#!/bin/bash
#SBATCH --job-name=calvin_val_joint
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

# CPU-only job: build the CALVIN joint+EE validation zarr.
# Strided coverage of the WHOLE val set (~target_rows), shuffled, chunk_size=8,
# written straight to a single local val.zarr (no shards). Needs the calvin_env
# sim stack (hydra/pybullet) -> flowervla env.

source /home/ziyinx/miniconda3/etc/profile.d/conda.sh
conda activate flowervla

cd /data/user_data/ziyinx/analogical_manipulation_clean

python data_processing/calvin_val_to_zarr_joint.py \
  --root /data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D \
  --out /data/user_data/ziyinx/datasets/calvin_zarr_joint \
  --target_rows 8000 \
  --chunk_size 8 \
  --seed 42
