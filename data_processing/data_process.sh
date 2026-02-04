#!/bin/bash
#SBATCH --job-name=run
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

#python data_processing/calvin_to_zarr.py
#python data_processing/rechunk.py --src /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/train.zarr --tgt /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/train_rechunked8.zarr --chunk_size 8 --shuffle False
python data_processing/rechunk.py --src /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val.zarr --tgt /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr --chunk_size 8 --shuffle False
# rm -rf /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/train.zarr
# rm -rf /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val.zarr