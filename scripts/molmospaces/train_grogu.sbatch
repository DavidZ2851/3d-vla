#!/bin/bash
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.log
# Few-shot GeomVLA fine-tune from the LIBERO checkpoint on grogu. Submit from the repo root:
#   sbatch --job-name=geomvla_potato --export=ALL,TRAIN=$HOME/geomvla_data/potato_onehouse_random_train.zarr,VAL=$HOME/geomvla_data/potato_onehouse_random_val.zarr,RUN=potato_onehouse_random scripts/molmospaces/train_grogu.sbatch
# Defaults: 2 GPUs x 16 frames (2 chunks of 8 consecutive frames each) = 32 frames/step, 10k steps,
# a checkpoint every 2000 steps. Override with NGPU / B / ITERS / CKPT_FREQ in --export.
cd ${SLURM_SUBMIT_DIR:-$HOME/3d-vla}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
. ${VENV:-$HOME/geomvla_venv}/bin/activate
export NGPU=${NGPU:-$(nvidia-smi -L | wc -l)} B=${B:-16} ITERS=${ITERS:-10000} CKPT_FREQ=${CKPT_FREQ:-2000} VAL_FREQ=${VAL_FREQ:-1000}
export CKPT=${CKPT:-$HOME/geomvla_ckpt/libero/last.pth} LOG_DIR=${LOG_DIR:-$HOME/geomvla_runs}
export MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-12345} % 20000))
bash scripts/molmospaces/finetune_libero_ckpt.sh
