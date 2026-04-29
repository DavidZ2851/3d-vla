#!/bin/bash
#SBATCH --job-name=motionexpert_train_qwen3vl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_qwen3vl.yaml}
NGPUS=${NGPUS:-4}
MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))}

export WANDB_MODE=${WANDB_MODE:-online}
export QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}

cd "$REPO_ROOT"
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train --config "$CONFIG_PATH" "$@"
