#!/bin/bash
#SBATCH --job-name=motionexpert_val_qwen3vl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_40GB|A100_80GB"

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_qwen3vl.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/data/user_data/ziyinx/datasets/train_logs/motion_expert/calvin_qwen3vl/best_model.pt}

export WANDB_MODE=${WANDB_MODE:-online}
export QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}

cd "$REPO_ROOT"
python3 -m motion_expert.train \
  --config "$CONFIG_PATH" \
  --resume "$CHECKPOINT_PATH" \
  --validate-only \
  "$@"
