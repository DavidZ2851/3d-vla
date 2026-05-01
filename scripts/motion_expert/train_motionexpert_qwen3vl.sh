#!/bin/bash
#SBATCH --job-name=motion_qwen3vl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_qwen3vl.yaml}
NGPUS=${NGPUS:-4}
MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))}
RESUME_CKPT=${RESUME_CKPT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/calvin_qwen3vl/checkpoint_step_12000.pt}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=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
export QWEN3VL_MODEL_PATH=${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}

cd "$REPO_ROOT"
RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS=(--resume "$RESUME_CKPT")
fi
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train --config "$CONFIG_PATH" "${RESUME_ARGS[@]}" "$@"
