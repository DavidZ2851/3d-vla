#!/bin/bash
#SBATCH --job-name=motion_uvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2_uvd.yaml}
NGPUS=${NGPUS:-4}
MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))}
RESUME_CKPT=${RESUME_CKPT:-}

export WANDB_MODE=${WANDB_MODE:-online}
export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "$REPO_ROOT"
RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS=(--resume "$RESUME_CKPT")
fi
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train_uvd --config "$CONFIG_PATH" "${RESUME_ARGS[@]}" "$@"
