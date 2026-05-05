#!/bin/bash
#SBATCH --job-name=motionexpert_val_florence2
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
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$REPO_ROOT/train_logs/motion_expert/calvin_florence2/best_model.pt}

export WANDB_MODE=${WANDB_MODE:-online}
export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}

cd "$REPO_ROOT"
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi
python3 -m motion_expert.train \
  --config "$CONFIG_PATH" \
  --resume "$CHECKPOINT_PATH" \
  --validate-only \
  "$@"
