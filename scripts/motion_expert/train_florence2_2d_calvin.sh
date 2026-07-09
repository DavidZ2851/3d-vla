#!/bin/bash
#SBATCH --job-name=motion2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err

# Train the standalone 2D MotionExpert on calvin_2d_motion_zarr (florence2).
set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2_2d.yaml}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=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
# Do NOT pin CUDA_VISIBLE_DEVICES: let Slurm/torchrun expose all allocated GPUs.
# Set it yourself (e.g. CUDA_VISIBLE_DEVICES=0,1) only to restrict to a subset.
export TOKENIZERS_PARALLELISM=false
export MOTION_EXPERT_FORCE_SDPA_ATTENTION=${MOTION_EXPERT_FORCE_SDPA_ATTENTION:-0}
export MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN=${MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN:-0}
export MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC=${MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC:-1}
_FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594
if [ -d "$_FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$_FLORENCE2_CACHE_SNAPSHOT}
else
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
fi

# Auto-detect #GPUs: honor an explicit CUDA_VISIBLE_DEVICES subset, else Slurm
# allocation, else all visible GPUs. Override with NGPUS=N.
if [ -z "${NGPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NGPUS=${SLURM_GPUS_ON_NODE}
  else
    NGPUS=$(nvidia-smi -L 2>/dev/null | grep -c GPU || echo 1)
  fi
fi
[ "${NGPUS:-0}" -ge 1 ] 2>/dev/null || NGPUS=1
MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID:-29555} | tail -c 4))
echo "[motion2d] launching torchrun with NGPUS=$NGPUS (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"

cd "$REPO_ROOT"
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train_2d --config "$CONFIG_PATH" "$@"
