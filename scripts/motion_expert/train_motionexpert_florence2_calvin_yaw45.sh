#!/bin/bash
#SBATCH --job-name=motion_florence2_yaw45
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=katef
#SBATCH --qos=katef_qos
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="A100_40GB"

# Robot-base yaw45 motion expert -- arm D of the frame-consistency experiment.
#
# Identical to train_motionexpert_florence2_calvin.sh except for the config,
# which flips data.camera_frame to calvin_base_yaw45 and points at a frame-specific
# normalization JSON. Same zarr, same hyperparameters; the online R+t at load
# time is the only difference, so any gap between the two runs is attributable
# to the frame.
#
# The resulting checkpoint is what `--motion_expert_frame camera` on
# flower_actor_3d_motion_latent_yaw45 expects (the flag value is a
# frame-agnostic "use my own frame"). Pairing it with the default (`world`)
# silently converts queries back to world before the expert call -- a frame
# mismatch that does not raise.

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2_yaw45.yaml}
NGPUS=${NGPUS:-4}
MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))}
RESUME_CKPT=${RESUME_CKPT:-}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY=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
export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}

cd "$REPO_ROOT"
RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS=(--resume "$RESUME_CKPT")
fi
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train --config "$CONFIG_PATH" "${RESUME_ARGS[@]}" "$@"
