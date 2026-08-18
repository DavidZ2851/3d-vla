#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_2dlamp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB|A6000|L40|6000Ada"

# Online CALVIN ABC->D evaluation of the LaMP-style fully-2D VLA + UVD motion
# expert (`flower_actor_2d_motion_uvd`), pinned to the trained baseline at
# train_logs/calvin/baseline_2d_lamp/last.pth (iter 50000, EMA weights).
#
# Hyperparameters below are the ones the checkpoint was trained with
# (scripts/calvin/train_flower_actor_2d_motion_uvd.sh): C=896, 16 shared attn
# layers, 2 camera ids, rectified flow with 5 denoise steps, UVD expert with
# hidden_dim=256. They were cross-checked against the checkpoint's tensor shapes;
# changing them will make load_pretrained_weights fail loudly.
#
# Preemption: this runs on the `preempt` partition with --requeue. Per-sequence
# results are appended to ${base_log_dir}/result.txt and already-evaluated
# sequence indices are skipped on restart, so a requeued job resumes where it
# stopped. Re-submit until the final line reports "Load 1000/1000 episodes".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
mkdir -p "/data/user_data/${USER}/logs"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

merged_config_file="${MERGED_CONFIG_FILE:-online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml}"
task_config_file="${TASK_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_tasks.yaml}"
ann_config_file="${ANN_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_validation.yaml}"

model_type="${MODEL_TYPE:-flower_actor_2d_motion_uvd}"
seed="${SEED:-121}"
checkpoint="${CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/baseline_2d_lamp/last.pth}"
checkpoint_alias="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${checkpoint}")")}"
base_log_dir="${BASE_LOG_DIR:-eval_logs/CALVIN/${model_type}/${checkpoint_alias}/seed${seed}}"
save_video="${SAVE_VIDEO:-false}"

pred_len="${PRED_LEN:-10}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
backbone="${BACKBONE:-florence2}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-0}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"
use_multiscale="${USE_MULTISCALE:-false}"
embedding_dim="${EMBEDDING_DIM:-896}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

# The policy checkpoint already carries the (finetuned) motion_expert.* weights;
# this config/checkpoint pair only builds the module before those are loaded.
motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_uvd.yaml}"
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/baseline_calvin_florence2_uvd/checkpoint_step_14000.pt}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
motion_finetune="${MOTION_FINETUNE:-false}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"

# 1000 sequences are split round-robin across ranks. To use more GPUs:
#   sbatch --gres=gpu:4 --export=ALL,NGPUS=4 online_evaluation_calvin/eval_calvin_baseline_2d_lamp.sh
ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "[eval] checkpoint=${checkpoint}"
echo "[eval] base_log_dir=${base_log_dir}"
echo "[eval] model=${model_type} ngpus=${ngpus} motion=${motion_expert_checkpoint}"

torchrun --nproc_per_node "${ngpus}" --master_port "${master_port}" \
    online_evaluation_calvin/evaluate_policy.py \
    --merged_config_file "${merged_config_file}" \
    --task_config_file "${task_config_file}" \
    --ann_config_file "${ann_config_file}" \
    --seed "${seed}" \
    --checkpoint "${checkpoint}" \
    --base_log_dir "${base_log_dir}" \
    --save_video "${save_video}" \
    --model_type "${model_type}" \
    --pred_len "${pred_len}" \
    --pre_tokenize "${pre_tokenize}" \
    --custom_img_size "${custom_img_size}" \
    --backbone "${backbone}" \
    --num_finetune_vision_layers "${num_finetune_vision_layers}" \
    --fps_subsampling_factor "${fps_subsampling_factor}" \
    --use_multiscale "${use_multiscale}" \
    --embedding_dim "${embedding_dim}" \
    --num_attn_heads "${num_attn_heads}" \
    --num_vis_instr_attn_layers "${num_vis_instr_attn_layers}" \
    --num_shared_attn_layers "${num_shared_attn_layers}" \
    --relative_action "${relative_action}" \
    --rotation_format "${rotation_format}" \
    --denoise_timesteps "${denoise_timesteps}" \
    --denoise_model "${denoise_model}" \
    --motion_expert_config "${motion_expert_config}" \
    --motion_expert_checkpoint "${motion_expert_checkpoint}" \
    --motion_solver_steps "${motion_solver_steps}" \
    --motion_steps_to_run "${motion_steps_to_run}" \
    --motion_finetune "${motion_finetune}" \
    --motion_scene_fusion "${motion_scene_fusion}"
