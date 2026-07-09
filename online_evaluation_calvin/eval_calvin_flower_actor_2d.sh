#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_2dfa
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "/data/user_data/${USER}/logs"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# CALVIN online-eval configs.
merged_config_file="${MERGED_CONFIG_FILE:-online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml}"
task_config_file="${TASK_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_tasks.yaml}"
ann_config_file="${ANN_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_validation.yaml}"

# Eval/checkpoint arguments.
seed="${SEED:-0}"
checkpoint="${CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/flower_actor_2d-CalvinContact10th-C1024-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-job8159084/last.pth}"
checkpoint_alias="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${checkpoint}")")}"
base_log_dir="${BASE_LOG_DIR:-eval_logs/CALVIN/flower_actor_2d/${checkpoint_alias}/seed${seed}}"
save_video="${SAVE_VIDEO:-false}"

# Keep these aligned with scripts/calvin/train_flower_actor_2d.sh.
model_type="${MODEL_TYPE:-flower_actor_2d}"
pred_len="${PRED_LEN:-10}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
backbone="${BACKBONE:-florence2}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-0}"
vision_lora_rank="${VISION_LORA_RANK:-0}"
vision_lora_alpha="${VISION_LORA_ALPHA:-16}"
vision_lora_dropout="${VISION_LORA_DROPOUT:-0.0}"
vision_lora_num_layers="${VISION_LORA_NUM_LAYERS:-0}"
vision_lora_target_modules="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj}"
llm_lora_rank="${LLM_LORA_RANK:-0}"
llm_lora_alpha="${LLM_LORA_ALPHA:-16}"
llm_lora_dropout="${LLM_LORA_DROPOUT:-0.0}"
llm_lora_num_layers="${LLM_LORA_NUM_LAYERS:-0}"
llm_lora_target_modules="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"
use_multiscale="${USE_MULTISCALE:-false}"
embedding_dim="${EMBEDDING_DIM:-1024}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "[eval] repo=${REPO_ROOT}"
echo "[eval] checkpoint=${checkpoint}"
echo "[eval] base_log_dir=${base_log_dir}"
echo "[eval] seed=${seed} ngpus=${ngpus} save_video=${save_video}"
echo "[eval] model=${model_type} backbone=${backbone} C=${embedding_dim} pred_len=${pred_len}"

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
    --vision_lora_rank "${vision_lora_rank}" \
    --vision_lora_alpha "${vision_lora_alpha}" \
    --vision_lora_dropout "${vision_lora_dropout}" \
    --vision_lora_num_layers "${vision_lora_num_layers}" \
    --vision_lora_target_modules "${vision_lora_target_modules}" \
    --llm_lora_rank "${llm_lora_rank}" \
    --llm_lora_alpha "${llm_lora_alpha}" \
    --llm_lora_dropout "${llm_lora_dropout}" \
    --llm_lora_num_layers "${llm_lora_num_layers}" \
    --llm_lora_target_modules "${llm_lora_target_modules}" \
    --fps_subsampling_factor "${fps_subsampling_factor}" \
    --use_multiscale "${use_multiscale}" \
    --embedding_dim "${embedding_dim}" \
    --num_attn_heads "${num_attn_heads}" \
    --num_vis_instr_attn_layers "${num_vis_instr_attn_layers}" \
    --num_shared_attn_layers "${num_shared_attn_layers}" \
    --relative_action "${relative_action}" \
    --rotation_format "${rotation_format}" \
    --denoise_timesteps "${denoise_timesteps}" \
    --denoise_model "${denoise_model}"
