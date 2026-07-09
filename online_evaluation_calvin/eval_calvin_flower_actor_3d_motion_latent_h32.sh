#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_motion_h32
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"

# Online CALVIN evaluation for flower_actor_3d_motion_latent trained with the
# h32 3D MotionExpert. All policy/eval defaults match the 16-step motion-latent
# setup except the motion expert config/checkpoint and log alias.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
mkdir -p "/data/user_data/${USER}/logs"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

merged_config_file="${MERGED_CONFIG_FILE:-online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml}"
task_config_file="${TASK_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_tasks.yaml}"
ann_config_file="${ANN_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_validation.yaml}"

model_type="${MODEL_TYPE:-flower_actor_3d_motion_latent}"
seed="${SEED:-0}"
checkpoint="${CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/3dfavla_motion_h32/last.pth}"
checkpoint_alias="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${checkpoint}")")}"
base_log_dir="${BASE_LOG_DIR:-${REPO_ROOT}/eval_logs/CALVIN/flower_3dfavla_motion_h32/${checkpoint_alias}/seed${seed}}"
save_video="${SAVE_VIDEO:-false}"

pred_len="${PRED_LEN:-10}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
backbone="${BACKBONE:-florence2}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-24}"
vision_lora_rank="${VISION_LORA_RANK:-0}"
vision_lora_alpha="${VISION_LORA_ALPHA:-32}"
vision_lora_dropout="${VISION_LORA_DROPOUT:-0.05}"
vision_lora_num_layers="${VISION_LORA_NUM_LAYERS:-0}"
vision_lora_target_modules="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}"
llm_lora_rank="${LLM_LORA_RANK:-64}"
llm_lora_alpha="${LLM_LORA_ALPHA:-128}"
llm_lora_dropout="${LLM_LORA_DROPOUT:-0.05}"
llm_lora_num_layers="${LLM_LORA_NUM_LAYERS:-36}"
llm_lora_target_modules="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"
use_multiscale="${USE_MULTISCALE:-false}"
embedding_dim="${EMBEDDING_DIM:-960}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_h32.yaml}"
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/calvin_florence2_h32/checkpoint_step_20000.pt}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
motion_finetune="${MOTION_FINETUNE:-false}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"
motion_action_fusion="${MOTION_ACTION_FUSION:-false}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}" >&2
    exit 1
fi
if [[ ! -f "${motion_expert_config}" ]]; then
    echo "ERROR: motion expert config not found: ${motion_expert_config}" >&2
    exit 1
fi
if [[ "${motion_expert_checkpoint}" != "None" && ! -f "${motion_expert_checkpoint}" ]]; then
    echo "ERROR: motion expert checkpoint not found: ${motion_expert_checkpoint}" >&2
    exit 1
fi

ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "[eval-h32] checkpoint=${checkpoint}"
echo "[eval-h32] base_log_dir=${base_log_dir}"
echo "[eval-h32] model=${model_type} motion=${motion_expert_checkpoint}"

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
    --denoise_model "${denoise_model}" \
    --motion_expert_config "${motion_expert_config}" \
    --motion_expert_checkpoint "${motion_expert_checkpoint}" \
    --motion_solver_steps "${motion_solver_steps}" \
    --motion_steps_to_run "${motion_steps_to_run}" \
    --motion_finetune "${motion_finetune}" \
    --motion_scene_fusion "${motion_scene_fusion}" \
    --motion_action_fusion "${motion_action_fusion}"
