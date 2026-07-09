#!/bin/bash
set -euo pipefail

# Edit these, or override them from the command line:
#   TASK=open_drawer bash online_evaluation_calvin/rollout_calvin_motion_latent_video.sh
#   LIST_TASKS=true bash online_evaluation_calvin/rollout_calvin_motion_latent_video.sh
TASK="${TASK:-move_slider_left}"
TASK_OCCURRENCE="${TASK_OCCURRENCE:-0}"
CAMERA="${CAMERA:-static}"              # static or gripper
OUT_DIR="${OUT_DIR:-vis_outputs/calvin_task_rollouts/${TASK}}"
RUN_PREFIX="${RUN_PREFIX:-true}"        # needed when TASK appears after setup subtasks
RECORD_PREFIX="${RECORD_PREFIX:-false}" # true saves prefix setup frames before target frames
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-true}"
LIST_TASKS="${LIST_TASKS:-false}"
SAVE_DEPTH_NPZ="${SAVE_DEPTH_NPZ:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# pretrained VLM backbone path
export QWEN3VL_MODEL_PATH="${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}"

# CALVIN env configs
MERGED_CONFIG_FILE="${MERGED_CONFIG_FILE:-online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml}"
TASK_CONFIG_FILE="${TASK_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_tasks.yaml}"
ANN_CONFIG_FILE="${ANN_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_validation.yaml}"

# flower_actor_3d_motion_latent checkpoint and motion expert
CHECKPOINT="${CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/calvin/3dfavla_motion/last.pth}"
MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG:-/data/user_data/ziyinx/analogical_manipulation_clean/motion_expert/configs/calvin_florence2.yaml}"
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt}"

# Model args: kept aligned with online_evaluation_calvin/eval_calvin.sh.
SEED="${SEED:-0}"
PRED_LEN="${PRED_LEN:-10}"
PRE_TOKENIZE="${PRE_TOKENIZE:-true}"
CUSTOM_IMG_SIZE="${CUSTOM_IMG_SIZE:-224}"
BACKBONE="${BACKBONE:-florence2}"
NUM_FINETUNE_VISION_LAYERS="${NUM_FINETUNE_VISION_LAYERS:-24}"
VISION_LORA_RANK="${VISION_LORA_RANK:-0}"
VISION_LORA_ALPHA="${VISION_LORA_ALPHA:-32}"
VISION_LORA_DROPOUT="${VISION_LORA_DROPOUT:-0.05}"
VISION_LORA_NUM_LAYERS="${VISION_LORA_NUM_LAYERS:-0}"
VISION_LORA_TARGET_MODULES="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}"
LLM_LORA_RANK="${LLM_LORA_RANK:-64}"
LLM_LORA_ALPHA="${LLM_LORA_ALPHA:-128}"
LLM_LORA_DROPOUT="${LLM_LORA_DROPOUT:-0.05}"
LLM_LORA_NUM_LAYERS="${LLM_LORA_NUM_LAYERS:-36}"
LLM_LORA_TARGET_MODULES="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
FPS_SUBSAMPLING_FACTOR="${FPS_SUBSAMPLING_FACTOR:-1}"
USE_MULTISCALE="${USE_MULTISCALE:-false}"
EMBEDDING_DIM="${EMBEDDING_DIM:-960}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-16}"
NUM_VIS_INSTR_ATTN_LAYERS="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
NUM_SHARED_ATTN_LAYERS="${NUM_SHARED_ATTN_LAYERS:-16}"
RELATIVE_ACTION="${RELATIVE_ACTION:-true}"
ROTATION_FORMAT="${ROTATION_FORMAT:-euler}"
DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-5}"
DENOISE_MODEL="${DENOISE_MODEL:-rectified_flow}"
MOTION_SOLVER_STEPS="${MOTION_SOLVER_STEPS:-10}"
MOTION_STEPS_TO_RUN="${MOTION_STEPS_TO_RUN:-1}"
MOTION_FINETUNE="${MOTION_FINETUNE:-false}"
MOTION_SCENE_FUSION="${MOTION_SCENE_FUSION:-true}"
MOTION_ACTION_FUSION="${MOTION_ACTION_FUSION:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
    "${PYTHON_BIN}" -m online_evaluation_calvin.rollout_task_video
    --merged_config_file "${MERGED_CONFIG_FILE}"
    --task_config_file "${TASK_CONFIG_FILE}"
    --ann_config_file "${ANN_CONFIG_FILE}"
    --seed "${SEED}"
    --checkpoint "${CHECKPOINT}"
    --model_type flower_actor_3d_motion_latent
    --pred_len "${PRED_LEN}"
    --pre_tokenize "${PRE_TOKENIZE}"
    --custom_img_size "${CUSTOM_IMG_SIZE}"
    --backbone "${BACKBONE}"
    --num_finetune_vision_layers "${NUM_FINETUNE_VISION_LAYERS}"
    --vision_lora_rank "${VISION_LORA_RANK}"
    --vision_lora_alpha "${VISION_LORA_ALPHA}"
    --vision_lora_dropout "${VISION_LORA_DROPOUT}"
    --vision_lora_num_layers "${VISION_LORA_NUM_LAYERS}"
    --vision_lora_target_modules "${VISION_LORA_TARGET_MODULES}"
    --llm_lora_rank "${LLM_LORA_RANK}"
    --llm_lora_alpha "${LLM_LORA_ALPHA}"
    --llm_lora_dropout "${LLM_LORA_DROPOUT}"
    --llm_lora_num_layers "${LLM_LORA_NUM_LAYERS}"
    --llm_lora_target_modules "${LLM_LORA_TARGET_MODULES}"
    --fps_subsampling_factor "${FPS_SUBSAMPLING_FACTOR}"
    --use_multiscale "${USE_MULTISCALE}"
    --embedding_dim "${EMBEDDING_DIM}"
    --num_attn_heads "${NUM_ATTN_HEADS}"
    --num_vis_instr_attn_layers "${NUM_VIS_INSTR_ATTN_LAYERS}"
    --num_shared_attn_layers "${NUM_SHARED_ATTN_LAYERS}"
    --relative_action "${RELATIVE_ACTION}"
    --rotation_format "${ROTATION_FORMAT}"
    --denoise_timesteps "${DENOISE_TIMESTEPS}"
    --denoise_model "${DENOISE_MODEL}"
    --motion_expert_config "${MOTION_EXPERT_CONFIG}"
    --motion_expert_checkpoint "${MOTION_EXPERT_CHECKPOINT}"
    --motion_solver_steps "${MOTION_SOLVER_STEPS}"
    --motion_steps_to_run "${MOTION_STEPS_TO_RUN}"
    --motion_finetune "${MOTION_FINETUNE}"
    --motion_scene_fusion "${MOTION_SCENE_FUSION}"
    --motion_action_fusion "${MOTION_ACTION_FUSION}"
)

if [ "${LIST_TASKS}" = "true" ]; then
    cmd+=(--list_tasks)
else
    cmd+=(
        --task "${TASK}"
        --task_occurrence "${TASK_OCCURRENCE}"
        --camera "${CAMERA}"
        --out_dir "${OUT_DIR}"
        --run_prefix "${RUN_PREFIX}"
        --record_prefix "${RECORD_PREFIX}"
        --stop_on_success "${STOP_ON_SUCCESS}"
        --save_depth_npz "${SAVE_DEPTH_NPZ}"
    )
fi

exec "${cmd[@]}"
