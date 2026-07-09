#!/bin/bash
set -euo pipefail

# Roll out ONE LIBERO task with flower_actor_3d_motion_latent and save the
# agentview RGB/depth video + raw metric depth, so you can later pick a frame
# and run the motion-expert spaghetti prediction on it.
#
# Pick a task by id or by name, e.g.:
#   LIST_TASKS=true bash online_evaluation_libero/rollout_libero_motion_latent_video.sh
#   TASK_ID=3 bash online_evaluation_libero/rollout_libero_motion_latent_video.sh
#   TASK_NAME="open the top drawer" bash online_evaluation_libero/rollout_libero_motion_latent_video.sh
#   BENCHMARK=libero_object TASK_ID=0 bash online_evaluation_libero/rollout_libero_motion_latent_video.sh

BENCHMARK="${BENCHMARK:-libero_10}"
TASK_ORDER_INDEX="${TASK_ORDER_INDEX:-0}"
TASK_ID="${TASK_ID:-9}"
TASK_NAME="${TASK_NAME:-}"            # if set, overrides TASK_ID (substring match on name/language)
INIT_STATE_INDEX="${INIT_STATE_INDEX:-0}"
CAMERA="${CAMERA:-agentview}"
OUT_DIR="${OUT_DIR:-vis_outputs/libero_task_rollouts/${BENCHMARK}}"
FPS="${FPS:-20}"
MAX_STEPS="${MAX_STEPS:-600}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-true}"
SETTLE_STEPS="${SETTLE_STEPS:-30}"
SAVE_DEPTH_NPZ="${SAVE_DEPTH_NPZ:-true}"
LIST_TASKS="${LIST_TASKS:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export QWEN3VL_MODEL_PATH="${QWEN3VL_MODEL_PATH:-/home/ziyinx/Xiaomi-Robotics-0-Pretrain}"

LIBERO_ROOT="${LIBERO_ROOT:-/data/user_data/ziyinx/LIBERO}"
IMG_H="${IMG_H:-224}"
IMG_W="${IMG_W:-224}"
SEED="${SEED:-0}"

# flower_actor_3d_motion_latent checkpoint and motion expert (LIBERO defaults).
CHECKPOINT="${CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/libero/3dfavla_motion/75k.pth}"
MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/libero_florence2.yaml}"
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/libero_florence2/B32_18k.pt}"

# Model args: kept aligned with online_evaluation_libero/eval_libero.sh.
MODEL_TYPE="${MODEL_TYPE:-flower_actor_3d_motion_latent}"
PREDICTION_LEN="${PREDICTION_LEN:-10}"
NUM_HISTORY="${NUM_HISTORY:-1}"
PRE_TOKENIZE="${PRE_TOKENIZE:-true}"
CUSTOM_IMG_SIZE="${CUSTOM_IMG_SIZE:-224}"
BACKBONE="${BACKBONE:-florence2}"
NUM_FINETUNE_VISION_LAYERS="${NUM_FINETUNE_VISION_LAYERS:-0}"
VISION_LORA_RANK="${VISION_LORA_RANK:-0}"
VISION_LORA_ALPHA="${VISION_LORA_ALPHA:-32}"
VISION_LORA_DROPOUT="${VISION_LORA_DROPOUT:-0.05}"
VISION_LORA_NUM_LAYERS="${VISION_LORA_NUM_LAYERS:-0}"
VISION_LORA_TARGET_MODULES="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}"
LLM_LORA_RANK="${LLM_LORA_RANK:-0}"
LLM_LORA_ALPHA="${LLM_LORA_ALPHA:-128}"
LLM_LORA_DROPOUT="${LLM_LORA_DROPOUT:-0.05}"
LLM_LORA_NUM_LAYERS="${LLM_LORA_NUM_LAYERS:-0}"
LLM_LORA_TARGET_MODULES="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
FPS_SUBSAMPLING_FACTOR="${FPS_SUBSAMPLING_FACTOR:-1}"
EMBEDDING_DIM="${EMBEDDING_DIM:-912}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-16}"
NUM_VIS_INSTR_ATTN_LAYERS="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
NUM_SHARED_ATTN_LAYERS="${NUM_SHARED_ATTN_LAYERS:-16}"
USE_MULTISCALE="${USE_MULTISCALE:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
RELATIVE_ACTION="${RELATIVE_ACTION:-true}"
ROTATION_FORMAT="${ROTATION_FORMAT:-axisangle}"
LIBERO_CONTROLLER="${LIBERO_CONTROLLER:-OSC_POSE}"
LIBERO_TCP_TARGET_MODE="${LIBERO_TCP_TARGET_MODE:-absolute}"
LIBERO_TARGET_POS_TOL="${LIBERO_TARGET_POS_TOL:-0.001}"
LIBERO_TARGET_ORI_TOL="${LIBERO_TARGET_ORI_TOL:-0.01}"
LIBERO_TARGET_MAX_STEPS="${LIBERO_TARGET_MAX_STEPS:-250}"
DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-5}"
DENOISE_MODEL="${DENOISE_MODEL:-rectified_flow}"
MOTION_SOLVER_STEPS="${MOTION_SOLVER_STEPS:-10}"
MOTION_STEPS_TO_RUN="${MOTION_STEPS_TO_RUN:-1}"
MOTION_FINETUNE="${MOTION_FINETUNE:-false}"
MOTION_SCENE_FUSION="${MOTION_SCENE_FUSION:-true}"
MOTION_ACTION_FUSION="${MOTION_ACTION_FUSION:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
    "${PYTHON_BIN}" -m online_evaluation_libero.rollout_task_video
    --libero_root "${LIBERO_ROOT}"
    --benchmark "${BENCHMARK}"
    --task_order_index "${TASK_ORDER_INDEX}"
    --seed "${SEED}"
    --img_h "${IMG_H}"
    --img_w "${IMG_W}"
    --checkpoint "${CHECKPOINT}"
    --model_type "${MODEL_TYPE}"
    --prediction_len "${PREDICTION_LEN}"
    --num_history "${NUM_HISTORY}"
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
    --embedding_dim "${EMBEDDING_DIM}"
    --num_attn_heads "${NUM_ATTN_HEADS}"
    --num_vis_instr_attn_layers "${NUM_VIS_INSTR_ATTN_LAYERS}"
    --num_shared_attn_layers "${NUM_SHARED_ATTN_LAYERS}"
    --use_multiscale "${USE_MULTISCALE}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --relative_action "${RELATIVE_ACTION}"
    --rotation_format "${ROTATION_FORMAT}"
    --libero_controller "${LIBERO_CONTROLLER}"
    --libero_tcp_target_mode "${LIBERO_TCP_TARGET_MODE}"
    --libero_target_pos_tol "${LIBERO_TARGET_POS_TOL}"
    --libero_target_ori_tol "${LIBERO_TARGET_ORI_TOL}"
    --libero_target_max_steps "${LIBERO_TARGET_MAX_STEPS}"
    --denoise_timesteps "${DENOISE_TIMESTEPS}"
    --denoise_model "${DENOISE_MODEL}"
    --motion_expert_config "${MOTION_EXPERT_CONFIG}"
    --motion_expert_checkpoint "${MOTION_EXPERT_CHECKPOINT}"
    --motion_solver_steps "${MOTION_SOLVER_STEPS}"
    --motion_steps_to_run "${MOTION_STEPS_TO_RUN}"
    --motion_finetune "${MOTION_FINETUNE}"
    --motion_scene_fusion "${MOTION_SCENE_FUSION}"
    --motion_action_fusion "${MOTION_ACTION_FUSION}"
    --max_steps "${MAX_STEPS}"
)

if [ "${LIST_TASKS}" = "true" ]; then
    cmd+=(--list_tasks)
else
    cmd+=(
        --task_id "${TASK_ID}"
        --init_state_index "${INIT_STATE_INDEX}"
        --camera "${CAMERA}"
        --out_dir "${OUT_DIR}"
        --fps "${FPS}"
        --stop_on_success "${STOP_ON_SUCCESS}"
        --settle_steps "${SETTLE_STEPS}"
        --save_depth_npz "${SAVE_DEPTH_NPZ}"
    )
    if [ "${TASK_NAME}" != "" ]; then
        cmd+=(--task_name "${TASK_NAME}")
    fi
fi

exec "${cmd[@]}"
