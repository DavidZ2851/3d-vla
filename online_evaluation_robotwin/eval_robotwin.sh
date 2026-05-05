#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh easy 0 0 grab_roller
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh hard 0 0 grab_roller,handover_mic
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh both 0 0 grab_roller
# Positional args:
#   1: split/task config shortcut: easy|clean|hard|randomized|both (default: easy)
#   2: gpu id (default: 0)
#   3: seed (default: 0)
#   4: task selector: comma-separated task names or "all" (default: grab_roller)
#
# Useful env overrides:
#   TASK_CONFIG=aloha_vla_clean50_tcp_analogical
#   POLICY_ACTION_MODE=absolute|relative
#   ROBOTWIN_ACTION_TYPE=ee|delta_ee
#   TEST_NUM=50
#   EVAL_VIDEO_LOG=True|False

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALOGICAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/user_data/ziyinx/RoboTwin}"

SPLIT="${1:-${SPLIT:-easy}}"
GPU_ID="${2:-${GPU_ID:-0}}"
SEED="${3:-${SEED:-0}}"
TASK_SELECTOR="${4:-${TASK_SELECTOR:-grab_roller}}"

POLICY_NAME="analogical_manipulation"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is required. Example: CHECKPOINT=/path/to/last.pth bash $0 easy 0 0 grab_roller" >&2
  exit 1
fi

CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${CHECKPOINT}")")}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"

EASY_CONFIG="${EASY_CONFIG:-aloha_vla_clean50_tcp_analogical}"
HARD_CONFIG="${HARD_CONFIG:-aloha_vla_randomized500_tcp_analogical}"

case "${SPLIT}" in
  easy|clean)
    TASK_CONFIGS=("${TASK_CONFIG:-${EASY_CONFIG}}")
    ;;
  hard|randomized)
    TASK_CONFIGS=("${TASK_CONFIG:-${HARD_CONFIG}}")
    ;;
  both)
    if [[ -n "${TASK_CONFIG:-}" ]]; then
      TASK_CONFIGS=("${TASK_CONFIG}")
    else
      TASK_CONFIGS=("${EASY_CONFIG}" "${HARD_CONFIG}")
    fi
    ;;
  *)
    TASK_CONFIGS=("${TASK_CONFIG:-${SPLIT}}")
    ;;
esac

ALL_TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

if [[ "${TASK_SELECTOR}" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "${TASK_SELECTOR}"
fi

MODEL_TYPE="${MODEL_TYPE:-flower_actor_3d}"
BACKBONE="${BACKBONE:-clip}"
NUM_HISTORY="${NUM_HISTORY:-1}"
PREDICTION_LEN="${PREDICTION_LEN:-10}"
CUSTOM_IMG_SIZE="${CUSTOM_IMG_SIZE:-224}"
ROTATION_FORMAT="${ROTATION_FORMAT:-quat_wxyz}"
POLICY_ACTION_MODE="${POLICY_ACTION_MODE:-absolute}"
ROBOTWIN_ACTION_TYPE="${ROBOTWIN_ACTION_TYPE:-ee}"

PRE_TOKENIZE="${PRE_TOKENIZE:-True}"
if [[ -z "${RELATIVE_ACTION:-}" ]]; then
  if [[ "${POLICY_ACTION_MODE}" == "relative" ]]; then
    RELATIVE_ACTION="True"
  else
    RELATIVE_ACTION="False"
  fi
fi
NUM_FINETUNE_VISION_LAYERS="${NUM_FINETUNE_VISION_LAYERS:-0}"
FPS_SUBSAMPLING_FACTOR="${FPS_SUBSAMPLING_FACTOR:-5}"
EMBEDDING_DIM="${EMBEDDING_DIM:-144}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-9}"
NUM_VIS_INSTR_ATTN_LAYERS="${NUM_VIS_INSTR_ATTN_LAYERS:-2}"
NUM_SHARED_ATTN_LAYERS="${NUM_SHARED_ATTN_LAYERS:-16}"
DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-10}"
DENOISE_MODEL="${DENOISE_MODEL:-rectified_flow}"

MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG:-None}"
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-None}"
MOTION_SOLVER_STEPS="${MOTION_SOLVER_STEPS:-10}"
MOTION_STEPS_TO_RUN="${MOTION_STEPS_TO_RUN:-1}"
MOTION_FINETUNE="${MOTION_FINETUNE:-False}"
MOTION_SCENE_FUSION="${MOTION_SCENE_FUSION:-True}"
MOTION_ACTION_FUSION="${MOTION_ACTION_FUSION:-False}"

VISION_LORA_RANK="${VISION_LORA_RANK:-0}"
VISION_LORA_ALPHA="${VISION_LORA_ALPHA:-32}"
VISION_LORA_DROPOUT="${VISION_LORA_DROPOUT:-0.05}"
VISION_LORA_NUM_LAYERS="${VISION_LORA_NUM_LAYERS:-0}"
VISION_LORA_TARGET_MODULES="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj}"
LLM_LORA_RANK="${LLM_LORA_RANK:-0}"
LLM_LORA_ALPHA="${LLM_LORA_ALPHA:-32}"
LLM_LORA_DROPOUT="${LLM_LORA_DROPOUT:-0.05}"
LLM_LORA_NUM_LAYERS="${LLM_LORA_NUM_LAYERS:-0}"
LLM_LORA_TARGET_MODULES="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj}"
USE_MULTISCALE="${USE_MULTISCALE:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
TEST_NUM="${TEST_NUM:-50}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-False}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ANALOGICAL_ROOT}:${PYTHONPATH:-}"
echo -e "\033[33mRoboTwin root: ${ROBOTWIN_ROOT}\033[0m"
echo -e "\033[33mGPU id: ${GPU_ID}; seed: ${SEED}; split: ${SPLIT}\033[0m"
echo -e "\033[33mcheckpoint: ${CHECKPOINT}\033[0m"
echo -e "\033[33mpolicy_action_mode=${POLICY_ACTION_MODE}; robotwin_action_type=${ROBOTWIN_ACTION_TYPE}\033[0m"
echo -e "\033[33mtest_num per (task,task_config): ${TEST_NUM}\033[0m"
echo -e "\033[33mtotal planned episodes: $((${#TASKS[@]} * ${#TASK_CONFIGS[@]} * TEST_NUM))\033[0m"
echo -e "\033[33meval_video_log: ${EVAL_VIDEO_LOG}\033[0m"

cd "${ROBOTWIN_ROOT}"

for task_config in "${TASK_CONFIGS[@]}"; do
  for task_name in "${TASKS[@]}"; do
    echo -e "\033[36m[eval] task=${task_name} task_config=${task_config}\033[0m"
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config "policy/${POLICY_NAME}/deploy_policy.yml" \
      --overrides \
      --task_name "${task_name}" \
      --task_config "${task_config}" \
      --ckpt_setting "${CHECKPOINT_ALIAS}" \
      --seed "${SEED}" \
      --test_num "${TEST_NUM}" \
      --eval_video_log "${EVAL_VIDEO_LOG}" \
      --policy_name "${POLICY_NAME}" \
      --instruction_type "${INSTRUCTION_TYPE}" \
      --checkpoint "${CHECKPOINT}" \
      --model_type "${MODEL_TYPE}" \
      --backbone "${BACKBONE}" \
      --num_history "${NUM_HISTORY}" \
      --prediction_len "${PREDICTION_LEN}" \
      --custom_img_size "${CUSTOM_IMG_SIZE}" \
      --rotation_format "${ROTATION_FORMAT}" \
      --policy_action_mode "${POLICY_ACTION_MODE}" \
      --robotwin_action_type "${ROBOTWIN_ACTION_TYPE}" \
      --pre_tokenize "${PRE_TOKENIZE}" \
      --relative_action "${RELATIVE_ACTION}" \
      --num_finetune_vision_layers "${NUM_FINETUNE_VISION_LAYERS}" \
      --fps_subsampling_factor "${FPS_SUBSAMPLING_FACTOR}" \
      --embedding_dim "${EMBEDDING_DIM}" \
      --num_attn_heads "${NUM_ATTN_HEADS}" \
      --num_vis_instr_attn_layers "${NUM_VIS_INSTR_ATTN_LAYERS}" \
      --num_shared_attn_layers "${NUM_SHARED_ATTN_LAYERS}" \
      --denoise_timesteps "${DENOISE_TIMESTEPS}" \
      --denoise_model "${DENOISE_MODEL}" \
      --motion_expert_config "${MOTION_EXPERT_CONFIG}" \
      --motion_expert_checkpoint "${MOTION_EXPERT_CHECKPOINT}" \
      --motion_solver_steps "${MOTION_SOLVER_STEPS}" \
      --motion_steps_to_run "${MOTION_STEPS_TO_RUN}" \
      --motion_finetune "${MOTION_FINETUNE}" \
      --motion_scene_fusion "${MOTION_SCENE_FUSION}" \
      --motion_action_fusion "${MOTION_ACTION_FUSION}" \
      --vision_lora_rank "${VISION_LORA_RANK}" \
      --vision_lora_alpha "${VISION_LORA_ALPHA}" \
      --vision_lora_dropout "${VISION_LORA_DROPOUT}" \
      --vision_lora_num_layers "${VISION_LORA_NUM_LAYERS}" \
      --vision_lora_target_modules "${VISION_LORA_TARGET_MODULES}" \
      --llm_lora_rank "${LLM_LORA_RANK}" \
      --llm_lora_alpha "${LLM_LORA_ALPHA}" \
      --llm_lora_dropout "${LLM_LORA_DROPOUT}" \
      --llm_lora_num_layers "${LLM_LORA_NUM_LAYERS}" \
      --llm_lora_target_modules "${LLM_LORA_TARGET_MODULES}" \
      --use_multiscale "${USE_MULTISCALE}" \
      --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  done
done
