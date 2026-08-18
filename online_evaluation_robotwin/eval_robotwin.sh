#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh easy 0 0 lift_pot True
#   CHECKPOINT=/path/to/last.pth bash online_evaluation_robotwin/eval_robotwin.sh hard 0 0 lift_pot,handover_mic False
# Positional args:
#   1: split/task config shortcut: easy|clean|hard|randomized|both (default: both)
#   2: gpu id (default: 0)
#   3: seed (default: 0)
#   4: task selector: comma-separated task names or "all" (default: all 9 action15_stride2 zarr tasks)
#   5: save videos: True|False (default: False)
#
# Useful env overrides:
#   TASK_CONFIG=aloha_vla_clean50_tcp_analogical
#   SAVE_VIDEO=True|False
#   POLICY_ACTION_MODE=absolute|relative
#   RELATIVE_ROTATION=True|False
#   ROBOTWIN_ACTION_TYPE=ee|delta_ee
#   ROTATION_FORMAT=quat_wxyz
#   ROBOTWIN_TCP_QUAT_CONVENTION=wxyz
#   FAST_EVAL_NO_INNER_RENDER=True|False
#   EXPERT_CHECK=True|False
#   STEP_LIM=200
#   TEST_NUM=50
#   SUMMARY_DIR=/path/to/summary_dir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALOGICAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/user_data/ziyinx/RoboTwin}"

SPLIT="${1:-${SPLIT:-both}}"
GPU_ID="${2:-${GPU_ID:-0}}"
SEED="${3:-${SEED:-0}}"
TASK_SELECTOR="${4:-${TASK_SELECTOR:-all}}"
SAVE_VIDEO="${5:-${SAVE_VIDEO:-${EVAL_VIDEO_LOG:-False}}}"

normalize_bool() {
  case "${1}" in
    1|true|True|TRUE|yes|Yes|YES|y|Y) echo "True" ;;
    0|false|False|FALSE|no|No|NO|n|N) echo "False" ;;
    *) echo "${1}" ;;
  esac
}

SAVE_VIDEO="$(normalize_bool "${SAVE_VIDEO}")"

POLICY_NAME="analogical_manipulation"
CHECKPOINT="${CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/robotwin2/flower_actor_3d_motion_latent-RoboTwin2-C912-B8-lr4e-5-tristage_flower-H1-rectified_flow-florence2-motion_finetuned_true-job8146359/150000.pth}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT is required. Example: CHECKPOINT=/path/to/last.pth bash $0 easy 0 0 lift_pot" >&2
  exit 1
fi

CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${CHECKPOINT}")")}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-seen}"

EASY_CONFIG="${EASY_CONFIG:-aloha_vla_clean50_tcp_analogical}"
HARD_CONFIG="${HARD_CONFIG:-aloha_vla_randomized500_tcp_analogical}"

label_for_config() {
  case "${1}" in
    *clean*) echo "clean" ;;
    *randomized*) echo "randomized" ;;
    *) echo "custom" ;;
  esac
}

TASK_CONFIGS=()
TASK_CONFIG_LABELS=()
add_task_config() {
  local config_name="$1"
  local label="${2:-$(label_for_config "${config_name}")}"
  TASK_CONFIGS+=("${config_name}")
  TASK_CONFIG_LABELS+=("${label}")
}

case "${SPLIT}" in
  easy|clean)
    add_task_config "${TASK_CONFIG:-${EASY_CONFIG}}" "clean"
    ;;
  hard|randomized)
    add_task_config "${TASK_CONFIG:-${HARD_CONFIG}}" "randomized"
    ;;
  both)
    if [[ -n "${TASK_CONFIG:-}" ]]; then
      add_task_config "${TASK_CONFIG}" "$(label_for_config "${TASK_CONFIG}")"
    else
      add_task_config "${EASY_CONFIG}" "clean"
      add_task_config "${HARD_CONFIG}" "randomized"
    fi
    ;;
  *)
    add_task_config "${TASK_CONFIG:-${SPLIT}}" "$(label_for_config "${TASK_CONFIG:-${SPLIT}}")"
    ;;
esac

ROBOTWIN2_ZARR_TASKS=(
  beat_block_hammer
  pick_dual_bottles
  place_phone_stand
  move_can_pot
  place_a2b_left
  handover_mic
  handover_block
  stack_bowls_two
  put_bottles_dustbin
)

if [[ "${TASK_SELECTOR}" == "all" ]]; then
  TASKS=("${ROBOTWIN2_ZARR_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "${TASK_SELECTOR}"
fi

is_zarr_task() {
  local needle="$1"
  local task
  for task in "${ROBOTWIN2_ZARR_TASKS[@]}"; do
    if [[ "${task}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

ALLOW_NON_ZARR_TASKS="$(normalize_bool "${ALLOW_NON_ZARR_TASKS:-False}")"
if [[ "${ALLOW_NON_ZARR_TASKS}" != "True" ]]; then
  for task_name in "${TASKS[@]}"; do
    if ! is_zarr_task "${task_name}"; then
      echo "Task '${task_name}' is not in the 9-task RobotWin2 action15_stride2 zarr subset." >&2
      echo "Set ALLOW_NON_ZARR_TASKS=True to override intentionally." >&2
      exit 1
    fi
  done
fi

MODEL_TYPE="${MODEL_TYPE:-flower_actor_3d_motion_latent}"
BACKBONE="${BACKBONE:-florence2}"
NUM_HISTORY="${NUM_HISTORY:-1}"
PREDICTION_LEN="${PREDICTION_LEN:-15}"
CUSTOM_IMG_SIZE="${CUSTOM_IMG_SIZE:-224}"
ROTATION_FORMAT="${ROTATION_FORMAT:-quat_wxyz}"
POLICY_ACTION_MODE="${POLICY_ACTION_MODE:-relative}"
RELATIVE_ROTATION="$(normalize_bool "${RELATIVE_ROTATION:-False}")"
ROBOTWIN_ACTION_TYPE="${ROBOTWIN_ACTION_TYPE:-ee}"
ROBOTWIN_TCP_QUAT_CONVENTION="${ROBOTWIN_TCP_QUAT_CONVENTION:-wxyz}"
FAST_EVAL_NO_INNER_RENDER="$(normalize_bool "${FAST_EVAL_NO_INNER_RENDER:-True}")"
EXPERT_CHECK="$(normalize_bool "${EXPERT_CHECK:-True}")"
STEP_LIM="${STEP_LIM:-None}"

PRE_TOKENIZE="${PRE_TOKENIZE:-True}"
if [[ -z "${RELATIVE_ACTION:-}" ]]; then
  if [[ "${POLICY_ACTION_MODE}" == "relative" ]]; then
    RELATIVE_ACTION="True"
  else
    RELATIVE_ACTION="False"
  fi
fi
NUM_FINETUNE_VISION_LAYERS="${NUM_FINETUNE_VISION_LAYERS:-0}"
FINETUNE_BACKBONE="${FINETUNE_BACKBONE:-True}"
FINETUNE_TEXT_ENCODER="${FINETUNE_TEXT_ENCODER:-False}"
FPS_SUBSAMPLING_FACTOR="${FPS_SUBSAMPLING_FACTOR:-1}"
EMBEDDING_DIM="${EMBEDDING_DIM:-912}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-16}"
NUM_VIS_INSTR_ATTN_LAYERS="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
NUM_SHARED_ATTN_LAYERS="${NUM_SHARED_ATTN_LAYERS:-16}"
DENOISE_TIMESTEPS="${DENOISE_TIMESTEPS:-5}"
DENOISE_MODEL="${DENOISE_MODEL:-rectified_flow}"
CONTINUOUS_OPENNESS="${CONTINUOUS_OPENNESS:-False}"
BINARY_CLOSED_PROB_THRESHOLD="${BINARY_CLOSED_PROB_THRESHOLD:-0.5}"
# Sticky-grasp gripper latch (off by default). Once the gripper closes it stays
# closed unless p_close stays <= GRIPPER_OPEN_THRESH for GRIPPER_MIN_OPEN_STEPS
# consecutive executed frames. Stops 1-2 frame prediction dips from dropping a
# grasped object during open-loop chunk execution.
GRIPPER_LATCH="${GRIPPER_LATCH:-False}"
GRIPPER_OPEN_THRESH="${GRIPPER_OPEN_THRESH:-0.2}"
GRIPPER_MIN_OPEN_STEPS="${GRIPPER_MIN_OPEN_STEPS:-3}"
GRIPPER_CLOSE_THRESH="${GRIPPER_CLOSE_THRESH:-None}"

MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG:-/data/user_data/ziyinx/analogical_manipulation_clean/motion_expert/configs/robotwin2_florence2.yaml}"
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-None}"
MOTION_SOLVER_STEPS="${MOTION_SOLVER_STEPS:-10}"
MOTION_STEPS_TO_RUN="${MOTION_STEPS_TO_RUN:-1}"
MOTION_FINETUNE="${MOTION_FINETUNE:-False}"
MOTION_SCENE_FUSION="${MOTION_SCENE_FUSION:-True}"
MOTION_ACTION_FUSION="${MOTION_ACTION_FUSION:-False}"

JOINT_ACTION_DIM="${JOINT_ACTION_DIM:-6}"
JOINT_PROPRIO_DIM="${JOINT_PROPRIO_DIM:-7}"
GRIPPER_DIM="${GRIPPER_DIM:-1}"
JOINT_NUM_LAYERS="${JOINT_NUM_LAYERS:-12}"
JOINT_EMBEDDING_DIM="${JOINT_EMBEDDING_DIM:-0}"
JOINT_NUM_ATTN_HEADS="${JOINT_NUM_ATTN_HEADS:-0}"
NUM_CAMERA_IDS="${NUM_CAMERA_IDS:-3}"

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
EVAL_VIDEO_LOG="${SAVE_VIDEO}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ANALOGICAL_ROOT}:${PYTHONPATH:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_DIR="${SUMMARY_DIR:-${ANALOGICAL_ROOT}/eval_logs/robotwin2/${CHECKPOINT_ALIAS}/seed${SEED}/${RUN_ID}}"
SUMMARY_CSV="${SUMMARY_CSV:-${SUMMARY_DIR}/success_rates.csv}"
SUMMARY_TXT="${SUMMARY_TXT:-${SUMMARY_DIR}/summary.txt}"
mkdir -p "${SUMMARY_DIR}"
echo "split,task_config,task,successes,rollouts,success_rate,result_file" > "${SUMMARY_CSV}"

echo -e "\033[33mRoboTwin root: ${ROBOTWIN_ROOT}\033[0m"
echo -e "\033[33mGPU id: ${GPU_ID}; seed: ${SEED}; split: ${SPLIT}\033[0m"
echo -e "\033[33mcheckpoint: ${CHECKPOINT}\033[0m"
echo -e "\033[33mmodel=${MODEL_TYPE}; backbone=${BACKBONE}; rotation=${ROTATION_FORMAT}; pred_len=${PREDICTION_LEN}\033[0m"
echo -e "\033[33mpolicy_action_mode=${POLICY_ACTION_MODE}; robotwin_action_type=${ROBOTWIN_ACTION_TYPE}; relative_action=${RELATIVE_ACTION}; relative_rotation=${RELATIVE_ROTATION}\033[0m"
echo -e "\033[33mcontinuous_openness=${CONTINUOUS_OPENNESS}; binary_closed_prob_threshold=${BINARY_CLOSED_PROB_THRESHOLD}; chunk_gripper_heuristics=disabled\033[0m"
echo -e "\033[33mrobotwin_tcp_quat_convention=${ROBOTWIN_TCP_QUAT_CONVENTION}\033[0m"
echo -e "\033[33mfast_eval_no_inner_render=${FAST_EVAL_NO_INNER_RENDER}; expert_check=${EXPERT_CHECK}; step_lim=${STEP_LIM}\033[0m"
echo -e "\033[33mtest_num per (task,task_config): ${TEST_NUM}\033[0m"
echo -e "\033[33mtotal planned episodes: $((${#TASKS[@]} * ${#TASK_CONFIGS[@]} * TEST_NUM))\033[0m"
echo -e "\033[33meval_video_log: ${EVAL_VIDEO_LOG}\033[0m"
echo -e "\033[33msummary csv: ${SUMMARY_CSV}\033[0m"

cd "${ROBOTWIN_ROOT}"

# Tasks whose eval produced no success rate at all. eval_policy.py can exit 0
# without evaluating anything -- most notably script/test_render.py, which on a
# node whose SAPIEN renderer fails to initialize prints "Render Error" and calls
# bare exit() (status 0). Without the check below, record_result would append an
# empty CSV row, the awk summary would skip it (it requires a non-empty success
# column), no <split>_success_rate line would be printed, and this script would
# still exit 0 -- i.e. a task that never ran would be reported as done.
EVAL_FAILED_TASKS=""

record_result() {
  local split_label="$1"
  local task_config="$2"
  local task_name="$3"
  local log_file="$4"
  local result_file
  local abs_result_file
  local success_rate
  local successes

  result_file="$(sed -n 's/^Data has been saved to //p' "${log_file}" | tail -n 1 || true)"
  abs_result_file="${result_file}"
  if [[ -n "${abs_result_file}" && "${abs_result_file}" != /* ]]; then
    abs_result_file="${ROBOTWIN_ROOT}/${abs_result_file}"
  fi

  success_rate=""
  successes=""
  if [[ -n "${abs_result_file}" && -f "${abs_result_file}" ]]; then
    success_rate="$(tail -n 1 "${abs_result_file}" | tr -d '[:space:]')"
    if [[ -n "${success_rate}" ]]; then
      successes="$(awk -v r="${success_rate}" -v n="${TEST_NUM}" 'BEGIN { printf "%d", (r * n) + 0.5 }')"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${split_label}" "${task_config}" "${task_name}" \
    "${successes}" "${TEST_NUM}" "${success_rate}" "${abs_result_file}" \
    >> "${SUMMARY_CSV}"

  if [[ -z "${success_rate}" ]]; then
    local reason="eval_policy.py produced no result file"
    if grep -q "Render Error" "${log_file}"; then
      reason="SAPIEN renderer failed to initialize on $(hostname) (\"Render Error\" from script/test_render.py)"
    elif [[ -n "${abs_result_file}" ]]; then
      reason="result file is missing or empty: ${abs_result_file}"
    fi
    echo -e "\033[31m[eval] FAILED split=${split_label} task=${task_name}: ${reason}\033[0m" >&2
    echo -e "\033[31m[eval]   log: ${log_file}\033[0m" >&2
    EVAL_FAILED_TASKS="${EVAL_FAILED_TASKS}${EVAL_FAILED_TASKS:+ }${split_label}/${task_name}"
  fi
}

for config_i in "${!TASK_CONFIGS[@]}"; do
  task_config="${TASK_CONFIGS[${config_i}]}"
  split_label="${TASK_CONFIG_LABELS[${config_i}]}"
  for task_name in "${TASKS[@]}"; do
    echo -e "\033[36m[eval] split=${split_label} task=${task_name} task_config=${task_config}\033[0m"
    eval_log="${SUMMARY_DIR}/${split_label}_${task_name}.log"
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
      --relative_rotation "${RELATIVE_ROTATION}" \
      --robotwin_action_type "${ROBOTWIN_ACTION_TYPE}" \
      --robotwin_tcp_quat_convention "${ROBOTWIN_TCP_QUAT_CONVENTION}" \
      --fast_eval_no_inner_render "${FAST_EVAL_NO_INNER_RENDER}" \
      --expert_check "${EXPERT_CHECK}" \
      --step_lim "${STEP_LIM}" \
      --pre_tokenize "${PRE_TOKENIZE}" \
      --relative_action "${RELATIVE_ACTION}" \
      --num_finetune_vision_layers "${NUM_FINETUNE_VISION_LAYERS}" \
      --finetune_backbone "${FINETUNE_BACKBONE}" \
      --finetune_text_encoder "${FINETUNE_TEXT_ENCODER}" \
      --fps_subsampling_factor "${FPS_SUBSAMPLING_FACTOR}" \
      --embedding_dim "${EMBEDDING_DIM}" \
      --num_attn_heads "${NUM_ATTN_HEADS}" \
      --num_vis_instr_attn_layers "${NUM_VIS_INSTR_ATTN_LAYERS}" \
      --num_shared_attn_layers "${NUM_SHARED_ATTN_LAYERS}" \
      --denoise_timesteps "${DENOISE_TIMESTEPS}" \
      --denoise_model "${DENOISE_MODEL}" \
      --continuous_openness "${CONTINUOUS_OPENNESS}" \
      --binary_closed_prob_threshold "${BINARY_CLOSED_PROB_THRESHOLD}" \
      --gripper_latch "${GRIPPER_LATCH}" \
      --gripper_open_thresh "${GRIPPER_OPEN_THRESH}" \
      --gripper_min_open_steps "${GRIPPER_MIN_OPEN_STEPS}" \
      --gripper_close_thresh "${GRIPPER_CLOSE_THRESH}" \
      --motion_expert_config "${MOTION_EXPERT_CONFIG}" \
      --motion_expert_checkpoint "${MOTION_EXPERT_CHECKPOINT}" \
      --motion_solver_steps "${MOTION_SOLVER_STEPS}" \
      --motion_steps_to_run "${MOTION_STEPS_TO_RUN}" \
      --motion_finetune "${MOTION_FINETUNE}" \
      --motion_scene_fusion "${MOTION_SCENE_FUSION}" \
      --motion_action_fusion "${MOTION_ACTION_FUSION}" \
      --joint_action_dim "${JOINT_ACTION_DIM}" \
      --joint_proprio_dim "${JOINT_PROPRIO_DIM}" \
      --gripper_dim "${GRIPPER_DIM}" \
      --joint_num_layers "${JOINT_NUM_LAYERS}" \
      --joint_embedding_dim "${JOINT_EMBEDDING_DIM}" \
      --joint_num_attn_heads "${JOINT_NUM_ATTN_HEADS}" \
      --num_camera_ids "${NUM_CAMERA_IDS}" \
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
      --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
      2>&1 | tee "${eval_log}"
    record_result "${split_label}" "${task_config}" "${task_name}" "${eval_log}"
  done
done

{
  echo "summary_csv=${SUMMARY_CSV}"
  for label in clean randomized custom; do
    awk -F, -v label="${label}" '
      NR > 1 && $1 == label && $4 != "" { success += $4; total += $5 }
      END {
        if (total > 0) {
          printf "%s_success_rate=%d/%d %.6f\n", label, success, total, success / total
        }
      }
    ' "${SUMMARY_CSV}"
  done
} | tee "${SUMMARY_TXT}"

echo -e "\033[32mWrote RobotWin2 eval summary to ${SUMMARY_CSV}\033[0m"

# Exit non-zero so a task that never actually ran shows up as a FAILED slurm job
# instead of a silent COMPLETED with no numbers in it.
if [[ -n "${EVAL_FAILED_TASKS}" ]]; then
  echo -e "\033[31m[eval] ERROR: no success rate was produced for: ${EVAL_FAILED_TASKS}\033[0m" >&2
  echo -e "\033[31m[eval] These (task, split) pairs must be re-run; the summary above is incomplete.\033[0m" >&2
  exit 1
fi
