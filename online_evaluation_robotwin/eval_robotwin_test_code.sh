#!/usr/bin/env bash

set -euo pipefail

# Circle-policy eval harness:
# This script writes a temporary RoboTwin policy module whose action is:
# each hand moves in a circle around its initial TCP position.
#
# Usage:
#   bash online_evaluation_robotwin/eval_robotwin_test_code.sh easy 0 0 grab_roller
#   TEST_NUM=5 EVAL_VIDEO_LOG=True bash online_evaluation_robotwin/eval_robotwin_test_code.sh hard 0 0 grab_roller,handover_mic

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data/user_data/ziyinx/RoboTwin}"

SPLIT="${1:-${SPLIT:-easy}}"
GPU_ID="${2:-${GPU_ID:-0}}"
SEED="${3:-${SEED:-0}}"
TASK_SELECTOR="${4:-${TASK_SELECTOR:-grab_roller}}"

POLICY_NAME="circle_test_policy"
CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-circle_fixed_policy}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
TEST_NUM="${TEST_NUM:-50}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-True}"
ROBOTWIN_ACTION_TYPE="${ROBOTWIN_ACTION_TYPE:-ee}"
STEP_LIM="${STEP_LIM:-}"

# Circle parameters
CIRCLE_RADIUS="${CIRCLE_RADIUS:-0.05}"          # meters
CIRCLE_OMEGA="${CIRCLE_OMEGA:-0.20}"            # radians per policy step
CIRCLE_Z_BIAS="${CIRCLE_Z_BIAS:-0.00}"          # meters
LEFT_PHASE="${LEFT_PHASE:-0.0}"                 # radians
RIGHT_PHASE="${RIGHT_PHASE:-3.141592653589793}" # radians

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
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock
  click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot
  move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave
  pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket
  place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
  place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale
  place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet
  rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two
  stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

if [[ "${TASK_SELECTOR}" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "${TASK_SELECTOR}"
fi

POLICY_DIR="${ROBOTWIN_ROOT}/policy/${POLICY_NAME}"
mkdir -p "${POLICY_DIR}"

cat > "${POLICY_DIR}/__init__.py" <<'PY'
from .deploy_policy import eval, get_model, reset_model
__all__ = ["eval", "get_model", "reset_model"]
PY

cat > "${POLICY_DIR}/deploy_policy.yml" <<'YAML'
policy_name: circle_test_policy
task_name: null
task_config: null
ckpt_setting: null
seed: null
instruction_type: unseen
test_num: 50
eval_video_log: true
robotwin_action_type: ee
circle_radius: 0.05
circle_omega: 0.20
circle_z_bias: 0.00
left_phase: 0.0
right_phase: 3.141592653589793
YAML

cat > "${POLICY_DIR}/deploy_policy.py" <<'PY'
import numpy as np


class CircleModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_cache = []
        self.step_idx = 0
        self.left_anchor = None
        self.right_anchor = None
        self.left_quat = None
        self.right_quat = None
        self.left_open = 0.5
        self.right_open = 0.5

    def update_obs(self, observation):
        self.obs_cache.append(observation)
        endpose = observation.get("endpose", {})
        left_tcp = np.asarray(endpose.get("left_tcp_pose", []), dtype=np.float64)
        right_tcp = np.asarray(endpose.get("right_tcp_pose", []), dtype=np.float64)
        if left_tcp.shape[0] == 7 and right_tcp.shape[0] == 7:
            if self.left_anchor is None:
                self.left_anchor = left_tcp[:3].copy()
                self.left_quat = left_tcp[3:7].copy()
            if self.right_anchor is None:
                self.right_anchor = right_tcp[:3].copy()
                self.right_quat = right_tcp[3:7].copy()
        if "left_gripper" in endpose:
            self.left_open = float(endpose["left_gripper"])
        if "right_gripper" in endpose:
            self.right_open = float(endpose["right_gripper"])

    def _circle_target(self, anchor_xyz, anchor_quat, phase):
        theta = self.step_idx * self.cfg["circle_omega"] + phase
        dx = self.cfg["circle_radius"] * np.cos(theta)
        dy = self.cfg["circle_radius"] * np.sin(theta)
        dz = self.cfg["circle_z_bias"]
        pos = anchor_xyz + np.array([dx, dy, dz], dtype=np.float64)
        return np.concatenate([pos, anchor_quat], axis=0)

    def get_action(self):
        if self.left_anchor is None or self.right_anchor is None:
            return np.zeros((1, 16), dtype=np.float32)
        left_pose = self._circle_target(self.left_anchor, self.left_quat, self.cfg["left_phase"])
        right_pose = self._circle_target(self.right_anchor, self.right_quat, self.cfg["right_phase"])
        action = np.concatenate([left_pose, [self.left_open], right_pose, [self.right_open]], axis=0)
        self.step_idx += 1
        return np.asarray([action], dtype=np.float32)


def get_model(usr_args):
    cfg = {
        "robotwin_action_type": usr_args.get("robotwin_action_type", "ee"),
        "circle_radius": float(usr_args.get("circle_radius", 0.05)),
        "circle_omega": float(usr_args.get("circle_omega", 0.2)),
        "circle_z_bias": float(usr_args.get("circle_z_bias", 0.0)),
        "left_phase": float(usr_args.get("left_phase", 0.0)),
        "right_phase": float(usr_args.get("right_phase", np.pi)),
    }
    return CircleModel(cfg)


def eval(TASK_ENV, model, observation):
    if len(model.obs_cache) == 0:
        model.update_obs(observation)
    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action, action_type=model.cfg["robotwin_action_type"])
        observation = TASK_ENV.get_obs()
        model.update_obs(observation)
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    model.obs_cache = []
    model.step_idx = 0
    model.left_anchor = None
    model.right_anchor = None
    model.left_quat = None
    model.right_quat = None
PY

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo -e "\033[33mRoboTwin root: ${ROBOTWIN_ROOT}\033[0m"
echo -e "\033[33mPolicy: ${POLICY_NAME} (auto-generated fixed circle controller)\033[0m"
echo -e "\033[33mGPU id: ${GPU_ID}; seed: ${SEED}; split: ${SPLIT}\033[0m"
echo -e "\033[33mtest_num per (task,task_config): ${TEST_NUM}\033[0m"
echo -e "\033[33mtotal planned episodes: $((${#TASKS[@]} * ${#TASK_CONFIGS[@]} * TEST_NUM))\033[0m"
echo -e "\033[33meval_video_log: ${EVAL_VIDEO_LOG}\033[0m"
if [[ -n "${STEP_LIM}" ]]; then
  echo -e "\033[33mstep_lim override: ${STEP_LIM}\033[0m"
else
  echo -e "\033[33mstep_lim override: <task default from _eval_step_limit.yml>\033[0m"
fi
echo -e "\033[33mcircle: radius=${CIRCLE_RADIUS}, omega=${CIRCLE_OMEGA}, z_bias=${CIRCLE_Z_BIAS}\033[0m"

cd "${ROBOTWIN_ROOT}"
for task_config in "${TASK_CONFIGS[@]}"; do
  for task_name in "${TASKS[@]}"; do
    STEP_LIM_ARGS=()
    if [[ -n "${STEP_LIM}" ]]; then
      STEP_LIM_ARGS=(--step_lim "${STEP_LIM}")
    fi
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
      --robotwin_action_type "${ROBOTWIN_ACTION_TYPE}" \
      "${STEP_LIM_ARGS[@]}" \
      --circle_radius "${CIRCLE_RADIUS}" \
      --circle_omega "${CIRCLE_OMEGA}" \
      --circle_z_bias "${CIRCLE_Z_BIAS}" \
      --left_phase "${LEFT_PHASE}" \
      --right_phase "${RIGHT_PHASE}"
  done
done
