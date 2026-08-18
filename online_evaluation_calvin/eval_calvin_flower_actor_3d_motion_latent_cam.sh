#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_motion_cam
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

# Online eval for the camera-frame-anchored motion-latent policy.
#
# --coord_frame camera makes the wrapper (a) build both point clouds and the
# proprio in CALVIN's fixed static-camera frame, and (b) map the predicted
# camera-frame delta chunk back to absolute WORLD TCP targets before handing
# them to the env -- verified to reproduce the world-frame executed trajectory
# to ~2e-7. Everything else matches the world-frame eval script.

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

model_type="${MODEL_TYPE:-flower_actor_3d_motion_latent_cam}"
coord_frame="${COORD_FRAME:-camera}"
# Default 'camera' = "the expert was trained in MY frame" (the value is
# frame-agnostic). Every arm of this experiment is frame-CONSISTENT.
motion_expert_frame="${MOTION_EXPERT_FRAME:-camera}"
seed="${SEED:-0}"
# Resolve the checkpoint. Training writes to a run dir whose name encodes the
# hyperparameters and job id (…-C960-B8-…-job9876919), so the old bare
# train_logs/calvin/<model_type>/last.pth never existed. Pick the most recently
# written run for this model_type and say so out loud; pass CHECKPOINT to pin one.
if [ -z "${CHECKPOINT:-}" ]; then
    checkpoint="$(ls -1t "${REPO_ROOT}/train_logs/calvin/${model_type}-"*/last.pth 2>/dev/null | head -1)"
    if [ -z "${checkpoint}" ]; then
        echo "[eval] FATAL: no checkpoint found for model_type=${model_type}." >&2
        echo "        Looked for ${REPO_ROOT}/train_logs/calvin/${model_type}-*/last.pth" >&2
        echo "        Pass CHECKPOINT=/path/to/<run>/last.pth explicitly." >&2
        exit 1
    fi
    echo "[eval] auto-resolved checkpoint (newest run for ${model_type})"
else
    checkpoint="${CHECKPOINT}"
fi
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
# embedding_dim must match the width the checkpoint was trained at.
# model_wrapper.py reconciles this from the checkpoint ONLY for the qwen_* model
# types (_maybe_override_qwen_eval_args_from_checkpoint returns early for
# everything else), so a florence2 checkpoint trained at a different width dies
# with "size mismatch for traj_encoder.weight". Recover it from the run dir name,
# which training always encodes as ...-C<width>-B<batch>-...
_ckpt_run="$(basename "$(dirname "${checkpoint}")")"
_ckpt_C="$(printf '%s' "${_ckpt_run}" | sed -n 's/.*-C\([0-9][0-9]*\)-B[0-9].*/\1/p')"
if [ -n "${_ckpt_C}" ]; then
    echo "[eval] embedding_dim=${_ckpt_C} (parsed from run dir)"
else
    echo "[eval] WARNING: could not parse -C<width>- from '${_ckpt_run}';" >&2
    echo "        falling back to 960. Pass EMBEDDING_DIM=<width> if that is wrong." >&2
fi
embedding_dim="${EMBEDDING_DIM:-${_ckpt_C:-960}}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_cam.yaml}"
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/calvin_florence2_cam/checkpoint_step_18000.pt}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
# Match training (scripts/calvin/train_flower_actor_motion_latent_cam.sh sets
# motion_finetune=true): the expert was updated in place, so its finetuned
# weights live in the policy checkpoint. Leaving this false also torch.compiles
# the expert, a needless divergence from the trained path.
motion_finetune="${MOTION_FINETUNE:-true}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"
motion_action_fusion="${MOTION_ACTION_FUSION:-false}"

ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

# Preflight: motion_expert_frame != world asserts "this expert was trained in MY
# frame". Verify the expert config declares it -- pairing a world-trained expert
# with a fixed-frame policy is a silent mismatch that nothing else raises on.
policy_frame="calvin_static"
if [ "${motion_expert_frame}" != "world" ]; then
  declared="$(grep -E '^[[:space:]]*camera_frame:' "${motion_expert_config}" | awk '{print $2}' | tr -d '\r')"
  declared="${declared:-world}"
  if [ "${declared}" != "${policy_frame}" ]; then
    echo "[eval] FATAL: motion_expert_frame=${motion_expert_frame} needs an expert trained in ${policy_frame}," >&2
    echo "        but ${motion_expert_config} declares camera_frame=${declared}." >&2
    exit 1
  fi
fi
if [ ! -f "${motion_expert_checkpoint}" ]; then
  echo "[eval] FATAL: motion expert checkpoint not found: ${motion_expert_checkpoint}" >&2
  echo "        Train it first: sbatch scripts/motion_expert/train_motionexpert_florence2_calvin_cam.sh" >&2
  exit 1
fi

# last.pth is rewritten in place every 5000 iters (utils/trainers/base.py:877),
# ~17 GB per write. Reading it mid-write loads a truncated file. If the training
# run is still alive, wait for the size to settle before touching it.
_sz_a="$(stat -c %s "${checkpoint}" 2>/dev/null || echo 0)"
sleep 20
_sz_b="$(stat -c %s "${checkpoint}" 2>/dev/null || echo 0)"
if [ "${_sz_a}" != "${_sz_b}" ]; then
    echo "[eval] ${checkpoint} is being written right now (${_sz_a} -> ${_sz_b})." >&2
    echo "        Waiting for the save to finish..." >&2
    while true; do
        sleep 60
        _sz_a="${_sz_b}"
        _sz_b="$(stat -c %s "${checkpoint}" 2>/dev/null || echo 0)"
        [ "${_sz_a}" = "${_sz_b}" ] && break
    done
    echo "[eval] settled at ${_sz_b} bytes." >&2
fi

echo "[eval] CAMERA-FRAME: coord_frame=${coord_frame} motion_expert_frame=${motion_expert_frame}"
echo "[eval] checkpoint=${checkpoint} ($(stat -c %s "${checkpoint}") bytes)"
echo "[eval] base_log_dir=${base_log_dir}"

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
    --coord_frame "${coord_frame}" \
    --motion_expert_frame "${motion_expert_frame}" \
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
    --motion_scene_fusion "${motion_scene_fusion}" \
    --motion_action_fusion "${motion_action_fusion}"
