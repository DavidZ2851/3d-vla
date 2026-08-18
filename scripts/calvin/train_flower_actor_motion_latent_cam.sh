#!/usr/bin/env bash
#SBATCH --job-name=calvin_motion_cam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"

set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-online}

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
DATA_PATH="${DATA_PATH:-/data/group_data/katefgroup-ssd/world_model/ziyinx}"
mkdir -p "/data/user_data/${USER}/logs"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# CALVIN train zarr now lives on the SSD tier; override with CALVIN_DATA_ROOT.
CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/data/group_data/katefgroup-ssd/world_model/ziyinx}"
train_data_dir="${TRAIN_DATA_DIR:-${CALVIN_DATA_ROOT}/calvin_zarr/train_rechunked8.zarr}"
eval_data_dir="${EVAL_DATA_DIR:-/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr}"
train_instructions="${TRAIN_INSTRUCTIONS:-instructions/calvin/train_instructions.json}"
val_instructions="${VAL_INSTRUCTIONS:-instructions/calvin/val_instructions.json}"

dataset="${DATASET:-CalvinCamContact10th}"
num_workers="${NUM_WORKERS:-4}"
memory_limit="${MEMORY_LIMIT:-6}"
B="${B:-8}"
B_val="${B_VAL:-8}"
chunk_size="${CHUNK_SIZE:-8}"

val_freq="${VAL_FREQ:-5000}"
eval_only="${EVAL_ONLY:-false}"
lr="${LR:-2e-5}"
lr_scheduler="${LR_SCHEDULER:-tristage_flower}"
backbone_lr="${BACKBONE_LR:-2e-5}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-0}"
vision_lr_scale="${VISION_LR_SCALE:-0.2}"
projector_lr_scale="${PROJECTOR_LR_SCALE:-0.4}"
wd="${WD:-0.05}"
train_iters="${TRAIN_ITERS:-50000}"
use_compile="${USE_COMPILE:-false}"
use_ema="${USE_EMA:-true}"

model_type="${MODEL_TYPE:-flower_actor_3d_motion_latent_cam}"
bimanual="${BIMANUAL:-false}"
keypose_only="${KEYPOSE_ONLY:-false}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"

backbone="${BACKBONE:-florence2}"
finetune_backbone="${FINETUNE_BACKBONE:-true}"
finetune_text_encoder="${FINETUNE_TEXT_ENCODER:-false}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"

C="${C:-960}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_history="${NUM_HISTORY:-1}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"

workspace_normalizer_buffer="${WORKSPACE_NORMALIZER_BUFFER:-0.0}"
rotation_format="${ROTATION_FORMAT:-euler}"
relative_action="${RELATIVE_ACTION:-true}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_cam.yaml}"
# Prefer the final 20k expert, fall back to 18k. Resolved when the job STARTS,
# not when it is submitted, so a queued policy run automatically picks up the
# 20k checkpoint that the expert run writes while this job waits for GPUs.
# The size check rejects a 20k file that is still being written (~3.2 GB).
_expert_dir="${REPO_ROOT}/train_logs/motion_expert/calvin_florence2_cam"
_expert_18k="${_expert_dir}/checkpoint_step_18000.pt"
_expert_20k="${_expert_dir}/checkpoint_step_20000.pt"
_expert_default="${_expert_18k}"
if [ -f "${_expert_20k}" ]; then
  if [ ! -f "${_expert_18k}" ] || \
     [ "$(stat -c %s "${_expert_20k}")" -ge "$(stat -c %s "${_expert_18k}")" ]; then
    _expert_default="${_expert_20k}"
  else
    echo "[train] WARNING: ${_expert_20k} is smaller than the 18k checkpoint" >&2
    echo "        (still being written?); falling back to 18k." >&2
  fi
fi
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${_expert_default}}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
motion_finetune="${MOTION_FINETUNE:-true}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"
motion_action_fusion="${MOTION_ACTION_FUSION:-false}"
# 'world' reuses the existing world-frame expert checkpoint unchanged (the
# camera-frame query is converted back to world for the expert call, and its
# tokens are then placed at camera-frame positions). 'camera' expects an expert
# trained on camera-frame tracks.
# Default 'camera' = "the expert was trained in MY frame" (the value is
# frame-agnostic). Every arm of this experiment is frame-CONSISTENT.
motion_expert_frame="${MOTION_EXPERT_FRAME:-camera}"

main_dir="${MAIN_DIR:-calvin}"
base_log_dir="${BASE_LOG_DIR:-${REPO_ROOT}/train_logs}"
run_log_dir="${RUN_LOG_DIR:-${model_type}-${dataset}-C${C}-B${B}-lr${lr}-${lr_scheduler}-H${num_history}-${denoise_model}-${backbone}-motion_finetuned_${motion_finetune}-job${SLURM_JOBID:-local}}"
checkpoint="${CHECKPOINT:-${base_log_dir}/${main_dir}/${run_log_dir}/last.pth}"
finetune="${FINETUNE:-false}"

ngpus="${NGPUS:-4}"
master_port="${MASTER_PORT:-$((10000 + ${SLURM_JOBID:-12345} % 50000))}"

# Preflight: motion_expert_frame != world asserts "this expert was trained in MY
# frame". Verify the expert config declares it -- pairing a world-trained expert
# with a fixed-frame policy is a silent mismatch that nothing else raises on.
policy_frame="calvin_static"
if [ "${motion_expert_frame}" != "world" ]; then
  declared="$(grep -E '^[[:space:]]*camera_frame:' "${motion_expert_config}" | awk '{print $2}' | tr -d '\r')"
  declared="${declared:-world}"
  if [ "${declared}" != "${policy_frame}" ]; then
    echo "[train] FATAL: motion_expert_frame=${motion_expert_frame} needs an expert trained in ${policy_frame}," >&2
    echo "        but ${motion_expert_config} declares camera_frame=${declared}." >&2
    exit 1
  fi
fi
if [ ! -f "${motion_expert_checkpoint}" ]; then
  echo "[train] FATAL: motion expert checkpoint not found: ${motion_expert_checkpoint}" >&2
  echo "        Train it first: sbatch scripts/motion_expert/train_motionexpert_florence2_calvin_cam.sh" >&2
  exit 1
fi

echo "[train] CAMERA-FRAME anchoring: model=${model_type} dataset=${dataset}"
echo "[train] motion expert=${motion_expert_checkpoint} (frame=${motion_expert_frame})"
echo "[train] log_dir=${base_log_dir}/${main_dir}/${run_log_dir}"

torchrun --nproc_per_node "${ngpus}" --master_port "${master_port}" \
    "${REPO_ROOT}/main.py" \
    --train_data_dir "${train_data_dir}" \
    --eval_data_dir "${eval_data_dir}" \
    --train_instructions "${train_instructions}" \
    --val_instructions "${val_instructions}" \
    --dataset "${dataset}" \
    --num_workers "${num_workers}" \
    --memory_limit "${memory_limit}" \
    --batch_size "${B}" \
    --batch_size_val "${B_val}" \
    --chunk_size "${chunk_size}" \
    --base_log_dir "${base_log_dir}" \
    --exp_log_dir "${main_dir}" \
    --run_log_dir "${run_log_dir}" \
    --checkpoint "${checkpoint}" \
    --val_freq "${val_freq}" \
    --eval_only "${eval_only}" \
    --lr "${lr}" \
    --lr_scheduler "${lr_scheduler}" \
    --backbone_lr "${backbone_lr}" \
    --wd "${wd}" \
    --train_iters "${train_iters}" \
    --use_compile "${use_compile}" \
    --use_ema "${use_ema}" \
    --model_type "${model_type}" \
    --bimanual "${bimanual}" \
    --keypose_only "${keypose_only}" \
    --pre_tokenize "${pre_tokenize}" \
    --custom_img_size "${custom_img_size}" \
    --backbone "${backbone}" \
    --finetune_backbone "${finetune_backbone}" \
    --finetune_text_encoder "${finetune_text_encoder}" \
    --fps_subsampling_factor "${fps_subsampling_factor}" \
    --embedding_dim "${C}" \
    --num_attn_heads "${num_attn_heads}" \
    --num_vis_instr_attn_layers "${num_vis_instr_attn_layers}" \
    --num_history "${num_history}" \
    --num_shared_attn_layers "${num_shared_attn_layers}" \
    --workspace_normalizer_buffer "${workspace_normalizer_buffer}" \
    --relative_action "${relative_action}" \
    --rotation_format "${rotation_format}" \
    --denoise_timesteps "${denoise_timesteps}" \
    --denoise_model "${denoise_model}" \
    --finetune "${finetune}" \
    --num_finetune_vision_layers "${num_finetune_vision_layers}" \
    --vision_lr_scale "${vision_lr_scale}" \
    --projector_lr_scale "${projector_lr_scale}" \
    --motion_expert_config "${motion_expert_config}" \
    --motion_expert_checkpoint "${motion_expert_checkpoint}" \
    --motion_solver_steps "${motion_solver_steps}" \
    --motion_steps_to_run "${motion_steps_to_run}" \
    --motion_finetune "${motion_finetune}" \
    --motion_scene_fusion "${motion_scene_fusion}" \
    --motion_action_fusion "${motion_action_fusion}" \
    --motion_expert_frame "${motion_expert_frame}"
