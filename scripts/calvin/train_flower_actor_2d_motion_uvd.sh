#!/usr/bin/env bash
#SBATCH --job-name=calvin_2dfa_motionuvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

# LaMP-style fully-2D CALVIN VLA with the UVD (pixel+depth) motion expert.
#
# Contrast with train_flower_actor_motion_latent_uvd.sh, which keeps the full 3D
# stack (lifted point clouds, world-XYZ token positions, RoPE-3D in both the
# scene<->motion fusion and the action decoder) and is UVD only inside the
# expert. Here NOTHING is lifted to 3D and there is no 3D RoPE anywhere:
#   * Encoder2DFA tags scene tokens with camera-id + 2D sinusoidal embeddings;
#   * TransformerHead2D predicts EEF actions with rotary_pe=False;
#   * scene<->motion fusion is gated CrossAttentionNoRoPE (no positions at all);
#   * the motion latent enters only through the scene tokens.
# The point cloud is read solely to look up the UVD expert's query depth (it
# needs (u, v, depth) by construction) and never reaches a learnable layer.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
DATA_PATH="${DATA_PATH:-/data/group_data/katefgroup-ssd/world_model/ziyinx}"
mkdir -p "/data/user_data/${USER}/logs"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# CALVIN train zarr now lives on the SSD tier; override with CALVIN_DATA_ROOT.
CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/data/group_data/katefgroup-ssd/world_model/ziyinx}"
train_data_dir="${TRAIN_DATA_DIR:-${CALVIN_DATA_ROOT}/calvin_zarr/train_rechunked8.zarr}"
eval_data_dir="${EVAL_DATA_DIR:-/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr}"
train_instructions="${TRAIN_INSTRUCTIONS:-instructions/calvin/train_instructions.json}"
val_instructions="${VAL_INSTRUCTIONS:-instructions/calvin/val_instructions.json}"

dataset="${DATASET:-CalvinContact10th}"
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

model_type="${MODEL_TYPE:-flower_actor_2d_motion_uvd}"
bimanual="${BIMANUAL:-false}"
keypose_only="${KEYPOSE_ONLY:-false}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
# CALVIN scene cameras tagged with camera-id embeddings: front + wrist.
num_camera_ids="${NUM_CAMERA_IDS:-2}"

backbone="${BACKBONE:-florence2}"
finetune_backbone="${FINETUNE_BACKBONE:-true}"
finetune_text_encoder="${FINETUNE_TEXT_ENCODER:-false}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"

C="${C:-896}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_history="${NUM_HISTORY:-1}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"

workspace_normalizer_buffer="${WORKSPACE_NORMALIZER_BUFFER:-0.0}"
rotation_format="${ROTATION_FORMAT:-euler}"
relative_action="${RELATIVE_ACTION:-true}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2_uvd.yaml}"
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/baseline_calvin_florence2_uvd/checkpoint_step_14000.pt}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
motion_finetune="${MOTION_FINETUNE:-true}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"

main_dir="${MAIN_DIR:-calvin}"
base_log_dir="${BASE_LOG_DIR:-${REPO_ROOT}/train_logs}"
run_log_dir="${RUN_LOG_DIR:-${model_type}-${dataset}-C${C}-B${B}-lr${lr}-${lr_scheduler}-H${num_history}-${denoise_model}-${backbone}-motionuvd_finetuned_${motion_finetune}-job${SLURM_JOBID:-local}}"
checkpoint="${CHECKPOINT:-${base_log_dir}/${main_dir}/${run_log_dir}/last.pth}"
finetune="${FINETUNE:-false}"

ngpus="${NGPUS:-4}"
master_port="${MASTER_PORT:-$((10000 + ${SLURM_JOBID:-12345} % 50000))}"

cd "${REPO_ROOT}"
echo "[train] model=${model_type} C=${C} B=${B} motion=${motion_expert_checkpoint}"
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
    --num_camera_ids "${num_camera_ids}" \
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
    --motion_scene_fusion "${motion_scene_fusion}"
