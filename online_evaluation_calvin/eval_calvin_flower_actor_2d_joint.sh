#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_2d_joint
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"

# Online CALVIN evaluation for flower_actor_2d_joint: the 2D visual-language
# joint-action baseline (no motion expert, no 3D pcd). Policy args mirror the
# training script scripts/calvin/train_flower_actor_2d_joint.sh. The model emits
# absolute arm-joint targets executed in POSITION_CONTROL via calvin_env's
# "joint_abs" action (same joint path as flower_actor_3d_motion_latent_joint).

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

model_type="${MODEL_TYPE:-flower_actor_2d_joint}"
seed="${SEED:-0}"
# Default points at the most recent finished 2d_joint run; override CHECKPOINT
# with the final checkpoint once training completes (e.g. .../<run>/100000.pth).
checkpoint="${CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/flower_actor_2d_joint-CalvinJoint10th-C800-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-job8315117/90000.pth}"
checkpoint_alias="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${checkpoint}")")}"
base_log_dir="${BASE_LOG_DIR:-${REPO_ROOT}/eval_logs/CALVIN/flower_actor_2d_joint/${checkpoint_alias}/seed${seed}}"
save_video="${SAVE_VIDEO:-false}"

pred_len="${PRED_LEN:-10}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
backbone="${BACKBONE:-florence2}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-0}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"
use_multiscale="${USE_MULTISCALE:-false}"
embedding_dim="${EMBEDDING_DIM:-800}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-4}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

# Joint-angle decoder head (must match training).
joint_action_dim="${JOINT_ACTION_DIM:-7}"
joint_proprio_dim="${JOINT_PROPRIO_DIM:-8}"
joint_num_layers="${JOINT_NUM_LAYERS:-8}"
joint_embedding_dim="${JOINT_EMBEDDING_DIM:-800}"
joint_num_attn_heads="${JOINT_NUM_ATTN_HEADS:-16}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "[eval-2d-joint] checkpoint=${checkpoint}"
echo "[eval-2d-joint] base_log_dir=${base_log_dir}"
echo "[eval-2d-joint] model=${model_type} backbone=${backbone} C=${embedding_dim} pred_len=${pred_len}"

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
    --joint_action_dim "${joint_action_dim}" \
    --joint_proprio_dim "${joint_proprio_dim}" \
    --joint_num_layers "${joint_num_layers}" \
    --joint_embedding_dim "${joint_embedding_dim}" \
    --joint_num_attn_heads "${joint_num_attn_heads}"
