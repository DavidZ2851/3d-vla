#!/bin/bash
#SBATCH --job-name=rw2_2d_joint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"
#SBATCH --exclude=babel-q9-28

set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-online}

REPO_ROOT=${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}
ZARR_ROOT=${ZARR_ROOT:-/data/user_data/ziyinx/datasets/zarr_datasets/robotwin2_zarr_joint}
train_data_dir=${TRAIN_DATA_DIR:-${ZARR_ROOT}/train_rechunked8.zarr}
eval_data_dir=${EVAL_DATA_DIR:-${ZARR_ROOT}/val_rechunked8.zarr}
train_instructions=${REPO_ROOT}/instructions/robotwin2/instructions_full.json
val_instructions=${REPO_ROOT}/instructions/robotwin2/instructions_full.json

# RoboTwin2 joint dataset: bimanual, 6 arm joints + 1 binary gripper per arm.
dataset=RoboTwin2Joint
num_workers=4
memory_limit=6
B=${BATCH_SIZE:-8}
B_val=${BATCH_SIZE_VAL:-8}
chunk_size=8

val_freq=5000
eval_only=false
lr=${LR:-4e-5}
lr_scheduler=tristage_flower
backbone_lr=${BACKBONE_LR:-4e-5}
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=${TRAIN_ITERS:-200000}
use_compile=false
use_ema=false
grounding_dropout_prob=0.0

# 2D ablation: no MotionExpert, no EEF chain-of-thought (joint head conditions
# only on 2DFA visual tokens + instruction).
model_type=flower_actor_2d_joint
# RoboTwin2 feeds 3 scene cameras (head/left/right). The 2D ablation tags each
# with a learnable camera-id embedding, so size that embedding to 3 (default 2
# fits only CALVIN/LIBERO).
num_camera_ids=3
bimanual=true
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=${EMBEDDING_DIM:-800}
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=${NUM_HISTORY:-1}
num_shared_attn_layers=4  # unused by the 2D joint baseline (EEF head removed)

# Joint head. joint_action_dim=6 arm joints, gripper_dim=1, joint_proprio_dim=7
# (6 joints + binary gripper command). These DIFFER from the CALVIN defaults
# (7/8) and MUST be set for RoboTwin2.
joint_action_dim=6
joint_proprio_dim=7
gripper_dim=1
joint_embedding_dim=${JOINT_EMBEDDING_DIM:-800}
joint_num_attn_heads=${JOINT_NUM_ATTN_HEADS:-16}
joint_num_layers=${JOINT_NUM_LAYERS:-12}
joint_loss_weight=${JOINT_LOSS_WEIGHT:-1.0}
joint_action_weight=${JOINT_ACTION_WEIGHT:-30.0}

workspace_normalizer_buffer=0.0
rotation_format=${ROTATION_FORMAT:-quat_wxyz}
relative_action=true
robotwin2_relative_rotation=${ROBOTWIN2_RELATIVE_ROTATION:-true}
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=robotwin2
base_log_dir=${BASE_LOG_DIR:-${REPO_ROOT}/train_logs}
run_log_dir=${RUN_LOG_DIR:-$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-job${SLURM_JOBID:-local}}
run_dir=${base_log_dir}/${main_dir}/${run_log_dir}
checkpoint=${CHECKPOINT:-}
if [[ "${checkpoint}" == "None" ]]; then
    checkpoint=""
fi
finetune=${FINETUNE:-false}

if [[ ! -d "${train_data_dir}" || ! -d "${eval_data_dir}" ]]; then
    echo "ERROR: missing RoboTwin2 joint zarr stores:" >&2
    echo "  train_data_dir=${train_data_dir}" >&2
    echo "  eval_data_dir=${eval_data_dir}" >&2
    exit 1
fi
if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

echo "[train] model                  = ${model_type}"
echo "[train] checkpoint/resume path = ${checkpoint:-<scratch>}"
echo "[train] output run dir         = ${run_dir}"
echo "[train] joint dims             = action_dim=${joint_action_dim} proprio_dim=${joint_proprio_dim} gripper_dim=${gripper_dim}"

ngpus=${NGPUS:-8}
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

torchrun --nproc_per_node $ngpus --master_port $master_port \
    $REPO_ROOT/main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --memory_limit $memory_limit \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --base_log_dir $base_log_dir \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint "$checkpoint" \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --backbone_lr $backbone_lr \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --custom_img_size $custom_img_size \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --joint_action_dim $joint_action_dim \
    --joint_proprio_dim $joint_proprio_dim \
    --gripper_dim $gripper_dim \
    --joint_num_layers $joint_num_layers \
    --joint_embedding_dim $joint_embedding_dim \
    --joint_num_attn_heads $joint_num_attn_heads \
    --joint_loss_weight $joint_loss_weight \
    --joint_action_weight $joint_action_weight \
    --num_camera_ids $num_camera_ids \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --robotwin2_relative_rotation $robotwin2_relative_rotation \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --finetune $finetune \
    --num_finetune_vision_layers $num_finetune_vision_layers \
    --vision_lr_scale $vision_lr_scale \
    --projector_lr_scale $projector_lr_scale \
    --grounding_dropout_prob $grounding_dropout_prob
