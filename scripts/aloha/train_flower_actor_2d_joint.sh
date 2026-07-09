#!/bin/bash
#SBATCH --job-name=aloha_2d_joint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"
#
# Train flower_actor_2d_joint on the COMBINED ALOHA zarr (bimanual + single-arm,
# 8 tasks). This is the pure-2D visual-language joint-action baseline:
#
#   * Cameras: the two head Azure Kinects (front, back) + the wrist camera are all
#     fed as 2D SCENE cameras (num_camera_ids=3). The 2DFA encoder tags each with a
#     learnable camera-id embedding and DISCARDS depth / point clouds -- there is no
#     3D RoPE between scene and action (joint control needs no scene<->action 3D
#     geometry). AlohaTrainTester routes the wrist as a third rgb3d scene camera
#     automatically for this model_type (see _feed_wrist_as_scene).
#   * Action: anchor-relative delta JOINT angles (6 arm joints, radians) + a binary
#     gripper command, anchored to the current joint-angle proprioception. No EEF
#     chain-of-thought, no MotionExpert (both removed from FlowerActor2DJoint).
#   * Normalizer: the per-joint arm envelope is read from the combined zarr's
#     aloha_joint_rel_action_stats.json sidecar (AlohaJointTrainer.USE_DATASET_JOINT_STATS).
#
# ALOHA arm joints are stored in DEGREES; AlohaJointDataset converts the 6 arm
# joints to RADIANS at load time so the radian normalizer matches. Offline training
# only -- no online evaluation hook.

set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT=${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}
BASE_LOG_DIR=${BASE_LOG_DIR:-/data/user_data/ziyinx/datasets/train_logs}
# Combined bimanual+single-arm ALOHA zarr (8 tasks) on the SSD world_model store.
ZARR_ROOT=${ZARR_ROOT:-/data/group_data/katefgroup-ssd/world_model/ziyinx/aloha_combined_zarr}

train_data_dir=${TRAIN_DATA_DIR:-$ZARR_ROOT/train.zarr}
eval_data_dir=${EVAL_DATA_DIR:-$ZARR_ROOT/val.zarr}
train_instructions=${TRAIN_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}
val_instructions=${VAL_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}

# AlohaJoint delivers rgb front/back/wrist + arm-joint targets/state; the trainer
# (dataset name contains 'aloha'+'joint' -> AlohaJointTrainer) threads them through.
dataset=AlohaJoint
num_workers=4
memory_limit=6
B=${BATCH_SIZE:-8}
B_val=${BATCH_SIZE_VAL:-8}
chunk_size=8

val_freq=5000
eval_only=false
lr=${LR:-2e-5}
lr_scheduler=tristage_flower
backbone_lr=${BACKBONE_LR:-2e-5}
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=${TRAIN_ITERS:-100000}
use_compile=false
use_ema=false
grounding_dropout_prob=0.0

# 2D visual-language joint baseline (no MotionExpert, no EEF chain-of-thought).
model_type=flower_actor_2d_joint
# Three 2D scene cameras (front, back, wrist) each get their own camera-id
# embedding. The default (2) fits only front/back; ALOHA needs 3.
num_camera_ids=3
bimanual=false
keypose_only=false
pre_tokenize=true
# ALOHA RGB is 224x398 (native 720:1280 aspect); leave custom_img_size unset so the
# preprocessor keeps native resolution (the Florence2 tower accepts non-square).
custom_img_size=None

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=${EMBEDDING_DIM:-256}
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=4  # unused by the 2D joint baseline (EEF head removed)

# Joint head. Single ALOHA arm: joint_action_dim=6 arm joints, gripper_dim=1,
# joint_proprio_dim=7 (6 joints + binary gripper command). These differ from the
# CALVIN defaults (7/8) and MUST be set for ALOHA.
joint_action_dim=6
joint_proprio_dim=7
gripper_dim=1
joint_embedding_dim=${JOINT_EMBEDDING_DIM:-256}
joint_num_attn_heads=${JOINT_NUM_ATTN_HEADS:-16}
joint_num_layers=${JOINT_NUM_LAYERS:-12}
joint_loss_weight=${JOINT_LOSS_WEIGHT:-1.0}
joint_action_weight=${JOINT_ACTION_WEIGHT:-30.0}

workspace_normalizer_buffer=0.0
rotation_format=rot6d   # required by AlohaJointDataset (EEF poses laid out as rot6d)
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=aloha
run_log_dir=${RUN_LOG_DIR:-$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-job${SLURM_JOBID:-local}}
checkpoint=${CHECKPOINT:-$BASE_LOG_DIR/${main_dir}/${run_log_dir}/last.pth}
if [[ "${checkpoint}" == "None" ]]; then
    checkpoint=""
fi
finetune=${FINETUNE:-false}

if [[ ! -d "${train_data_dir}" || ! -d "${eval_data_dir}" ]]; then
    echo "ERROR: missing ALOHA combined joint zarr stores:" >&2
    echo "  train_data_dir=${train_data_dir}" >&2
    echo "  eval_data_dir=${eval_data_dir}" >&2
    exit 1
fi
if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    checkpoint=""
fi

echo "[train] model        = ${model_type}"
echo "[train] dataset      = ${dataset}  (zarr=${ZARR_ROOT})"
echo "[train] cameras      = front+back+wrist as 2D scene cams (num_camera_ids=${num_camera_ids})"
echo "[train] joint dims   = action_dim=${joint_action_dim} proprio_dim=${joint_proprio_dim} gripper_dim=${gripper_dim}"
echo "[train] output dir   = ${BASE_LOG_DIR}/${main_dir}/${run_log_dir}"

ngpus=${NGPUS:-8}
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))
custom_img_size_args=()
if [[ -n "$custom_img_size" && "$custom_img_size" != "None" && "$custom_img_size" != "none" ]]; then
    custom_img_size_args=(--custom_img_size "$custom_img_size")
fi

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
    --base_log_dir $BASE_LOG_DIR \
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
    "${custom_img_size_args[@]}" \
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
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --finetune $finetune \
    --num_finetune_vision_layers $num_finetune_vision_layers \
    --vision_lr_scale $vision_lr_scale \
    --projector_lr_scale $projector_lr_scale \
    --grounding_dropout_prob $grounding_dropout_prob
