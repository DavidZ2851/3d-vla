#!/bin/bash
#SBATCH --job-name=aloha_head_rgbd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

set -euo pipefail

# Train flower_actor_3d on ALOHA using only the two head RGBD cameras:
#   3D cameras: front, back
#   ignored:    wrist RGB
#
# Defaults:
#   train: /data/user_data/ziyinx/datasets/zarr_datasets/aloha/train.zarr
#   val:   /data/group_data/katefgroup/VLA/zarr_datasets/aloha_zarr/val.zarr

export WANDB_MODE=${WANDB_MODE:-online}

REPO_ROOT=${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}
ZARR_ROOT=${ZARR_ROOT:-/data/user_data/ziyinx/datasets/zarr_datasets/aloha}
VAL_ZARR=${VAL_ZARR:-/data/group_data/katefgroup/VLA/zarr_datasets/aloha_zarr/val.zarr}
INSTR_ROOT=${INSTR_ROOT:-/data/group_data/katefgroup/VLA/zarr_datasets/aloha_zarr}

train_data_dir=${TRAIN_DATA_DIR:-${ZARR_ROOT}/train.zarr}
eval_data_dir=${EVAL_DATA_DIR:-${VAL_ZARR}}
train_instructions=${TRAIN_INSTRUCTIONS:-${INSTR_ROOT}/instructions.json}
val_instructions=${VAL_INSTRUCTIONS:-${INSTR_ROOT}/instructions.json}

dataset=AlohaHeadRGBD
num_workers=${NUM_WORKERS:-4}
memory_limit=${MEMORY_LIMIT:-6}
B=${BATCH_SIZE:-32}
B_val=${BATCH_SIZE_VAL:-32}
chunk_size=${CHUNK_SIZE:-8}

val_freq=${VAL_FREQ:-1000}
eval_only=${EVAL_ONLY:-false}
lr=${LR:-2e-5}
lr_scheduler=${LR_SCHEDULER:-tristage_flower}
backbone_lr=${BACKBONE_LR:-2e-5}
num_finetune_vision_layers=${NUM_FINETUNE_VISION_LAYERS:-0}
vision_lr_scale=${VISION_LR_SCALE:-0.2}
projector_lr_scale=${PROJECTOR_LR_SCALE:-0.4}
wd=${WEIGHT_DECAY:-0.05}
train_iters=${TRAIN_ITERS:-200000}
use_compile=${USE_COMPILE:-false}
use_ema=${USE_EMA:-false}
grounding_dropout_prob=${GROUNDING_DROPOUT_PROB:-0.6}

model_type=flower_actor_3d
bimanual=false
keypose_only=false
pre_tokenize=true

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=${EMBEDDING_DIM:-256}
num_attn_heads=${NUM_ATTN_HEADS:-16}
num_vis_instr_attn_layers=${NUM_VIS_INSTR_ATTN_LAYERS:-3}
num_history=1
num_shared_attn_layers=${NUM_SHARED_ATTN_LAYERS:-16}

workspace_normalizer_buffer=0.0
rotation_format=rot6d
relative_action=true
denoise_timesteps=${DENOISE_TIMESTEPS:-5}
denoise_model=rectified_flow

base_log_dir=${BASE_LOG_DIR:-${REPO_ROOT}/train_logs}
main_dir=aloha_head_rgbd
run_log_dir=${RUN_LOG_DIR:-$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone-job${SLURM_JOBID:-local}}
checkpoint=${CHECKPOINT:-${base_log_dir}/${main_dir}/${run_log_dir}/last.pth}
finetune=${FINETUNE:-false}

ngpus=${NGPUS:-1}
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

${PYTHON:-python} -c "import sys, zarr; p=sys.argv[1]; required={'rgb_front','rgb_back','depth_front','depth_back','proprioception','rel_action','instr_id'}; keys=set(zarr.open(p, mode='r').array_keys()); missing=sorted(required-keys); print(f'[aloha_head_rgbd] train zarr keys: {sorted(keys)}'); sys.exit('[aloha_head_rgbd] missing required train zarr keys: '+str(missing) if missing else 0)" "$train_data_dir"

torchrun --nproc_per_node "$ngpus" --master_port "$master_port" \
    "${REPO_ROOT}/main.py" \
    --train_data_dir "$train_data_dir" \
    --eval_data_dir "$eval_data_dir" \
    --train_instructions "$train_instructions" \
    --val_instructions "$val_instructions" \
    --dataset "$dataset" \
    --num_workers "$num_workers" \
    --memory_limit "$memory_limit" \
    --batch_size "$B" \
    --batch_size_val "$B_val" \
    --chunk_size "$chunk_size" \
    --base_log_dir "$base_log_dir" \
    --exp_log_dir "$main_dir" \
    --run_log_dir "$run_log_dir" \
    --checkpoint "$checkpoint" \
    --val_freq "$val_freq" \
    --eval_only "$eval_only" \
    --lr "$lr" \
    --lr_scheduler "$lr_scheduler" \
    --backbone_lr "$backbone_lr" \
    --wd "$wd" \
    --train_iters "$train_iters" \
    --use_compile "$use_compile" \
    --use_ema "$use_ema" \
    --model_type "$model_type" \
    --bimanual "$bimanual" \
    --keypose_only "$keypose_only" \
    --pre_tokenize "$pre_tokenize" \
    --backbone "$backbone" \
    --finetune_backbone "$finetune_backbone" \
    --finetune_text_encoder "$finetune_text_encoder" \
    --fps_subsampling_factor "$fps_subsampling_factor" \
    --embedding_dim "$C" \
    --num_attn_heads "$num_attn_heads" \
    --num_vis_instr_attn_layers "$num_vis_instr_attn_layers" \
    --num_history "$num_history" \
    --num_shared_attn_layers "$num_shared_attn_layers" \
    --workspace_normalizer_buffer "$workspace_normalizer_buffer" \
    --relative_action "$relative_action" \
    --rotation_format "$rotation_format" \
    --denoise_timesteps "$denoise_timesteps" \
    --denoise_model "$denoise_model" \
    --finetune "$finetune" \
    --num_finetune_vision_layers "$num_finetune_vision_layers" \
    --vision_lr_scale "$vision_lr_scale" \
    --projector_lr_scale "$projector_lr_scale" \
    --grounding_dropout_prob "$grounding_dropout_prob"
