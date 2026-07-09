#!/bin/bash
#SBATCH --job-name=aloha_rot6d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --partition=katef
#SBATCH --qos=katef_qos
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="A100_40GB"

# Train flower_actor_3d on the single-arm ALOHA real-world zarr produced by
# data_processing/aloha_to_zarr.py. Each zarr row is one datapoint with
# T_his=1 and T_action=15; chunk_size below is only the loader grouping window.
# Front/back Azure Kinects are 3D cameras; wrist is RGB-only 2D.
# Offline training only — no online evaluation.

export WANDB_MODE=online
export WANDB_API_KEY=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

DATA_PATH=/data/user_data/ziyinx/datasets
ZARR_ROOT=${ZARR_ROOT:-/data/user_data/ziyinx/datasets/zarr_datasets/aloha_singlearm_zarr}
train_data_dir=${TRAIN_DATA_DIR:-$ZARR_ROOT/train.zarr}
eval_data_dir=${EVAL_DATA_DIR:-$ZARR_ROOT/val.zarr}
train_instructions=${TRAIN_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}
val_instructions=${VAL_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}

dataset=Aloha
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

# Training/testing arguments
val_freq=5000
eval_only=false
lr=5e-5
lr_scheduler=tristage_flower
backbone_lr=5e-5
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=200000
use_compile=false
use_ema=false
grounding_dropout_prob=0.6

# Model arguments
model_type=flower_actor_3d
bimanual=false
keypose_only=false
pre_tokenize=true
# ALOHA RGB/depth are 224x398 (preserves 720:1280 aspect). Leave custom_img_size
# unset so the preprocessor keeps native resolution; the modified Florence2 vision
# tower accepts non-square inputs.
custom_img_size=None

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=256
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=rot6d
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=aloha
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone-job${SLURM_JOBID}
checkpoint=$DATA_PATH/train_logs/${main_dir}/${run_log_dir}/last.pth
finetune=false

ngpus=8
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

torchrun --nproc_per_node $ngpus --master_port $master_port \
    /data/user_data/ziyinx/analogical_manipulation_clean/main.py \
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
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
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
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
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
