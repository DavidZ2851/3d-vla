#!/bin/bash
#SBATCH --job-name=calvin_flower_cond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

export WANDB_MODE=online   # or "offline" / "disabled"
export WANDB_API_KEY=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

DATA_PATH=/data/user_data/ziyinx/datasets
train_data_dir=$DATA_PATH/zarr_datasets/calvin_zarr/train_rechunked8.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr
train_instructions=instructions/calvin/train_instructions.json
val_instructions=instructions/calvin/val_instructions.json

dataset=CalvinContact10th
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

# Training/testing arguments, change these for HPT
val_freq=5000
eval_only=false
lr=2e-5
lr_scheduler=tristage_flower
backbone_lr=2e-5
num_finetune_vision_layers=0  # unused by Florence2, kept for CLI parity with mibot script
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=60000
use_compile=false
use_ema=true

# Model arguments, change (some of) these for new architectures
# flower_actor_3d: no contact-point conditioning
# flower_actor_3d_grounding_cond: condition decoder on 3D contact point when provided
model_type=flower_actor_3d_grounding_cond
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=1024
num_attn_heads=16
num_vis_instr_attn_layers=3  # unused by Florence2, only applied to CLIP.
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=euler
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=calvin
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone
# train from scratch:
checkpoint=$DATA_PATH/train_logs/${main_dir}/${run_log_dir}/last.pth
finetune=false
# load finetune_ckpt:
# checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/geniesim3.0/flower_actor_3d-GenieSim3.0-C1024-B8-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_false-fullpretrain/last.pth
# finetune=true

ngpus=4
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
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --finetune $finetune \
    --num_finetune_vision_layers $num_finetune_vision_layers \
    --vision_lr_scale $vision_lr_scale \
    --projector_lr_scale $projector_lr_scale
