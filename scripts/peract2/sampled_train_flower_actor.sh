#!/bin/bash
#SBATCH --job-name=peract2_sampled_scratch256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"

export WANDB_MODE=online   # or "offline" / "disabled"
export WANDB_API_KEY=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

# ---- Data paths (edit as needed) ----
DATA_PATH=/data/user_data/ziyinx/datasets
train_data_dir=$DATA_PATH/zarr_datasets/Peract2_sampled_zarr/train_rechunked4.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/Peract2_sampled_zarr/val_rechunked4.zarr
train_instructions=instructions/peract2_sampled/instructions.json
val_instructions=instructions/peract2_sampled/instructions.json

# ---- Dataset / loader ----
dataset="Peract2_3dfront"
num_workers=4
memory_limit=8
B=32
B_val=32
chunk_size=4
lv2_batch_size=2

# ---- Training ----
val_freq=2000
eval_only=false
lr=1e-4
lr_scheduler=constant
backbone_lr=1e-5
wd=1e-10
train_iters=600000
use_compile=false
use_ema=false

# ---- Model ----
model_type=flower_actor_3d
bimanual=true
keypose_only=true
pre_tokenize=true
custom_img_size=256

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=256
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=3
num_shared_attn_layers=4

workspace_normalizer_buffer=0.0
rotation_format=quat_xyzw
relative_action=false
denoise_timesteps=5
denoise_model=rectified_flow


main_dir=peract2
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone-job${SLURM_JOBID}
# train from scratch:
checkpoint=$DATA_PATH/train_logs/${main_dir}/${run_log_dir}/last.pth
finetune=false
# load finetune_ckpt:
# checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/geniesim3.0/flower_actor_3d-GenieSim3.0-C256-B8-lr1e-4-constant-H3-rectified_flow-florence2-finetuned_true_smaller/iter220k.pth
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
    --lv2_batch_size $lv2_batch_size \
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
    --finetune $finetune
