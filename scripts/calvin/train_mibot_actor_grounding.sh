#!/bin/bash
#SBATCH --job-name=calvin_contact_prediction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_40GB|A100_80GB"

export WANDB_MODE=online   # or "offline" / "disabled"
export WANDB_API_KEY=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
export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain

# ---- Data paths (edit as needed) ----
DATA_PATH=/data/user_data/ziyinx/datasets
train_data_dir=$DATA_PATH/zarr_datasets/calvin_zarr/train_rechunked8.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr
train_instructions=instructions/calvin/train_instructions.json
val_instructions=instructions/calvin/val_instructions.json

# ---- Dataset / loader ----
# CalvinContact10th reads '3d_contact_point' from the zarr if present.
# Falls back to None gracefully when the field is absent (no-op for both variants).
dataset=CalvinContact10th
num_workers=8
memory_limit=8
B=16
B_val=16
chunk_size=8
lv2_batch_size=1

# ---- Training ----
val_freq=5000
eval_only=false
lr=3e-5
lr_scheduler=tristage_flower
wd=0.05
train_iters=100000
use_compile=false
use_ema=false

# ---- Model ----
# Choose ONE of the two grounding variants:
#   qwen_flower_actor_3d_grounding_cond  -- conditions on contact point as extra context token
#   qwen_flower_actor_3d_grounding_pred  -- predicts contact point as auxiliary training goal
model_type=qwen_flower_actor_3d_grounding_pred
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=qwen3vl
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=1
use_multiscale=false

# ---- Grounding loss weight (variant 2 only; ignored by variant 1) ----
grounding_loss_weight=10.0

# ---- LoRA Configuration ----
num_finetune_vision_layers=24
vision_lora_rank=0
vision_lora_alpha=32
vision_lora_dropout=0.05
vision_lora_num_layers=0
vision_lora_target_modules=attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2
llm_lora_rank=64
llm_lora_alpha=128
llm_lora_dropout=0.05
llm_lora_num_layers=36
llm_lora_target_modules=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj

C=1024
num_attn_heads=16
num_vis_instr_attn_layers=3  # unused by Qwen3VL, only applied to CLIP.
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=euler
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=calvin
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone-job${SLURM_JOBID}
# train from scratch:
checkpoint=$DATA_PATH/train_logs/${main_dir}/${run_log_dir}/last.pth
finetune=false

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
    --use_multiscale $use_multiscale \
    --grounding_loss_weight $grounding_loss_weight \
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
    --vision_lora_rank $vision_lora_rank \
    --vision_lora_alpha $vision_lora_alpha \
    --vision_lora_dropout $vision_lora_dropout \
    --vision_lora_num_layers $vision_lora_num_layers \
    --vision_lora_target_modules $vision_lora_target_modules \
    --llm_lora_rank $llm_lora_rank \
    --llm_lora_alpha $llm_lora_alpha \
    --llm_lora_dropout $llm_lora_dropout \
    --llm_lora_num_layers $llm_lora_num_layers \
    --llm_lora_target_modules $llm_lora_target_modules
