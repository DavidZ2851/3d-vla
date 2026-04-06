#!/bin/bash
#SBATCH --job-name=libero_mibot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_40GB"

export WANDB_MODE=online   # or "offline" / "disabled"
export WANDB_API_KEY=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

# ---- VLM Model path ----
export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain

# ---- Data paths (edit as needed) ----
DATA_PATH=/data/user_data/ziyinx/datasets
train_data_dir=$DATA_PATH/zarr_datasets/libero_zarr/train_rechunked8.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/val_rechunked8.zarr
train_instructions=instructions/libero/instructions.json
val_instructions=instructions/libero/val_instructions.json

# ---- Dataset / loader ----
dataset=Libero
num_workers=8
memory_limit=8
B=64
B_val=64
chunk_size=8
lv2_batch_size=1

# ---- Training ----
val_freq=2000
eval_only=false
lr=1e-4
lr_scheduler=tristage_flower
backbone_lr=1e-6
num_finetune_vision_layers=0
vision_lora_rank=16
vision_lora_alpha=32
vision_lora_dropout=0.05
vision_lora_num_layers=6
vision_lora_target_modules=attn.qkv,attn.proj
llm_lora_rank=16
llm_lora_alpha=32
llm_lora_dropout=0.05
llm_lora_num_layers=4
llm_lora_target_modules=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj
vision_lora_lr_scale=0.15
llm_lora_lr_scale=0.05
merger_lr_scale=0.4
wd=1e-10
train_iters=200000
use_compile=false
use_ema=false

# ---- Model ----
model_type=qwen_flower_actor_3d
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=128

backbone=qwen3vl
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=1

C=1024
num_attn_heads=16
num_vis_instr_attn_layers=3  # unused by Qwen3VL, only applied to CLIP.
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=axisangle
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=libero
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-visLoraR${vision_lora_rank}-visL${vision_lora_num_layers}-llmLoraR${llm_lora_rank}-llmL${llm_lora_num_layers}-job${SLURM_JOBID}
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
    --llm_lora_target_modules $llm_lora_target_modules \
    --vision_lora_lr_scale $vision_lora_lr_scale \
    --llm_lora_lr_scale $llm_lora_lr_scale \
    --merger_lr_scale $merger_lr_scale
