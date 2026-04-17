#!/bin/bash
#SBATCH --job-name=eval_libero
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

libero_root=/data/user_data/ziyinx/LIBERO

# Task/eval arguments
benchmark=libero_10
task_order_index=0
task_id=-1
all_tasks=true
num_eval=20
max_steps=600
img_h=128
img_w=128
seed=0

# Output
exp=libero_eval
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/libero/your_run/last.pth
output_file=eval_logs/$exp/$(basename "$(dirname "$checkpoint")")/seed$seed/eval.json

# Model arguments (keep aligned with training)
model_type=qwen_flower_actor_3d_grounding_cond_llm_visual
prediction_len=1
num_history=1
pre_tokenize=true
custom_img_size=128
backbone=qwen3vl
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
fps_subsampling_factor=1
embedding_dim=1024
num_attn_heads=16
num_vis_instr_attn_layers=3
num_shared_attn_layers=16
use_multiscale=false
gradient_checkpointing=false
relative_action=true
rotation_format=axisangle
denoise_timesteps=5
denoise_model=rectified_flow

python online_evaluation_libero/evaluate_policy.py \
    --libero_root "$libero_root" \
    --checkpoint "$checkpoint" \
    --benchmark "$benchmark" \
    --task_order_index "$task_order_index" \
    --task_id "$task_id" \
    --all_tasks "$all_tasks" \
    --num_eval "$num_eval" \
    --max_steps "$max_steps" \
    --img_h "$img_h" \
    --img_w "$img_w" \
    --seed "$seed" \
    --output_file "$output_file" \
    --model_type "$model_type" \
    --prediction_len "$prediction_len" \
    --num_history "$num_history" \
    --pre_tokenize "$pre_tokenize" \
    --custom_img_size "$custom_img_size" \
    --backbone "$backbone" \
    --num_finetune_vision_layers "$num_finetune_vision_layers" \
    --vision_lora_rank "$vision_lora_rank" \
    --vision_lora_alpha "$vision_lora_alpha" \
    --vision_lora_dropout "$vision_lora_dropout" \
    --vision_lora_num_layers "$vision_lora_num_layers" \
    --vision_lora_target_modules "$vision_lora_target_modules" \
    --llm_lora_rank "$llm_lora_rank" \
    --llm_lora_alpha "$llm_lora_alpha" \
    --llm_lora_dropout "$llm_lora_dropout" \
    --llm_lora_num_layers "$llm_lora_num_layers" \
    --llm_lora_target_modules "$llm_lora_target_modules" \
    --fps_subsampling_factor "$fps_subsampling_factor" \
    --embedding_dim "$embedding_dim" \
    --num_attn_heads "$num_attn_heads" \
    --num_vis_instr_attn_layers "$num_vis_instr_attn_layers" \
    --num_shared_attn_layers "$num_shared_attn_layers" \
    --use_multiscale "$use_multiscale" \
    --gradient_checkpointing "$gradient_checkpointing" \
    --relative_action "$relative_action" \
    --rotation_format "$rotation_format" \
    --denoise_timesteps "$denoise_timesteps" \
    --denoise_model "$denoise_model"
