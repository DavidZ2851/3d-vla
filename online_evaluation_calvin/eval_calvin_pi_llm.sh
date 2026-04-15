#!/bin/bash
#SBATCH --job-name=eval_calvin_pi_llm
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain

merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

seed=0
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/calvin/qwen_pi0_llm_actor_3d-Calvin10th-C1024-B16-lr3e-5-tristage_flower-H1-pi0-qwen3vl-finetuned_false/last.pth
base_log_dir=eval_logs/CALVIN/qwen_pi_llm/
save_video=false

model_type=qwen_pi0_llm_actor_3d
pred_len=10
pre_tokenize=true
custom_img_size=224
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
use_multiscale=false
embedding_dim=1024
num_attn_heads=16
num_vis_instr_attn_layers=3
num_shared_attn_layers=16
relative_action=true
rotation_format=euler
denoise_timesteps=5
denoise_model=pi0

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --merged_config_file $merged_config_file \
    --task_config_file $task_config_file \
    --ann_config_file $ann_config_file \
    --seed $seed \
    --checkpoint $checkpoint \
    --base_log_dir $base_log_dir \
    --save_video $save_video \
    --model_type $model_type \
    --pred_len $pred_len \
    --pre_tokenize $pre_tokenize \
    --custom_img_size $custom_img_size \
    --backbone $backbone \
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
    --fps_subsampling_factor $fps_subsampling_factor \
    --use_multiscale $use_multiscale \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_shared_attn_layers $num_shared_attn_layers \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
