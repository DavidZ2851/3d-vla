#!/bin/bash
#SBATCH --job-name=eval_calvin
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

# pretrained VLM backbone path
export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain

# You probably don't need to change these
merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

# Things you may want to change if you train/eval a new model
seed=0
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/calvin/qwen_flower_actor_3d-Calvin10th-C1024-B64-lr1e-4-tristage_flower-H1-rectified_flow-qwen3vl-finetuned_false-job6767753/iter100k.pth
base_log_dir=eval_logs/CALVIN/qwen_flower_actor_3d-Calvin10th-C1024-B64-lr1e-4-tristage_flower-H1-rectified_flow-qwen3vl-finetuned_false-job6767753-100k/
save_video=false

# Things you can change if you customize the model architecture
model_type=qwen_flower_actor_3d
pred_len=10
pre_tokenize=true
custom_img_size=224
backbone=qwen3vl
fps_subsampling_factor=1
embedding_dim=1024
num_attn_heads=16
num_vis_instr_attn_layers=3
num_shared_attn_layers=16
relative_action=true
rotation_format=euler
denoise_timesteps=5
denoise_model=rectified_flow

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
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $embedding_dim \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_shared_attn_layers $num_shared_attn_layers \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
