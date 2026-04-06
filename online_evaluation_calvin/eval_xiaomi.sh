#!/bin/bash
#SBATCH --job-name=eval_xiaomi_calvin
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

model_path=/home/ziyinx/Xiaomi-Robotics-0-Calvin
split=abc
crop_ratio=0.95
pred_len=10
seed=0
# Shared evaluate_policy loop with CALVIN multistep_sequences (same as other policies).
# Optional: pass --eval_sequences_file /path/to.json to override.
action_exec=relative_array
base_log_dir=eval_logs/CALVIN/xiaomi_calvin_shared_rollout/
save_video=false

# This script evaluates the original Xiaomi action model directly.
# Qwen actor vision/LLM LoRA flags are configured in eval_calvin.sh instead.

ngpus=1

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/eval_xiaomi_calvin.py \
    --merged_config_file $merged_config_file \
    --task_config_file $task_config_file \
    --ann_config_file $ann_config_file \
    --model_path $model_path \
    --split $split \
    --crop_ratio $crop_ratio \
    --pred_len $pred_len \
    --seed $seed \
    --action_exec $action_exec \
    --base_log_dir $base_log_dir \
    --save_video $save_video
