#!/bin/bash
#SBATCH --job-name=eval_libero_10
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err
#SBATCH --constraint="L40S|L40|A100_80GB"

libero_root=/data/user_data/ziyinx/LIBERO

# Task/eval arguments
# Runs all LIBERO suites sequentially.
suites=(libero_10)
task_order_index=0
task_id=0
all_tasks=true
num_eval=50
max_steps=600
img_h=224
img_w=224
seed=2
libero_controller=OSC_POSE
libero_tcp_target_mode=absolute
libero_target_pos_tol=0.001
libero_target_ori_tol=0.01
libero_target_max_steps=250

# Output
exp=LIBERO
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/libero/flower_actor_3d-Libero-C1024-B16-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-job7856005/80000.pth
save_video=true
video_fps=20
video_env_index=0

# Circle fake-policy test mode
use_circle_policy=false
circle_radius=0.03
circle_omega=0.20
circle_z_bias=0.00
circle_keep_openness=true
circle_fixed_openness=0.02

# Model arguments (keep aligned with training)
model_type=flower_actor_3d
prediction_len=10
num_history=1
pre_tokenize=true
custom_img_size=224
backbone=florence2
num_finetune_vision_layers=0
vision_lora_rank=0
vision_lora_alpha=32
vision_lora_dropout=0.05
vision_lora_num_layers=0
vision_lora_target_modules=attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2
llm_lora_rank=0
llm_lora_alpha=128
llm_lora_dropout=0.05
llm_lora_num_layers=0
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

ngpus=1

for benchmark in "${suites[@]}"; do
    output_file=eval_logs/$exp/$(basename "$(dirname "$checkpoint")")/seed$seed/$benchmark/eval.json
    video_dir=eval_logs/$exp/videos/seed$seed/$benchmark
    echo "[eval] Starting benchmark=$benchmark num_eval=$num_eval all_tasks=$all_tasks"

    torchrun --nproc_per_node $ngpus --master_port $RANDOM \
        online_evaluation_libero/evaluate_policy.py \
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
        --save_video "$save_video" \
        --video_dir "$video_dir" \
        --video_fps "$video_fps" \
        --video_env_index "$video_env_index" \
        --use_circle_policy "$use_circle_policy" \
        --circle_radius "$circle_radius" \
        --circle_omega "$circle_omega" \
        --circle_z_bias "$circle_z_bias" \
        --circle_keep_openness "$circle_keep_openness" \
        --circle_fixed_openness "$circle_fixed_openness" \
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
        --libero_controller "$libero_controller" \
        --libero_tcp_target_mode "$libero_tcp_target_mode" \
        --libero_target_pos_tol "$libero_target_pos_tol" \
        --libero_target_ori_tol "$libero_target_ori_tol" \
        --libero_target_max_steps "$libero_target_max_steps" \
        --denoise_timesteps "$denoise_timesteps" \
        --denoise_model "$denoise_model"
done
