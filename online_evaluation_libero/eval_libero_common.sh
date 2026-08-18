#!/bin/bash
# Shared helpers for LIBERO / LIBERO-PRO / LIBERO-plus online evaluation.

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
cd "$REPO_ROOT" || exit 1

# Defaults shared across variants.
task_order_index="${task_order_index:-0}"
task_id="${task_id:--1}"
all_tasks="${all_tasks:-true}"
task_start="${task_start:-0}"
task_end="${task_end:--1}"
num_eval="${num_eval:-50}"
max_steps="${max_steps:-600}"
auto_max_steps="${auto_max_steps:-false}"
aggregate_plus_categories="${aggregate_plus_categories:-false}"
img_h="${img_h:-224}"
img_w="${img_w:-224}"
seed="${seed:-0}"
save_video="${save_video:-false}"
video_fps="${video_fps:-20}"
video_env_index="${video_env_index:-0}"
ngpus="${ngpus:-1}"

libero_controller="${libero_controller:-OSC_POSE}"
libero_tcp_target_mode="${libero_tcp_target_mode:-absolute}"
libero_target_pos_tol="${libero_target_pos_tol:-0.001}"
libero_target_ori_tol="${libero_target_ori_tol:-0.01}"
libero_target_max_steps="${libero_target_max_steps:-250}"

prediction_len="${prediction_len:-10}"
num_history="${num_history:-1}"
pre_tokenize="${pre_tokenize:-true}"
custom_img_size="${custom_img_size:-224}"
backbone="${backbone:-florence2}"
num_finetune_vision_layers="${num_finetune_vision_layers:-0}"
vision_lora_rank="${vision_lora_rank:-0}"
vision_lora_alpha="${vision_lora_alpha:-32}"
vision_lora_dropout="${vision_lora_dropout:-0.05}"
vision_lora_num_layers="${vision_lora_num_layers:-0}"
vision_lora_target_modules="${vision_lora_target_modules:-attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}"
llm_lora_rank="${llm_lora_rank:-0}"
llm_lora_alpha="${llm_lora_alpha:-128}"
llm_lora_dropout="${llm_lora_dropout:-0.05}"
llm_lora_num_layers="${llm_lora_num_layers:-0}"
llm_lora_target_modules="${llm_lora_target_modules:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
fps_subsampling_factor="${fps_subsampling_factor:-1}"
num_attn_heads="${num_attn_heads:-16}"
num_vis_instr_attn_layers="${num_vis_instr_attn_layers:-3}"
num_shared_attn_layers="${num_shared_attn_layers:-16}"
use_multiscale="${use_multiscale:-false}"
gradient_checkpointing="${gradient_checkpointing:-false}"
relative_action="${relative_action:-true}"
rotation_format="${rotation_format:-axisangle}"
denoise_timesteps="${denoise_timesteps:-5}"
denoise_model="${denoise_model:-rectified_flow}"

motion_expert_config="${motion_expert_config:-$REPO_ROOT/motion_expert/configs/libero_florence2.yaml}"
motion_expert_checkpoint="${motion_expert_checkpoint:-$REPO_ROOT/train_logs/motion_expert/libero_florence2/B32_18k.pt}"
motion_solver_steps="${motion_solver_steps:-10}"
motion_steps_to_run="${motion_steps_to_run:-1}"
motion_finetune="${motion_finetune:-false}"
motion_scene_fusion="${motion_scene_fusion:-true}"
motion_action_fusion="${motion_action_fusion:-false}"

configure_flower_actor_3d() {
    model_type="flower_actor_3d"
    embedding_dim="${embedding_dim:-1024}"
    motion_expert_config=""
    motion_expert_checkpoint=""
}

configure_flower_actor_3d_motion_latent() {
    model_type="flower_actor_3d_motion_latent"
    embedding_dim="${embedding_dim:-912}"
}

run_libero_eval() {
    local benchmark="$1"
    local output_file="$2"
    local video_dir="$3"

    echo "[eval] benchmark=$benchmark libero_root=$libero_root checkpoint=$checkpoint"
    torchrun --nproc_per_node "$ngpus" --master_port "${MASTER_PORT:-$((20000 + RANDOM % 20000))}" \
        online_evaluation_libero/evaluate_policy.py \
        --libero_root "$libero_root" \
        --libero_variant "$libero_variant" \
        --checkpoint "$checkpoint" \
        --benchmark "$benchmark" \
        --task_order_index "$task_order_index" \
        --task_id "$task_id" \
        --all_tasks "$all_tasks" \
        --task_start "$task_start" \
        --task_end "$task_end" \
        --num_eval "$num_eval" \
        --max_steps "$max_steps" \
        --auto_max_steps "$auto_max_steps" \
        --aggregate_plus_categories "$aggregate_plus_categories" \
        --img_h "$img_h" \
        --img_w "$img_w" \
        --seed "$seed" \
        --output_file "$output_file" \
        --save_video "$save_video" \
        --video_dir "$video_dir" \
        --video_fps "$video_fps" \
        --video_env_index "$video_env_index" \
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
        --denoise_model "$denoise_model" \
        --motion_expert_config "$motion_expert_config" \
        --motion_expert_checkpoint "$motion_expert_checkpoint" \
        --motion_solver_steps "$motion_solver_steps" \
        --motion_steps_to_run "$motion_steps_to_run" \
        --motion_finetune "$motion_finetune" \
        --motion_scene_fusion "$motion_scene_fusion" \
        --motion_action_fusion "$motion_action_fusion"
}
