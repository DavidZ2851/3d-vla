#!/bin/bash
#SBATCH --job-name=eval_peract2_finetune_bimanual_push_box
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

exp=bimanual_reproduce
# export TMPDIR=/data/group_data/katefgroup/VLA/ziyinx/tmp
# mkdir -p "$TMPDIR"
unset LD_LIBRARY_PATH
unset PYTHONPATH
export COPPELIASIM_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean/PyRep_peract2/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export RLBENCH_RENDER_MODE=OPENGL

# tasks=(
#     bimanual_push_box
#     bimanual_lift_ball
#     bimanual_dual_push_buttons
#     bimanual_pick_plate
#     bimanual_put_item_in_drawer
#     bimanual_put_bottle_in_fridge
#     bimanual_handover_item
#     bimanual_pick_laptop
#     bimanual_straighten_rope
#     bimanual_sweep_to_dustpan
#     bimanual_lift_tray
#     bimanual_handover_item_easy
#     bimanual_take_tray_out_of_oven
# )
tasks=(
    bimanual_push_box
)

# Testing arguments
# scratch
# checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/peract2/flower_actor_3d-Peract2_3dfront-C1024-B64-lr1e-4-constant-H3-rectified_flow-florence2-finetuned_false-job6485989/last.pth
# checkpoint_alias=flower_actor_3d-Peract2_3dfront-C1024-B64-lr1e-4-constant-H3-rectified_flow-florence2-finetuned_false-job6485989
# finetune
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/peract2/flower_actor_3d-Peract2_3dfront-C256-B32-lr1e-4-constant-H3-rectified_flow-florence2-finetuned_true-job6581506/last.pth
checkpoint_alias=flower_actor_3d-Peract2_3dfront-C256-B32-lr1e-4-constant-H3-rectified_flow-florence2-finetuned_true-job6581506
num_episodes=100
max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0
replay=false

# Dataset arguments
data_dir=/data/group_data/katefgroup/datasets/peract2/peract2_test/  # load eval env layouts
dataset=Peract2_3dfront
image_size=256,256

# Model arguments
model_type=flower_actor_3d
bimanual=true
prediction_len=1  # for keypose
backbone=florence2
fps_subsampling_factor=1  # was 4
embedding_dim=256  # was 120
num_attn_heads=16  # was 8
num_vis_instr_attn_layers=3
num_shared_attn_layers=4
num_history=3
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
    xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint $checkpoint \
        --task ${tasks[$i]} \
        --num_episodes $num_episodes \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --seed $seed \
        --replay $replay \
        --data_dir $data_dir \
        --dataset $dataset \
        --image_size $image_size \
        --output_file eval_logs/$exp/$checkpoint_alias/seed$seed/${tasks[$i]}/eval.json  \
        --model_type $model_type \
        --bimanual $bimanual \
        --prediction_len $prediction_len \
        --backbone $backbone \
        --fps_subsampling_factor $fps_subsampling_factor \
        --embedding_dim $embedding_dim \
        --num_attn_heads $num_attn_heads \
        --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
        --num_history $num_history \
        --relative_action $relative_action \
        --rotation_format $rotation_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model \
        --num_shared_attn_layers $num_shared_attn_layers
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed
