#!/bin/bash
#SBATCH --job-name=eval_peract_scratch_placecup
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err

exp=peract_reproduce
# export TMPDIR=/data/group_data/katefgroup/VLA/ziyinx/tmp
# mkdir -p "$TMPDIR"
unset LD_LIBRARY_PATH
unset PYTHONPATH
export COPPELIASIM_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export RLBENCH_RENDER_MODE=OPENGL
# tasks=(
#     close_jar
#     insert_onto_square_peg
#     light_bulb_in
#     meat_off_grill
#     open_drawer
#     place_shape_in_shape_sorter
#     place_wine_at_rack_location
#     push_buttons
#     put_groceries_in_cupboard
#     put_item_in_drawer
#     put_money_in_safe
#     reach_and_drag
#     slide_block_to_color_target
#     stack_blocks
#     stack_cups
#     sweep_to_dustpan_of_size
#     turn_tap
#     place_cups
# )
tasks=(
    place_cups
)
# Testing arguments
# scratch
checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/peract/flower_actor_3d-PeractDat-C1024-B4-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-scratch/last.pth
checkpoint_alias=flower_actor_3d-PeractDat-C1024-B4-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-scratch
# finetune
# checkpoint=/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/peract/flower_actor_3d-PeractDat-C1024-B4-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-finetune_full/last.pth
# checkpoint_alias=flower_actor_3d-PeractDat-C1024-B4-lr5e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true-finetune

num_episodes=25
max_tries=2
max_steps=20
headless=true
collision_checking=false
seed=0
replay=false

# Dataset arguments
data_dir=/data/group_data/katefgroup/VLA/peract_raw/test
dataset=PeractDat
image_size=128,128

# Model arguments
model_type=flower_actor_3d
bimanual=false
prediction_len=1  # 50
backbone=florence2
fps_subsampling_factor=1 # was 4
embedding_dim=1024
num_attn_heads=16  # was 8
num_vis_instr_attn_layers=3  #was 2
num_history=1
relative_action=true
rotation_format=quat_xyzw
denoise_timesteps=5  #10
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
        --denoise_model $denoise_model
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
