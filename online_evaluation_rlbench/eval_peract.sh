exp=refactored_better_3dda_peract_sc20
tasks=(
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
    place_cups
)
tasks=(
    insert_onto_square_peg
)

# Testing arguments
checkpoint=train_logs/Peract/lv2_denoise3d-PeractDat-C120-B16-lr1e-4-constant-H3-rectified_flow/best.pth
checkpoint_alias=lv2_denoise3d-PeractDat-C120-B16-lr1e-4-constant-H3-rectified_flow
checkpoint=diffuser_actor_peract.pth
checkpoint_alias=old_3dda

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
image_size=256,256

# Model arguments
model_type=3dda
bimanual=false
prediction_len=1
backbone=florence2
fps_subsampling_factor=4
embedding_dim=1024
num_attn_heads=8
num_vis_instr_attn_layers=2
num_history=1
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=10
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
