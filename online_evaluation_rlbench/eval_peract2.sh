exp=bimanual_reproduce
tasks=(
    bimanual_push_box
    bimanual_lift_ball
    bimanual_dual_push_buttons
    bimanual_pick_plate
    bimanual_put_item_in_drawer
    bimanual_put_bottle_in_fridge
    bimanual_handover_item
    bimanual_pick_laptop
    bimanual_straighten_rope
    bimanual_sweep_to_dustpan
    bimanual_lift_tray
    bimanual_handover_item_easy
    bimanual_take_tray_out_of_oven
)
tasks=(
    bimanual_straighten_rope
)


# Testing arguments
checkpoint=train_logs/Peract2/reproduce_denoise3d-Peract2_3dfront_3dwrist-C120-B64-lr1e-4-constant-H3-rectified_flow-DT10/best.pth
checkpoint_alias=reproduce_denoise3d-Peract2_3dfront_3dwrist-C120-B64-lr1e-4-constant-H3-rectified_flow-DT10
# checkpoint=train_logs/Peract2/flower_actor-Peract2_3dfront_3dwrist-C1024-B8-lr2e-5-constant-H3-rectified_flow-florence2-finetuned_true/best.pth
# checkpoint_alias=flower_actor-Peract2_3dfront_3dwrist-C1024-B8-lr2e-5-constant-H3-rectified_flow-florence2-finetuned_true
checkpoint=train_logs/Peract2/k2k_flower_actor_3d-Peract2TCSingle-C120-B16-lr1e-4-constant-H1-rectified_flow-florence2-finetuned_false/best.pth
checkpoint_alias=k2k_flower_actor_3d-Peract2TCSingle-C120-B16-lr1e-4-constant-H1-rectified_flow-florence2-finetuned_false

num_episodes=100
max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0
replay=true

# Dataset arguments
data_dir=/data/group_data/katefgroup/VLA/peract2_raw_squash/train/
dataset=Peract2_3dfront_3dwrist
image_size=256,256

# Model arguments
model_type=flower_actor_3d
bimanual=true
prediction_len=50
backbone=florence2
fps_subsampling_factor=4
embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=1
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
        --denoise_model $denoise_model
done

python online_evaluation_rlbench/collect_results.py \
    --folder eval_logs/$exp/$checkpoint_alias/seed$seed/
