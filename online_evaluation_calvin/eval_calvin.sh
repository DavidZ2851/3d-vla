# You probably don't need to change these
merged_config_file=online_evaluation_calvin/configs/merged_config_val_abc_d.yaml
task_config_file=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann_config_file=online_evaluation_calvin/configs/new_playtable_validation.yaml

# Things you may want to change if you train/eval a new model
seed=0
checkpoint=train_logs/CALVIN/flower_actor_3d-Calvin10th-C1024-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true/interm20000.pth
base_log_dir=eval_logs/CALVIN/flower_actor_3d-Calvin10th-C1024-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true/
checkpoint=train_logs/CALVIN/flower_actor_3d-Calvin10th-C1024-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true/best.pth
base_log_dir=eval_logs/CALVIN/flower_actor_3d-Calvin10th-C1024-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-finetuned_true/
save_video=false

# Things you can change if you customize the model architecture
model_type=flower_actor_3d
pred_len=10
pre_tokenize=true
custom_img_size=224
backbone=florence2
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
