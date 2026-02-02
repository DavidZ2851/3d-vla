main_dir=Peract2

DATA_PATH=/data/user_data/ngkanats

train_data_dir=$DATA_PATH/zarr_datasets/Peract2_k2k_zarr/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/Peract2_k2k_zarr/val.zarr
train_instructions=instructions/peract2/instructions.json
val_instructions=instructions/peract2/instructions.json

dataset=Peract2TCSingle  # Peract2_3dfront_3dwrist
num_workers=4
B=16
B_val=16
chunk_size=1

# Training/testing arguments, change these for HPT
val_freq=4000
eval_only=false
lr=1e-4
lr_scheduler=constant  # tristage_flower
wd=5e-10
train_iters=600000
use_compile=false
use_ema=false
lv2_batch_size=4

# Model arguments, change (some of) these for new architectures
model_type=flower_actor_3d
bimanual=true
keypose_only=true
pre_tokenize=true
# custom_img_size=224

backbone=florence2
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=4

workspace_normalizer_buffer=0.05
rotation_format=quat_xyzw
relative_action=false
denoise_timesteps=10
denoise_model=rectified_flow

run_log_dir=keypose_$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

ngpus=4

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
