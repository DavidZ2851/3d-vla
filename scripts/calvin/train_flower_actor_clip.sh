main_dir=CALVIN

DATA_PATH=/data/user_data/ngkanats

train_data_dir=$DATA_PATH/zarr_datasets/CALVIN_zarr/train_rechunked8.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/CALVIN_zarr/val_rechunked8.zarr
train_instructions=instructions/calvin/train_instructions.json
val_instructions=instructions/calvin/val_instructions.json

dataset=Calvin
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

# Training/testing arguments, change these for HPT
val_freq=10000
eval_only=false
lr=2e-5
lr_scheduler=tristage_flower
wd=0.05
train_iters=50000
use_compile=false
use_ema=true

# Model arguments, change (some of) these for new architectures
model_type=flower_actor
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=1024
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=euler
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth
# checkpoint=flower.pth

ngpus=4

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --memory_limit $memory_limit \
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
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --custom_img_size $custom_img_size \
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
