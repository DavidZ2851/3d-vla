main_dir=Peract2

if [ -d "/lustre/fsw/portfolios/nvr/users/ngkanatsios" ]; then
    DATA_PATH="/lustre/fsw/portfolios/nvr/users/ngkanatsios"
else
    DATA_PATH="/data/user_data/ngkanats"
fi

train_data_dir=$DATA_PATH/zarr_datasets/Peract2_zarr/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/Peract2_zarr/val.zarr
train_instructions=instructions/peract2/instructions.json
val_instructions=instructions/peract2/instructions.json

dataset=Peract2_3dfront_3dwrist
num_workers=4
B=64
B_val=64
chunk_size=1

# Training/testing arguments, change these for HPT
val_freq=4000
eval_only=false
lr=1e-4
lr_scheduler=constant
wd=1e-10
train_iters=600000
use_compile=false

# Model arguments, change (some of) these for new architectures
model_type=denoise3d  # denoise3ddf
bimanual=true
keypose_only=true

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

workspace_normalizer_buffer=0.05
quaternion_format=xyzw
relative_action=false
denoise_timesteps=100
denoise_model=ddpm

run_log_dir=reproduce_$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-DT$denoise_timesteps
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
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --quaternion_format $quaternion_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model
