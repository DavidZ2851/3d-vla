#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --partition=katef
#SBATCH --qos=katef_qos
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="A100_40GB"

export WANDB_MODE=${WANDB_MODE:-online}
export MOTION_EXPERT_USE_FLASH_ATTN_VARLEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
DATA_PATH=/data/group_data/katefgroup-ssd/world_model/ziyinx
BASE_LOG_DIR=$REPO_ROOT/train_logs

# FULL libero_zarr (886,043 train / 56,575 val rows, libero_90 included), NOT
# the filtered no-libero_90 subdataset this run previously used -- that copy
# under /data/user_data/ziyinx/datasets is gone. Re-run
# data_processing/filter_libero_zarr_drop_libero90.py to rebuild the filtered one.
# Train was copied to the ssd mount by 3dfa-vla-data-processing/
# sync_libero_zarr_to_ssd.sbatch; val is small and still read from group_data.
train_data_dir=$DATA_PATH/libero_zarr/train_rechunked8.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/val_rechunked8.zarr
train_instructions=instructions/libero/instructions.json
val_instructions=instructions/libero/instructions.json

dataset=Libero
# Master RNG seed: data order + model init + flow-matching noise.
# 0 == the historical behaviour; override per run, e.g. SEED=1 sbatch ...
seed=${SEED:-42}
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

val_freq=5000
eval_only=false
lr=3e-5
lr_scheduler=tristage_flower
backbone_lr=3e-5
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=200000
use_compile=false
use_ema=false

model_type=flower_actor_3d_motion_latent
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=912
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=axisangle
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

motion_expert_config=${MOTION_EXPERT_CONFIG:-$REPO_ROOT/motion_expert/configs/libero_florence2.yaml}
motion_expert_checkpoint=${MOTION_EXPERT_CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/libero_florence2/B32_18k.pt}
motion_solver_steps=10
motion_steps_to_run=1
motion_finetune=true
motion_scene_fusion=true
motion_action_fusion=false

main_dir=libero
run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-motion_finetuned_$motion_finetune-scratch-seed$seed-job${SLURM_JOBID}
# Train from scratch: no policy checkpoint, so iter/optimizer/scheduler start at
# 0 and the workspace normalizer is computed on the current dataset. The
# florence2 backbone and the motion expert are still loaded pretrained
# (motion_expert_checkpoint below); only the policy weights are random.
checkpoint=None
finetune=false

ngpus=8
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

torchrun --nproc_per_node $ngpus --master_port $master_port \
    $REPO_ROOT/main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --seed $seed \
    --num_workers $num_workers \
    --memory_limit $memory_limit \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --base_log_dir $BASE_LOG_DIR \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --eval_only $eval_only \
    --lr $lr \
    --lr_scheduler $lr_scheduler \
    --backbone_lr $backbone_lr \
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
    --denoise_model $denoise_model \
    --finetune $finetune \
    --num_finetune_vision_layers $num_finetune_vision_layers \
    --vision_lr_scale $vision_lr_scale \
    --projector_lr_scale $projector_lr_scale \
    --motion_expert_config $motion_expert_config \
    --motion_expert_checkpoint $motion_expert_checkpoint \
    --motion_solver_steps $motion_solver_steps \
    --motion_steps_to_run $motion_steps_to_run \
    --motion_finetune $motion_finetune \
    --motion_scene_fusion $motion_scene_fusion \
    --motion_action_fusion $motion_action_fusion
