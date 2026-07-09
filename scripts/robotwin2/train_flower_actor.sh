#!/bin/bash
#SBATCH --job-name=rw2
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

set -euo pipefail

export WANDB_MODE=${WANDB_MODE:-online}

REPO_ROOT=${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}
ZARR_ROOT=${ZARR_ROOT:-/data/group_data/katefgroup-ssd/world_model/ziyinx/robotwin2_zarr/}
train_data_dir=${TRAIN_DATA_DIR:-/data/group_data/katefgroup-ssd/world_model/ziyinx/robotwin2_zarr/train.zarr}
eval_data_dir=${EVAL_DATA_DIR:-/data/group_data/katefgroup-ssd/world_model/ziyinx/robotwin2_zarr/val.zarr}
train_instructions=${REPO_ROOT}/instructions/robotwin2/instructions_full.json
val_instructions=${REPO_ROOT}/instructions/robotwin2/instructions_full.json

dataset=RoboTwin2
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

val_freq=10000
eval_only=false
lr=4e-5
lr_scheduler=tristage_flower
backbone_lr=4e-5
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=600000
use_compile=false
use_ema=false
grounding_dropout_prob=0.0

model_type=flower_actor_3d
bimanual=true
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
num_history=${NUM_HISTORY:-1}
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=${ROTATION_FORMAT:-quat_wxyz}
relative_action=true
robotwin2_relative_rotation=${ROBOTWIN2_RELATIVE_ROTATION:-true}
denoise_timesteps=5
denoise_model=rectified_flow

main_dir=robotwin2
base_log_dir=${BASE_LOG_DIR:-${REPO_ROOT}/train_logs}
run_log_dir=${RUN_LOG_DIR:-$model_type-$dataset-action16raw6-anchorrelrot-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-finetuned_$finetune_backbone-job${SLURM_JOBID:-local}}
run_dir=${base_log_dir}/${main_dir}/${run_log_dir}
checkpoint=${CHECKPOINT:-}
if [[ "${checkpoint}" == "None" ]]; then
    checkpoint=""
fi
finetune=${FINETUNE:-false}

if [[ ! -d "${train_data_dir}" || ! -d "${eval_data_dir}" ]]; then
    echo "ERROR: missing merged RobotWin2 zarr stores:" >&2
    echo "  train_data_dir=${train_data_dir}" >&2
    echo "  eval_data_dir=${eval_data_dir}" >&2
    exit 1
fi
if [[ -n "${checkpoint}" && ! -f "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}" >&2
    exit 1
fi
echo "[train] checkpoint/resume path = ${checkpoint:-<scratch>}"
echo "[train] output run dir         = ${run_dir}"
echo "[train] finetune mode          = ${finetune}"
echo "[train] action normalization    = fixed hand-designed RoboTwinTrainTester constants"

ngpus=8
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

torchrun --nproc_per_node $ngpus --master_port $master_port \
    $REPO_ROOT/main.py \
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
    --base_log_dir $base_log_dir \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint "$checkpoint" \
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
    --robotwin2_relative_rotation $robotwin2_relative_rotation \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --finetune $finetune \
    --num_finetune_vision_layers $num_finetune_vision_layers \
    --vision_lr_scale $vision_lr_scale \
    --projector_lr_scale $projector_lr_scale \
    --grounding_dropout_prob $grounding_dropout_prob
