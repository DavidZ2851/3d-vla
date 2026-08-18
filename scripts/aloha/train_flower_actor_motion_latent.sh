#!/bin/bash
#SBATCH --job-name=aloha_motion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

# Train flower_actor_3d_motion_latent on the single-arm ALOHA real-world zarr.
# Front/back are 3D cameras. Wrist is RGB-only 2D and is enabled by
# AlohaTrainTester through the ALOHA-only rgb2d context flag.

export WANDB_MODE=${WANDB_MODE:-online}
export MOTION_EXPERT_USE_FLASH_ATTN_VARLEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
DATA_PATH=/data/user_data/ziyinx/datasets
BASE_LOG_DIR=$DATA_PATH/train_logs
ZARR_ROOT=${ZARR_ROOT:-/data/user_data/ziyinx/datasets/zarr_datasets/aloha_zarr}

train_data_dir=${TRAIN_DATA_DIR:-$ZARR_ROOT/train.zarr}
eval_data_dir=${EVAL_DATA_DIR:-$ZARR_ROOT/val.zarr}
train_instructions=${TRAIN_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}
val_instructions=${VAL_INSTRUCTIONS:-$ZARR_ROOT/instructions.json}

dataset=Aloha
num_workers=4
memory_limit=6
# Per-GPU batch, in samples; the loader batches B/chunk_size trajectory chunks.
# B=8 peaks at ~35 GiB on an L40S (46 GiB) with the full-finetune Florence2-large
# backbone + trainable motion expert. B=16 OOMs on L40S -- only raise it if you
# also restrict --constraint to A100_80GB.
B=${BATCH_SIZE:-8}
B_val=${BATCH_SIZE_VAL:-8}
chunk_size=8

val_freq=200
eval_only=false
lr=${LR:-2e-5}
lr_scheduler=tristage_flower
backbone_lr=${BACKBONE_LR:-2e-5}
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=${TRAIN_ITERS:-100000}
use_compile=false
use_ema=false
grounding_dropout_prob=0.0

model_type=flower_actor_3d_motion_latent
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=None

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=256
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=rot6d
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

motion_expert_config=${MOTION_EXPERT_CONFIG:-$REPO_ROOT/motion_expert/configs/aloha_florence2.yaml}
motion_expert_checkpoint=${MOTION_EXPERT_CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/motion_expert/aloha_florence2/checkpoint_step_18000.pt}
motion_solver_steps=10
motion_steps_to_run=1
motion_finetune=true
motion_scene_fusion=true
motion_action_fusion=false

main_dir=aloha
run_log_dir=${RUN_LOG_DIR:-$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-motion_finetuned_$motion_finetune-job${SLURM_JOBID:-local}}
checkpoint=${CHECKPOINT:-$BASE_LOG_DIR/${main_dir}/${run_log_dir}/last.pth}
if [[ "${checkpoint}" == "None" ]]; then
    checkpoint=""
fi
finetune=${FINETUNE:-false}

if [[ ! -d "${train_data_dir}" || ! -d "${eval_data_dir}" ]]; then
    echo "ERROR: missing ALOHA zarr stores:" >&2
    echo "  train_data_dir=${train_data_dir}" >&2
    echo "  eval_data_dir=${eval_data_dir}" >&2
    exit 1
fi
if [[ ! -f "${motion_expert_config}" ]]; then
    echo "ERROR: motion expert config not found: ${motion_expert_config}" >&2
    exit 1
fi
if [[ "${motion_expert_checkpoint}" != "None" && ! -f "${motion_expert_checkpoint}" ]]; then
    echo "ERROR: motion expert checkpoint not found: ${motion_expert_checkpoint}" >&2
    exit 1
fi

echo "[train] model         = ${model_type}"
echo "[train] dataset       = ${dataset}"
echo "[train] zarr root     = ${ZARR_ROOT}"
echo "[train] motion ckpt   = ${motion_expert_checkpoint}"
echo "[train] output dir    = ${BASE_LOG_DIR}/${main_dir}/${run_log_dir}"

# Must match #SBATCH --gres=gpu:N above.
ngpus=${NGPUS:-4}
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))
custom_img_size_args=()
if [[ -n "$custom_img_size" && "$custom_img_size" != "None" && "$custom_img_size" != "none" ]]; then
    custom_img_size_args=(--custom_img_size "$custom_img_size")
fi

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
    --base_log_dir $BASE_LOG_DIR \
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
    "${custom_img_size_args[@]}" \
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
    --grounding_dropout_prob $grounding_dropout_prob \
    --motion_expert_config $motion_expert_config \
    --motion_expert_checkpoint $motion_expert_checkpoint \
    --motion_solver_steps $motion_solver_steps \
    --motion_steps_to_run $motion_steps_to_run \
    --motion_finetune $motion_finetune \
    --motion_scene_fusion $motion_scene_fusion \
    --motion_action_fusion $motion_action_fusion
