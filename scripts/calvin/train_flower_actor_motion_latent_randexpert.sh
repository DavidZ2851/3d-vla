#!/bin/bash
#SBATCH --job-name=calvin_motion_randexpert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

# Random-init frozen MotionExpert ablation for CALVIN.
#
# Identical to scripts/calvin/train_flower_actor_motion_latent.sh (the job-9809173
# recipe) in EVERY respect except two lines:
#
#   motion_expert_checkpoint=None   -> main.py's str_none maps "None" to Python
#                                      None, so flower_actor.py:495 skips
#                                      _load_motion_expert_checkpoint and the
#                                      expert keeps its random initialization.
#   motion_finetune=false           -> flower_actor.py:497 sets requires_grad=False
#                                      on every motion_expert parameter, so it
#                                      stays random for the whole run.
#
# Architecture and parameter count are unchanged: size comes from
# motion_expert_config (calvin_florence2.yaml), not from the checkpoint. Only the
# ~80M decoder (point_encoder + flow_decoder) is random -- the expert's own
# Florence-2 backbone is still loaded pretrained by FlorenceTraceFeatureExtractor,
# exactly as in the reference run. This isolates "was the motion pretraining
# worth anything" from "does a second frozen VLM branch add capacity".
#
# No existing file is modified; motion_finetune is hardcoded in the reference
# script, so this has to be a separate entry point rather than an env override.
#
# Usage:  sbatch scripts/calvin/train_flower_actor_motion_latent_randexpert.sh

export WANDB_MODE=${WANDB_MODE:-online}

# The reference script relies on the submitting shell already having flowervla
# activated (sbatch --export=ALL carries PATH through). Submitting from a clean
# shell silently falls back to system python3.9 and dies on
# `ModuleNotFoundError: No module named 'tensorboard'`. Activate explicitly so
# this entry point works regardless of how it is launched.
CONDA_SH="${CONDA_SH:-/home/ziyinx/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla}"
if [[ -f "${CONDA_SH}" && -d "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  echo "[randexpert] conda env: ${CONDA_ENV}"
else
  echo "[randexpert] WARNING: conda env not found; relying on inherited PATH" >&2
fi
echo "[randexpert] python: $(command -v python) ($(python -V 2>&1))"

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
DATA_PATH=/data/user_data/ziyinx/datasets

# CALVIN train zarr now lives on the SSD tier; override with CALVIN_DATA_ROOT.
CALVIN_DATA_ROOT="${CALVIN_DATA_ROOT:-/data/group_data/katefgroup-ssd/world_model/ziyinx}"
train_data_dir=${CALVIN_DATA_ROOT}/calvin_zarr/train_rechunked8.zarr
eval_data_dir=/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr
train_instructions=instructions/calvin/train_instructions.json
val_instructions=instructions/calvin/val_instructions.json

dataset=CalvinContact10th
# Master RNG seed: data order + model init + flow-matching noise.
# Also seeds the random MotionExpert init, so the ablation is reproducible.
seed=${SEED:-42}
num_workers=4
memory_limit=6
B=8
B_val=8
chunk_size=8

val_freq=5000
eval_only=false
lr=2e-5
lr_scheduler=tristage_flower
backbone_lr=2e-5
num_finetune_vision_layers=0
vision_lr_scale=0.2
projector_lr_scale=0.4
wd=0.05
train_iters=50000
use_compile=false
use_ema=true

model_type=flower_actor_3d_motion_latent
bimanual=false
keypose_only=false
pre_tokenize=true
custom_img_size=224

backbone=florence2
finetune_backbone=true
finetune_text_encoder=false
fps_subsampling_factor=1

C=960
num_attn_heads=16
num_vis_instr_attn_layers=3
num_history=1
num_shared_attn_layers=16

workspace_normalizer_buffer=0.0
rotation_format=euler
relative_action=true
denoise_timesteps=5
denoise_model=rectified_flow

motion_expert_config=${MOTION_EXPERT_CONFIG:-$REPO_ROOT/motion_expert/configs/calvin_florence2.yaml}
# ---- the ablation: random init, never loaded, never trained ----
motion_expert_checkpoint=None
motion_finetune=false
# ----------------------------------------------------------------
motion_solver_steps=10
motion_steps_to_run=1
motion_scene_fusion=true
motion_action_fusion=false

main_dir=calvin
run_log_dir=$model_type-randexpert_frozen-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model-$backbone-motion_finetuned_$motion_finetune-seed$seed-job${SLURM_JOBID}
checkpoint=$DATA_PATH/train_logs/${main_dir}/${run_log_dir}/last.pth
finetune=false

ngpus=4
master_port=$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))

echo "[randexpert] motion_expert_checkpoint=${motion_expert_checkpoint} (random init)"
echo "[randexpert] motion_finetune=${motion_finetune} (frozen)"
echo "[randexpert] log_dir=${REPO_ROOT}/train_logs/${main_dir}/${run_log_dir}"

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
