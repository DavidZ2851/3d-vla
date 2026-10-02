#!/bin/bash
# Fine-tune GeomVLA (flower_actor_3d_motion_latent) from the released LIBERO checkpoint on a
# MolmoSpaces zarr made by data_processing/molmospaces_to_zarr.py (LIBERO layout, robot base frame).
# Architecture args must match the LIBERO checkpoint (C=912, 16 heads, axisangle, ...).
#   TRAIN=<train.zarr> VAL=<val.zarr> RUN=<name> [NGPU=1 B=8 ITERS=20000 LR=3e-5 VAL_FREQ=1000] bash finetune_libero_ckpt.sh
set -e
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export MOTION_EXPERT_USE_FLASH_ATTN_VARLEN=${MOTION_EXPERT_USE_FLASH_ATTN_VARLEN:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-online}
CKPT=${CKPT:-/data/haotian/geomvla_ckpt/libero/last.pth}
NGPU=${NGPU:-1}; B=${B:-8}; ITERS=${ITERS:-20000}; LR=${LR:-3e-5}; VAL_FREQ=${VAL_FREQ:-1000}
cd $REPO_ROOT
torchrun --nproc_per_node $NGPU --master_port ${MASTER_PORT:-29517} main.py \
    --train_data_dir $TRAIN --eval_data_dir $VAL \
    --train_instructions ${TRAIN%/}.instructions.json --val_instructions ${VAL%/}.instructions.json \
    --dataset Libero --seed ${SEED:-42} --num_workers 4 --memory_limit 6 \
    --batch_size $B --batch_size_val $B --chunk_size ${CHUNK:-8} \
    --base_log_dir ${LOG_DIR:-$REPO_ROOT/train_logs} --exp_log_dir molmospaces --run_log_dir $RUN \
    --checkpoint $CKPT --finetune true \
    --val_freq $VAL_FREQ --interm_ckpt_freq ${CKPT_FREQ:-2000} --eval_only false \
    --lr $LR --lr_scheduler ${SCHED:-constant} --backbone_lr $LR --wd 0.05 --train_iters $ITERS \
    --use_compile false --use_ema false \
    --model_type flower_actor_3d_motion_latent --bimanual false --keypose_only false --pre_tokenize true \
    --custom_img_size 224 --backbone florence2 --finetune_backbone true --finetune_text_encoder false \
    --fps_subsampling_factor 1 --embedding_dim 912 --num_attn_heads 16 --num_vis_instr_attn_layers 3 \
    --num_history 1 --num_shared_attn_layers 16 --workspace_normalizer_buffer 0.0 \
    --relative_action true --rotation_format axisangle --denoise_timesteps 5 --denoise_model rectified_flow \
    --num_finetune_vision_layers 0 --vision_lr_scale 0.2 --projector_lr_scale 0.4 \
    --motion_expert_config $REPO_ROOT/motion_expert/configs/libero_florence2.yaml --motion_expert_checkpoint None \
    --motion_solver_steps 10 --motion_steps_to_run 1 --motion_finetune true \
    --motion_scene_fusion true --motion_action_fusion false
