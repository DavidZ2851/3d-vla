#!/usr/bin/env bash
# Live CALVIN online-eval inference timing for one preset over a few sequences.
# Usage: PRESET=baseline|motion bash run_bench_eval.sh
set -euo pipefail
REPO=/data/user_data/ziyinx/analogical_manipulation_clean
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain
PY=/data/user_data/ziyinx/miniconda3_more/envs/flowervla/bin/python

PRESET="${PRESET:-baseline}"
export BENCH_TIMING=1
export BENCH_WARMUP="${BENCH_WARMUP:-10}"
export BENCH_MAX_SEQ="${BENCH_MAX_SEQ:-4}"
export BENCH_JSON_OUT="/data/user_data/ziyinx/bench_online_${PRESET}.json"

merged=online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml
task=online_evaluation_calvin/configs/new_playtable_tasks.yaml
ann=online_evaluation_calvin/configs/new_playtable_validation.yaml

COMMON=(--merged_config_file "$merged" --task_config_file "$task" --ann_config_file "$ann"
  --seed 0 --save_video false --backbone florence2 --pred_len 10 --pre_tokenize true
  --custom_img_size 224 --num_finetune_vision_layers 24 --vision_lora_rank 0 --llm_lora_rank 0
  --num_attn_heads 16 --num_vis_instr_attn_layers 3 --num_shared_attn_layers 16
  --relative_action true --rotation_format euler --denoise_timesteps 5 --denoise_model rectified_flow)

if [[ "$PRESET" == "baseline" ]]; then
  ARGS=(--model_type flower_actor_3d
    --checkpoint "$REPO/train_logs/calvin/3dfavla/last.pth"
    --embedding_dim 1024
    --base_log_dir "/tmp/bench_eval_baseline_$$")
else
  ARGS=(--model_type flower_actor_3d_motion_latent
    --checkpoint "$REPO/train_logs/calvin/3dfavla_motion/last.pth"
    --embedding_dim 960
    --motion_expert_config "$REPO/motion_expert/configs/calvin_florence2.yaml"
    --motion_expert_checkpoint "$REPO/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt"
    --motion_solver_steps 10 --motion_steps_to_run 1 --motion_finetune false
    --motion_scene_fusion true --motion_action_fusion false
    --base_log_dir "/tmp/bench_eval_motion_$$")
fi

"$PY" -m torch.distributed.run --nproc_per_node 1 --master_port $((10000 + RANDOM % 50000)) \
  online_evaluation_calvin/evaluate_policy.py "${COMMON[@]}" "${ARGS[@]}"
