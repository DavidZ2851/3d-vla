#!/bin/bash
#SBATCH --job-name=eval_libero_plus
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/eval_libero_common.sh"

libero_root="${LIBERO_ROOT:-/data/user_data/ziyinx/LIBERO-plus}"
libero_variant="${LIBERO_VARIANT:-plus}"
exp="${EXP:-LIBERO-plus}"
seed="${SEED:-0}"
auto_max_steps="${AUTO_MAX_STEPS:-true}"
aggregate_plus_categories="${AGGREGATE_PLUS_CATEGORIES:-true}"

# LIBERO-plus recommends one rollout per task.
num_eval="${NUM_EVAL:-1}"
all_tasks="${ALL_TASKS:-true}"
task_start="${TASK_START:-0}"
task_end="${TASK_END:--1}"

checkpoint="${CHECKPOINT:-}"
model="${MODEL:-}"

case "${PRESET:-}" in
    3dfavla)
        checkpoint="${checkpoint:-$REPO_ROOT/train_logs/libero/3dfavla/75k.pth}"
        model="${model:-flower_actor_3d}"
        ;;
    3dfavla_motion)
        checkpoint="${checkpoint:-$REPO_ROOT/train_logs/libero/3dfavla_motion/75k.pth}"
        model="${model:-flower_actor_3d_motion_latent}"
        ;;
    3dfavla_motion_subset)
        checkpoint="${checkpoint:-$REPO_ROOT/train_logs/libero/3dfavla_motion_subset/95k.pth}"
        model="${model:-flower_actor_3d_motion_latent}"
        ;;
esac

if [[ -z "$checkpoint" || -z "$model" ]]; then
    echo "Set CHECKPOINT and MODEL, or PRESET=3dfavla|3dfavla_motion|3dfavla_motion_subset"
    exit 1
fi

case "$model" in
    flower_actor_3d)
        configure_flower_actor_3d
        ;;
    flower_actor_3d_motion_latent)
        configure_flower_actor_3d_motion_latent
        ;;
    *)
        echo "Unsupported MODEL=$model"
        exit 1
        ;;
esac

ckpt_tag="$(basename "$(dirname "$checkpoint")")/$(basename "$checkpoint" .pth)"

suites=(libero_spatial libero_object libero_goal libero_10)
if [[ -n "${SUITES:-}" ]]; then
    read -r -a suites <<< "$SUITES"
fi

for benchmark in "${suites[@]}"; do
    chunk_tag=""
    if [[ "$task_end" -ge 0 ]]; then
        chunk_tag="_tasks${task_start}_${task_end}"
    fi
    output_file="eval_logs/$exp/$ckpt_tag/seed${seed}/$benchmark${chunk_tag}/eval.json"
    video_dir="eval_logs/$exp/videos/seed${seed}/$ckpt_tag/$benchmark${chunk_tag}"
    run_libero_eval "$benchmark" "$output_file" "$video_dir"
done

echo "[done] LIBERO-plus evaluation finished for checkpoint=$checkpoint"
