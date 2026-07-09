#!/bin/bash
#SBATCH --job-name=eval_libero_pro
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/eval_libero_common.sh"

libero_root="${LIBERO_ROOT:-/data/user_data/ziyinx/LIBERO-PRO}"
libero_variant="${LIBERO_VARIANT:-pro}"
exp="${EXP:-LIBERO-PRO}"
seed="${SEED:-0}"
auto_max_steps="${AUTO_MAX_STEPS:-true}"
num_eval="${NUM_EVAL:-50}"

# Checkpoint selection:
#   CHECKPOINT=/path/to/ckpt.pth MODEL=flower_actor_3d bash ...
#   CHECKPOINT=/path/to/ckpt.pth MODEL=flower_actor_3d_motion_latent bash ...
#   PRESET=3dfavla|3dfavla_motion|3dfavla_motion_subset bash ...
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

# Original LIBERO suites (optional baseline) + 20 LIBERO-PRO perturbation suites.
suites=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
    libero_goal_object
    libero_spatial_object
    libero_10_object
    libero_object_object
    libero_goal_swap
    libero_spatial_swap
    libero_10_swap
    libero_object_swap
    libero_goal_lan
    libero_spatial_lan
    libero_10_lan
    libero_object_lan
    libero_goal_task
    libero_spatial_task
    libero_10_task
    libero_object_task
    libero_goal_env
    libero_spatial_env
    libero_10_env
    libero_object_env
)

if [[ -n "${SUITES:-}" ]]; then
    # Example: SUITES="libero_goal_object libero_spatial_swap"
    read -r -a suites <<< "$SUITES"
fi

for benchmark in "${suites[@]}"; do
    output_file="eval_logs/$exp/$ckpt_tag/seed${seed}/$benchmark/eval.json"
    video_dir="eval_logs/$exp/videos/seed${seed}/$ckpt_tag/$benchmark"
    run_libero_eval "$benchmark" "$output_file" "$video_dir"
done

echo "[done] LIBERO-PRO evaluation finished for checkpoint=$checkpoint"
