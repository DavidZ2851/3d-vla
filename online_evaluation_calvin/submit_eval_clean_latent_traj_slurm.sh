#!/usr/bin/env bash
#SBATCH --job-name=eval_calvin_motion_ablate
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=katef
#SBATCH --qos=katef_qos
#SBATCH --output=/data/user_data/%u/logs/%x_%A_%a.out
#SBATCH --error=/data/user_data/%u/logs/%x_%A_%a.err
#SBATCH --constraint="A100_40GB"

# Online CALVIN evaluation for the two new 3D motion ablations:
#   array task 0: flower_actor_3d_motion_clean_latent
#   array task 1: flower_actor_3d_motion_traj
#
# Usage:
#   sbatch online_evaluation_calvin/submit_eval_clean_latent_traj_slurm.sh
#
# Optional overrides:
#   SEED=1 SAVE_VIDEO=true sbatch online_evaluation_calvin/submit_eval_clean_latent_traj_slurm.sh
#   VARIANT=traj sbatch --array=0 online_evaluation_calvin/submit_eval_clean_latent_traj_slurm.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
mkdir -p "/data/user_data/${USER}/logs"

CONDA_SH="${CONDA_SH:-/home/ziyinx/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/data/user_data/ziyinx/miniconda3_more/envs/flowervla}"
if [[ -f "${CONDA_SH}" && -d "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "[eval] conda env not activated; expecting torchrun/python on PATH"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "${VARIANT:-${SLURM_ARRAY_TASK_ID:-0}}" in
  0|clean|clean_latent)
    variant="clean_latent"
    model_type="flower_actor_3d_motion_clean_latent"
    checkpoint="${CLEAN_LATENT_CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/flower_actor_3d_motion_clean_latent-CalvinContact10th-C912-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-motion_finetuned_true-job8187874/last.pth}"
    ;;
  1|traj)
    variant="traj"
    model_type="flower_actor_3d_motion_traj"
    checkpoint="${TRAJ_CHECKPOINT:-${REPO_ROOT}/train_logs/calvin/flower_actor_3d_motion_traj-CalvinContact10th-C912-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-motion_finetuned_true-job8187710/last.pth}"
    ;;
  *)
    echo "ERROR: VARIANT must be clean_latent or traj; got '${VARIANT:-${SLURM_ARRAY_TASK_ID:-}}'" >&2
    exit 1
    ;;
esac

if [[ ! -f "${checkpoint}" ]]; then
  echo "ERROR: checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

merged_config_file="${MERGED_CONFIG_FILE:-online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml}"
task_config_file="${TASK_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_tasks.yaml}"
ann_config_file="${ANN_CONFIG_FILE:-online_evaluation_calvin/configs/new_playtable_validation.yaml}"

seed="${SEED:-0}"
save_video="${SAVE_VIDEO:-false}"
checkpoint_alias="${CHECKPOINT_ALIAS:-$(basename "$(dirname "${checkpoint}")")}"
run_tag="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-${variant}}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${run_tag}}"
base_log_dir="${BASE_LOG_DIR:-${REPO_ROOT}/eval_logs/CALVIN/${model_type}/${checkpoint_alias}/seed${seed}/${run_id}}"

# Keep these aligned with scripts/calvin/train_flower_actor_3d_motion_{clean_latent,traj}.sh.
pred_len="${PRED_LEN:-10}"
pre_tokenize="${PRE_TOKENIZE:-true}"
custom_img_size="${CUSTOM_IMG_SIZE:-224}"
backbone="${BACKBONE:-florence2}"
num_finetune_vision_layers="${NUM_FINETUNE_VISION_LAYERS:-0}"
vision_lora_rank="${VISION_LORA_RANK:-0}"
vision_lora_alpha="${VISION_LORA_ALPHA:-16}"
vision_lora_dropout="${VISION_LORA_DROPOUT:-0.0}"
vision_lora_num_layers="${VISION_LORA_NUM_LAYERS:-0}"
vision_lora_target_modules="${VISION_LORA_TARGET_MODULES:-attn.qkv,attn.proj}"
llm_lora_rank="${LLM_LORA_RANK:-0}"
llm_lora_alpha="${LLM_LORA_ALPHA:-16}"
llm_lora_dropout="${LLM_LORA_DROPOUT:-0.0}"
llm_lora_num_layers="${LLM_LORA_NUM_LAYERS:-0}"
llm_lora_target_modules="${LLM_LORA_TARGET_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj}"
fps_subsampling_factor="${FPS_SUBSAMPLING_FACTOR:-1}"
use_multiscale="${USE_MULTISCALE:-false}"
embedding_dim="${EMBEDDING_DIM:-912}"
num_attn_heads="${NUM_ATTN_HEADS:-16}"
num_vis_instr_attn_layers="${NUM_VIS_INSTR_ATTN_LAYERS:-3}"
num_shared_attn_layers="${NUM_SHARED_ATTN_LAYERS:-16}"
relative_action="${RELATIVE_ACTION:-true}"
rotation_format="${ROTATION_FORMAT:-euler}"
denoise_timesteps="${DENOISE_TIMESTEPS:-5}"
denoise_model="${DENOISE_MODEL:-rectified_flow}"

motion_expert_config="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/calvin_florence2.yaml}"
motion_expert_checkpoint="${MOTION_EXPERT_CHECKPOINT:-${REPO_ROOT}/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt}"
motion_solver_steps="${MOTION_SOLVER_STEPS:-10}"
motion_steps_to_run="${MOTION_STEPS_TO_RUN:-1}"
motion_finetune="${MOTION_FINETUNE:-true}"
motion_scene_fusion="${MOTION_SCENE_FUSION:-true}"
motion_action_fusion="${MOTION_ACTION_FUSION:-false}"

if [[ ! -f "${motion_expert_config}" ]]; then
  echo "ERROR: motion expert config not found: ${motion_expert_config}" >&2
  exit 1
fi
if [[ "${motion_expert_checkpoint}" != "None" && ! -f "${motion_expert_checkpoint}" ]]; then
  echo "ERROR: motion expert checkpoint not found: ${motion_expert_checkpoint}" >&2
  exit 1
fi

ngpus="${NGPUS:-1}"
master_port="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

echo "[eval] variant=${variant}"
echo "[eval] model_type=${model_type}"
echo "[eval] checkpoint=${checkpoint}"
echo "[eval] base_log_dir=${base_log_dir}"
echo "[eval] seed=${seed} ngpus=${ngpus} save_video=${save_video}"
echo "[eval] motion_expert=${motion_expert_checkpoint}"

torchrun --nproc_per_node "${ngpus}" --master_port "${master_port}" \
  online_evaluation_calvin/evaluate_policy.py \
  --merged_config_file "${merged_config_file}" \
  --task_config_file "${task_config_file}" \
  --ann_config_file "${ann_config_file}" \
  --seed "${seed}" \
  --checkpoint "${checkpoint}" \
  --base_log_dir "${base_log_dir}" \
  --save_video "${save_video}" \
  --model_type "${model_type}" \
  --pred_len "${pred_len}" \
  --pre_tokenize "${pre_tokenize}" \
  --custom_img_size "${custom_img_size}" \
  --backbone "${backbone}" \
  --num_finetune_vision_layers "${num_finetune_vision_layers}" \
  --vision_lora_rank "${vision_lora_rank}" \
  --vision_lora_alpha "${vision_lora_alpha}" \
  --vision_lora_dropout "${vision_lora_dropout}" \
  --vision_lora_num_layers "${vision_lora_num_layers}" \
  --vision_lora_target_modules "${vision_lora_target_modules}" \
  --llm_lora_rank "${llm_lora_rank}" \
  --llm_lora_alpha "${llm_lora_alpha}" \
  --llm_lora_dropout "${llm_lora_dropout}" \
  --llm_lora_num_layers "${llm_lora_num_layers}" \
  --llm_lora_target_modules "${llm_lora_target_modules}" \
  --fps_subsampling_factor "${fps_subsampling_factor}" \
  --use_multiscale "${use_multiscale}" \
  --embedding_dim "${embedding_dim}" \
  --num_attn_heads "${num_attn_heads}" \
  --num_vis_instr_attn_layers "${num_vis_instr_attn_layers}" \
  --num_shared_attn_layers "${num_shared_attn_layers}" \
  --relative_action "${relative_action}" \
  --rotation_format "${rotation_format}" \
  --denoise_timesteps "${denoise_timesteps}" \
  --denoise_model "${denoise_model}" \
  --motion_expert_config "${motion_expert_config}" \
  --motion_expert_checkpoint "${motion_expert_checkpoint}" \
  --motion_solver_steps "${motion_solver_steps}" \
  --motion_steps_to_run "${motion_steps_to_run}" \
  --motion_finetune "${motion_finetune}" \
  --motion_scene_fusion "${motion_scene_fusion}" \
  --motion_action_fusion "${motion_action_fusion}"
