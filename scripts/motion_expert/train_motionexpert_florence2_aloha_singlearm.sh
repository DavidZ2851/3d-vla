#!/bin/bash
#SBATCH --job-name=motion_alohasa
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

# Motion expert on the ALOHA SINGLE-ARM SpaTracker2 tracks (5 tasks, front+back
# head cameras pooled). Identical to train_motionexpert_florence2_aloha.sh except
# for the config, which swaps in the single-arm zarr, its own percentile
# normalization stats, and its own 25-entry instruction table.
#
# Prereqs (both produced by the chained jobs in
# 3dfa-vla-data-processing/data_processing/):
#   motion_aloha_singlearm_zarr.sbatch   -> the train/val rechunked8 stores
#   motion_aloha_singlearm_stats.sbatch  -> datasets/aloha_singlearm_motion_normalization_stats.json
# The stats are dataset-specific; reusing the bimanual JSON mis-scales the flow
# targets, so this script hard-fails when the file is missing.

set -euo pipefail

# Activate flowervla explicitly. Florence-2's remote modeling code calls
# `self._supports_sdpa`, which transformers removed after 4.4x: the system
# python3.9 (transformers 4.57.1) dies at model construction with
#   AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
# flowervla pins transformers 4.40.2, which works. The sibling bimanual script
# relies on the SUBMITTING shell already having flowervla active (sbatch inherits
# it via --export=ALL); that silently breaks when submitted from a plain shell or
# from a dependency-chained launcher, so activate it here instead.
source /home/ziyinx/miniconda3/etc/profile.d/conda.sh
conda activate flowervla

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/aloha_singlearm_florence2.yaml}
FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594
NGPUS=${NGPUS:-4}
MASTER_PORT=${MASTER_PORT:-$(expr 10000 + $(echo -n ${SLURM_JOBID:-12345} | tail -c 4))}
RESUME_CKPT=${RESUME_CKPT:-}

export WANDB_MODE=${WANDB_MODE:-online}
export MOTION_EXPERT_FORCE_SDPA_ATTENTION=${MOTION_EXPERT_FORCE_SDPA_ATTENTION:-0}
export MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN=${MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN:-0}
export MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC=${MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC:-1}
if [ -d "$FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$FLORENCE2_CACHE_SNAPSHOT}
else
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
fi

cd "$REPO_ROOT"

# Preflight: the config silently trains with whatever it finds, so check the
# three dataset-specific inputs up front.
python - "$CONFIG_PATH" <<'PY'
import sys, os, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
missing = []
for label, p in (
    ("train_zarr", cfg["data"]["train_zarr"]),
    ("val_zarr", cfg["data"]["val_zarr"]),
    ("instructions", cfg["data"]["instructions_path"]),
    ("normalization_stats", cfg["train"]["normalization_stats_path"]),
):
    if not os.path.exists(p):
        missing.append(f"  {label}: {p}")
if missing:
    sys.exit("ERROR: missing motion-expert inputs:\n" + "\n".join(missing))
print("[preflight] all dataset inputs present")
PY

# Guard the env too: a wrong transformers only fails ~15s in, but from inside a
# 4-GPU allocation.
python - <<'PY'
import sys, transformers
major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
print(f"[preflight] python={sys.executable} transformers={transformers.__version__}")
if (major, minor) > (4, 44):
    sys.exit(
        f"ERROR: transformers {transformers.__version__} removed the _supports_sdpa "
        "attribute Florence-2 relies on. Activate the flowervla env (4.40.2)."
    )
PY

RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS=(--resume "$RESUME_CKPT")
fi
torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" \
  -m motion_expert.train --config "$CONFIG_PATH" "${RESUME_ARGS[@]}" "$@"
