#!/bin/bash
#SBATCH --job-name=eval_libero_pro
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/ziyinx/logs/%x_%j.out
#SBATCH --error=/data/user_data/ziyinx/logs/%x_%j.err
#SBATCH --constraint="L40S|A100_80GB"

# Thin wrapper around eval_libero_pro.sh that activates the eval conda env first.
# Florence2 requires the flowervla env (transformers 4.40.2); see env notes.
# PRESET / SUITES / EXP / SEED are inherited from the submitting env via --export.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-flowervla}"

# NOTE: SLURM copies only this batch script into /var/spool/slurmd/<job>/, so a
# path derived from BASH_SOURCE points at the spool dir (where eval_libero_pro.sh
# does not exist). Use the real repo location instead.
REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
exec bash "$REPO_ROOT/online_evaluation_libero/eval_libero_pro.sh"
