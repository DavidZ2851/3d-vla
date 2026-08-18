#!/bin/bash
#SBATCH --job-name=rw2_sweep_submitter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#
# Run queue_eval_50_rev.sh as a CPU batch job instead of a background process on
# a login/interactive node.
#
# Why: the sweep takes far longer than an interactive allocation. A submitter
# started with nohup inside an interactive job is in that job's cgroup and is
# killed when the allocation ends, stranding the sweep half-submitted. As a CPU
# job it has its own allocation and survives.
#
# The submitter is pure bookkeeping (squeue + sbatch on a poll loop), so 1 cpu /
# 2G on the cpu partition is plenty; it holds no GPU. It does count as one job
# against the shared 50-job-per-user cap, which the MAX_QUEUED accounting in
# queue_eval_50_rev.sh already handles since it counts all of the user's jobs.
#
# Walltime is the cpu partition maximum (2 days). If the sweep outlives that,
# just resubmit: queue_eval_50_rev.sh's resume guard skips pairs that already
# have a summary dir or are already queued.
#
# Usage:
#   SPLIT=randomized RW2_SWAP_RB=True SKIP_TASKS= \
#     CHECKPOINTS=/path/to/300k.pth \
#     sbatch online_evaluation_robotwin/submit_queue_eval_50.sbatch
#
# All queue_eval_50_rev.sh env vars (SPLIT, CHECKPOINTS, RW2_SWAP_RB,
# SKIP_TASKS, MAX_QUEUED, POLL_SECONDS, SEED, NUM_EPISODES, PARTITION) are
# forwarded: sbatch exports the submitting environment by default.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"

echo "[submitter-job] $(date '+%F %T') running on $(hostname), job ${SLURM_JOB_ID:-<none>}"
echo "[submitter-job] split=${SPLIT:-clean} swap_rb=${RW2_SWAP_RB:-False}"
echo "[submitter-job] checkpoints=${CHECKPOINTS:-<default>}"
echo "[submitter-job] skip_tasks=${SKIP_TASKS-<default>}"

exec bash "${REPO_ROOT}/online_evaluation_robotwin/queue_eval_50_rev.sh"
