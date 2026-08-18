#!/bin/bash
#SBATCH --job-name=rw2_trainseed_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#SBATCH --constraint="L40S"
#SBATCH --exclude=babel-p5-24,babel-s5-24

# Train-set REPLAY eval: run the policy on the exact scenes its training data
# was collected from, by replaying the data-collection seeds through
# RW2_EVAL_SEED_LIST (see RoboTwin/script/eval_policy.py). Unlike the normal
# eval schedule there is no 100000*(1+SEED) offset and no expert-skip: every
# listed seed is used exactly once, in order.
#
# Which seeds? RoboTwin writes collection seeds to
#   <ROOT>/<task>/<variant>/seed.txt   (episodeN <-> the (N+1)-th token)
# and the zarr converter (3dfa-vla-data-processing/data_processing/
# robotwin2_to_zarr.py, `_build_val_uid_set`) holds out a val fraction per
# (task, variant) by sha256(uid) order -- NOT by episode index. With the
# default TRAIN_VAL_RATIO=24:1 a 50-episode clean variant is split 48 train / 2
# val, but the 2 val episodes are hash-selected (for hanging_mug/clean they are
# episodes 13 and 46, NOT 48 and 49). This script reproduces that rule exactly
# so only the 48 TRAINED scenes are replayed.
#
# Usage:
#   sbatch online_evaluation_robotwin/submit_eval_trainseeds_slurm.sh
#   TASK=beat_block_hammer sbatch online_evaluation_robotwin/submit_eval_trainseeds_slurm.sh
#   EPISODES=0,1,2 sbatch online_evaluation_robotwin/submit_eval_trainseeds_slurm.sh   # explicit override
#
# ------------------------- CONFIGURE HERE -----------------------------------
TASK="${TASK:-hanging_mug}"
# clean | randomized -- picks both the collection variant (seed source) and the
# matching *_analogical eval config.
SPLIT="${SPLIT:-clean}"
# Checkpoint to evaluate. rw2_all is the scaled-up 50-task motion-latent model.
CHECKPOINT="${CHECKPOINT:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/robotwin2/rw2_all/300k.pth}"
# Must match the checkpoint (rw2_all: policy 1024, motion expert hidden 512).
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT:-}"
# Train/val split rule used when the training zarr was built.
TRAIN_VAL_RATIO="${TRAIN_VAL_RATIO:-24:1}"
# Comma/space separated episode indices to replay. Empty -> derive the train
# episodes from TRAIN_VAL_RATIO (recommended).
EPISODES="${EPISODES:-}"
# Cap on how many train episodes to replay (0 = all). The randomized variant has
# 500 episodes / ~480 train, far more than one job can run, so subsample. The
# subsample is evenly spaced over the train episodes (not the first N), so it
# spans the whole collection range, and is fully deterministic.
MAX_EPISODES="${MAX_EPISODES:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
ROBOTWIN2_DATA_ROOT="${ROBOTWIN2_DATA_ROOT:-/data/group_data/katefgroup/datasets/ziyinx/RoboTwin2}"
SUMMARY_ROOT="${SUMMARY_ROOT:-}"
# ----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG:-${REPO_ROOT}/motion_expert/configs/robotwin2_florence2_large.yaml}"
CONDA_BASE="${CONDA_BASE:-/home/ziyinx/miniconda3}"
CONDA_ENV="${CONDA_ENV:-flowervla}"

case "${SPLIT}" in
  clean)      VARIANT="aloha_vla_clean50_tcp" ;;
  randomized) VARIANT="aloha_vla_randomized500_tcp" ;;
  *) echo "ERROR: SPLIT must be 'clean' or 'randomized', got '${SPLIT}'." >&2; exit 1 ;;
esac

TASK_VARIANT_DIR="${ROBOTWIN2_DATA_ROOT}/${TASK}/${VARIANT}"
SEED_FILE="${TASK_VARIANT_DIR}/seed.txt"
if [[ ! -f "${SEED_FILE}" ]]; then
  echo "ERROR: no collection seed file: ${SEED_FILE}" >&2
  echo "       (task '${TASK}' has no '${VARIANT}' data, so it has no train-set scenes to replay)" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Derive the replay seed list. Mirrors robotwin2_to_zarr.py: uid =
# "<task>/<variant>/episode<N>", val = first `round(n*val_w/denom)` uids in
# (sha256(uid)[:8] big-endian, uid) order; everything else is train.
SEED_LIST="$(python3 - "${TASK}" "${VARIANT}" "${TASK_VARIANT_DIR}" "${TRAIN_VAL_RATIO}" "${EPISODES}" "${MAX_EPISODES}" <<'PY'
import hashlib, sys
from pathlib import Path

task, variant, task_dir, ratio, episodes_override, max_episodes = sys.argv[1:7]
task_dir = Path(task_dir)
max_episodes = int(max_episodes)

seeds = [int(tok) for tok in (task_dir / "seed.txt").read_text().split()]
n_hdf5 = len(list((task_dir / "data").glob("episode*.hdf5")))
if n_hdf5 != len(seeds):
    sys.exit(f"seed.txt has {len(seeds)} seeds but data/ has {n_hdf5} episodes")

if episodes_override.strip():
    train_eps = [int(x) for x in episodes_override.replace(",", " ").split()]
    bad = [e for e in train_eps if not 0 <= e < len(seeds)]
    if bad:
        sys.exit(f"EPISODES out of range for {len(seeds)} episodes: {bad}")
else:
    train_w, val_w = (int(x) for x in ratio.split(":"))
    denom = train_w + val_w
    uids = [f"{task}/{variant}/episode{i}" for i in range(len(seeds))]
    key = lambda u: (int.from_bytes(hashlib.sha256(u.encode()).digest()[:8], "big"), u)
    val_count = (len(uids) * val_w + denom // 2) // denom
    val = set(sorted(uids, key=key)[:val_count])
    train_eps = [i for i, u in enumerate(uids) if u not in val]

val_eps = sorted(set(range(len(seeds))) - set(train_eps))
n_train = len(train_eps)
if max_episodes and n_train > max_episodes:
    # Evenly spaced over the train episodes, so the subsample spans the whole
    # collection range instead of a prefix of it.
    picked = [train_eps[(i * n_train) // max_episodes] for i in range(max_episodes)]
    assert len(set(picked)) == max_episodes, "even-stride subsample collided"
    train_eps = picked

print(",".join(str(seeds[i]) for i in train_eps))
print(f"episodes({len(train_eps)} of {n_train} train): {','.join(map(str, train_eps))}", file=sys.stderr)
print(f"held-out val episodes({len(val_eps)}): {','.join(map(str, val_eps))}", file=sys.stderr)
PY
)"

NUM_SEEDS="$(awk -F, '{print NF}' <<< "${SEED_LIST}")"

CKPT_ALIAS="$(basename "$(dirname "${CHECKPOINT}")")"
RUN_TAG="${SLURM_JOB_ID:-manual_$$}"
RUN_ID="${RUN_ID:-trainseeds_${TASK}_${SPLIT}_$(date +%Y%m%d_%H%M%S)_${RUN_TAG}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${REPO_ROOT}/eval_logs/robotwin2_trainseeds/${CKPT_ALIAS}}"
SUMMARY_DIR="${SUMMARY_DIR:-${SUMMARY_ROOT}/${RUN_ID}}"
mkdir -p "${SUMMARY_DIR}"

echo -e "\033[95m[trainseeds] task=${TASK} split=${SPLIT} variant=${VARIANT}\033[0m"
echo -e "\033[95m[trainseeds] replaying ${NUM_SEEDS} collection seeds: ${SEED_LIST}\033[0m"
echo -e "\033[95m[trainseeds] checkpoint: ${CHECKPOINT}\033[0m"
echo -e "\033[95m[trainseeds] summary dir: ${SUMMARY_DIR}\033[0m"

unset RW2_OVERFIT_SEED
export RW2_EVAL_SEED_LIST="${SEED_LIST}"

# ALLOW_NON_ZARR_TASKS=True: the 9-task whitelist in eval_robotwin.sh predates
# the 50-task checkpoint. EXPERT_CHECK is forced off inside eval_policy.py when
# a seed list is set; passed here too so the printed config is not misleading.
CHECKPOINT="${CHECKPOINT}" \
GPU_ID=0 \
TEST_NUM="${NUM_SEEDS}" \
EXPERT_CHECK=False \
SAVE_VIDEO="${SAVE_VIDEO}" \
SUMMARY_DIR="${SUMMARY_DIR}" \
ALLOW_NON_ZARR_TASKS=True \
MOTION_EXPERT_CHECKPOINT="${MOTION_EXPERT_CHECKPOINT}" \
EMBEDDING_DIM="${EMBEDDING_DIM}" \
MOTION_EXPERT_CONFIG="${MOTION_EXPERT_CONFIG}" \
bash "${REPO_ROOT}/online_evaluation_robotwin/eval_robotwin_flower_actor_3d_motion_latent.sh" \
  "${SPLIT}" 0 0 "${TASK}" "${SAVE_VIDEO}"

echo -e "\033[32m[trainseeds] done. summary dir: ${SUMMARY_DIR}\033[0m"
