#!/usr/bin/env bash
#
# Drip-feed a full RoboTwin2 50-task sweep for one or more checkpoints, in
# DESCENDING task order (turn_switch first, beat_block_hammer last), on either
# the clean or the randomized split.
#
# Why a submitter loop instead of a job array: this account is capped at 50
# submitted jobs per user, and the cap is shared across QOS (normal, katef_qos,
# ... all draw on the same counter), with array elements counting individually.
# The full sweep is 50 tasks x N checkpoints, so it cannot be resident in the
# queue at once. This script keeps the queue filled to MAX_QUEUED and submits
# the next pending (task, checkpoint) pair whenever a slot frees, which also
# preserves the intended ordering: equal-priority jobs start in submit order.
#
# Each pair is submitted as its own job via submit_eval_50_slurm.sh (which
# carries the #SBATCH resource headers: partition general, 1x L40S, 8 cpus,
# 48G, 24h).
#
# Usage (detached, survives logout):
#   nohup setsid env SPLIT=randomized RW2_SWAP_RB=True CHECKPOINTS=<...>.pth \
#     bash online_evaluation_robotwin/queue_eval_50_rev.sh \
#     > /data/user_data/$USER/logs/rw2_rand_queue.log 2>&1 < /dev/null &
#
# Running two of these at once (e.g. clean and randomized) is fine: both simply
# top up to MAX_QUEUED, so they interleave as slots free.
#
# Env overrides:
#   SPLIT         clean | randomized (default clean)
#   CHECKPOINTS   space-separated .pth paths, evaluated in the listed order
#                 within each task (default: rw2_all 240k then 300k)
#   RW2_SWAP_RB   True to feed R/B-swapped images -- required for checkpoints
#                 trained before the datasets/robotwin2.py color fix, i.e.
#                 train_logs/robotwin2/rw2_bad_color/*. See model_wrapper.py.
#   SKIP_TASKS    space-separated task names to leave out (default:
#                 stack_blocks_two, already evaluated separately)
#   MAX_QUEUED    stop submitting at this many of the user's jobs in squeue
#                 (default 48; leaves headroom under the 50 cap for
#                 interactive/other jobs)
#   POLL_SECONDS  seconds between top-up checks (default 180)
#   SEED / NUM_EPISODES  forwarded to submit_eval_50_slurm.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user_data/ziyinx/analogical_manipulation_clean}"
SCRIPT_DIR="${REPO_ROOT}/online_evaluation_robotwin"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/train_logs/robotwin2/rw2_all}"

CHECKPOINTS="${CHECKPOINTS:-${CKPT_ROOT}/240k.pth ${CKPT_ROOT}/300k.pth}"
SPLIT="${SPLIT:-clean}"
RW2_SWAP_RB="${RW2_SWAP_RB:-}"
# `-` not `:-`: an explicitly empty SKIP_TASKS= means "skip nothing" (run all 50),
# which is what the randomized split needs. With `:-` an empty value would
# silently fall back to the default and quietly drop a task from the sweep.
SKIP_TASKS="${SKIP_TASKS-stack_blocks_two}"
PARTITION="${PARTITION:-general}"
MAX_QUEUED="${MAX_QUEUED:-48}"
POLL_SECONDS="${POLL_SECONDS:-180}"
SEED="${SEED:-0}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# Trained-first ordering, identical to submit_eval_50_array_slurm.sh: indices
# 0-14 are the 15 trained tasks, 15-49 the 35 held-out ones. Submitted in
# reverse, so index 49 goes first.
TASKS=(
  beat_block_hammer
  pick_dual_bottles
  move_can_pot
  handover_mic
  stack_bowls_two
  place_phone_stand
  click_alarmclock
  grab_roller
  place_a2b_left
  place_bread_skillet
  scan_object
  handover_block
  open_laptop
  blocks_ranking_rgb
  hanging_mug
  adjust_bottle
  blocks_ranking_size
  click_bell
  dump_bin_bigbin
  lift_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_microwave
  pick_diverse_bottles
  place_a2b_right
  place_bread_basket
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stamp_seal
  turn_switch
)

for ckpt in ${CHECKPOINTS}; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

case "${SPLIT}" in
  clean|randomized) ;;
  *) echo "ERROR: SPLIT must be 'clean' or 'randomized', got '${SPLIT}'." >&2; exit 1 ;;
esac

# --- build the pending queue: task descending, checkpoints in listed order ---
PENDING_TASK=()
PENDING_CKPT=()
for (( ti=${#TASKS[@]}-1; ti>=0; ti-- )); do
  task="${TASKS[${ti}]}"
  skip=0
  for s in ${SKIP_TASKS}; do [[ "${s}" == "${task}" ]] && skip=1; done
  [[ "${skip}" == "1" ]] && continue
  for ckpt in ${CHECKPOINTS}; do
    PENDING_TASK+=("${task}")
    PENDING_CKPT+=("${ckpt}")
  done
done

TOTAL="${#PENDING_TASK[@]}"
echo "[queue] $(date '+%F %T') ${TOTAL} jobs pending (task order: ${TASKS[-1]} -> ${TASKS[0]})"
echo "[queue] split: ${SPLIT}, rw2_swap_rb: ${RW2_SWAP_RB:-False}"
echo "[queue] checkpoints: ${CHECKPOINTS}"
echo "[queue] skipping tasks: ${SKIP_TASKS:-<none>}"
echo "[queue] max queued: ${MAX_QUEUED}, poll: ${POLL_SECONDS}s"

i=0
while (( i < TOTAL )); do
  queued="$(squeue -h -u "${USER}" | wc -l)"
  free=$(( MAX_QUEUED - queued ))

  if (( free <= 0 )); then
    echo "[queue] $(date '+%F %T') queue full (${queued}/${MAX_QUEUED}), $(( TOTAL - i )) left; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  while (( free > 0 && i < TOTAL )); do
    task="${PENDING_TASK[${i}]}"
    ckpt="${PENDING_CKPT[${i}]}"
    # Include the run directory and the split in the alias: a bare "300k" is
    # ambiguous across runs (rw2_all/300k.pth vs rw2_bad_color/300k.pth), which
    # makes job names useless for telling two sweeps apart in squeue.
    alias="$(basename "$(dirname "${ckpt}")")_$(basename "${ckpt}" .pth)"
    jobname="${alias}_${SPLIT:0:4}_${task}"

    # Resume guard: this submitter is long-lived, so it can be killed and
    # relaunched (node expiry, walltime, requeue) mid-sweep. Skip any pair that
    # already has a run, otherwise a restart re-submits the whole list.
    #   - summary dir: created by submit_eval_50_slurm.sh once the job starts
    #   - queued job name: covers pairs submitted but not yet started
    # Only catches jobs submitted under the current naming scheme; pass
    # SKIP_TASKS explicitly when resuming a sweep that used an older one.
    run_dir="$(basename "$(dirname "${ckpt}")")"
    if compgen -G "${REPO_ROOT}/eval_logs/robotwin2_normal/${run_dir}/normal_seed${SEED}_${task}_${SPLIT}_*" > /dev/null 2>&1; then
      echo "[queue] $(date '+%F %T') [$(( i + 1 ))/${TOTAL}] skip ${alias} ${task}: summary dir exists"
      i=$(( i + 1 ))
      continue
    fi
    if squeue -h -u "${USER}" -o "%j" | grep -qxF "${jobname}"; then
      echo "[queue] $(date '+%F %T') [$(( i + 1 ))/${TOTAL}] skip ${alias} ${task}: already queued"
      i=$(( i + 1 ))
      continue
    fi

    if out="$(TASK="${task}" SPLIT="${SPLIT}" SEED="${SEED}" NUM_EPISODES="${NUM_EPISODES}" \
              CHECKPOINT="${ckpt}" RW2_SWAP_RB="${RW2_SWAP_RB}" \
              sbatch --partition="${PARTITION}" \
                --job-name="${jobname}" \
                "${SCRIPT_DIR}/submit_eval_50_slurm.sh" 2>&1)"; then
      echo "[queue] $(date '+%F %T') [$(( i + 1 ))/${TOTAL}] ${alias} ${task} -> ${out}"
      i=$(( i + 1 ))
      free=$(( free - 1 ))
    else
      # Most likely the shared 50-job cap moved under us; back off and retry
      # the same pair on the next poll.
      echo "[queue] $(date '+%F %T') submit rejected for ${alias} ${task}: ${out}"
      break
    fi
  done

  if (( i < TOTAL )); then
    sleep "${POLL_SECONDS}"
  fi
done

echo "[queue] $(date '+%F %T') all ${TOTAL} jobs submitted."
