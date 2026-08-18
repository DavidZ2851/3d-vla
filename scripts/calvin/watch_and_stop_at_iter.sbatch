#!/bin/bash
#SBATCH --job-name=stop_at_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/user_data/%u/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/logs/%x_%j.err
#
# Watches a training run and scancels it once the target-iteration checkpoint is
# fully on disk.
#
# Why this is not just "sleep N; scancel": the trainer overwrites a single
# ~19.5 GB last.pth in place (utils/trainers/base.py:877 keeps the "last.pth"
# name for any iter <= 200000). Killing mid-write destroys BOTH the new
# checkpoint and the previous one. So the watcher waits for three independent
# signals before touching anything:
#
#   1. the log shows the val block for the target iter  (Step <target-1>:)
#   2. last.pth has been rewritten since that log line  (mtime advanced)
#   3. last.pth size is full and has not changed for STABLE_SECS
#
# Only then does it copy the checkpoint aside and cancel. If any signal never
# arrives it exits without cancelling — failing safe means "training continues",
# never "checkpoint destroyed".
#
# Disarm at any time with:  scancel <this job id>

set -uo pipefail

TARGET_JOB=${TARGET_JOB:-9809173}
TARGET_ITER=${TARGET_ITER:-50000}
CKPT_DIR=${CKPT_DIR:-/data/user_data/ziyinx/analogical_manipulation_clean/train_logs/calvin/flower_actor_3d_motion_latent-CalvinContact10th-C960-B8-lr2e-5-tristage_flower-H1-rectified_flow-florence2-motion_finetuned_true-seed42-job9809173}
TRAIN_LOG=${TRAIN_LOG:-/data/user_data/ziyinx/logs/calvin_motion_scene_9809173.out}
CKPT=${CKPT:-$CKPT_DIR/last.pth}
KEEP_COPY=${KEEP_COPY:-1}
MIN_SIZE=${MIN_SIZE:-19000000000}     # full checkpoint is ~19.5 GB
STABLE_SECS=${STABLE_SECS:-600}       # next write is ~2 h away, 10 min is ample
POLL=${POLL:-60}

MARKER="Step $((TARGET_ITER - 1)):"
say() { echo "[$(date '+%F %T')] $*"; }

say "watching job=$TARGET_JOB for iter=$TARGET_ITER"
say "  ckpt=$CKPT"
say "  log =$TRAIN_LOG"
say "  marker='$MARKER'  min_size=$MIN_SIZE  stable=${STABLE_SECS}s"

alive() { squeue -h -j "$TARGET_JOB" -o "%T" 2>/dev/null | grep -q RUNNING; }

# ---- 1. wait for the target-iteration validation block -----------------------
say "phase 1: waiting for '$MARKER' in the training log"
while true; do
    if ! alive; then say "job $TARGET_JOB is no longer RUNNING; nothing to do"; exit 0; fi
    # The trainer prints this marker twice per checkpoint step (train-split eval
    # then val-split eval). The save follows the second one, so require two.
    hits=$(grep -c "^${MARKER}$" "$TRAIN_LOG" 2>/dev/null || echo 0)
    if [ "${hits:-0}" -ge 2 ]; then say "phase 1 done: marker seen ${hits}x"; break; fi
    sleep "$POLL"
done

MARK_TS=$(date +%s)

# ---- 2. wait for last.pth to be rewritten after that ------------------------
say "phase 2: waiting for $CKPT to be rewritten (mtime > $MARK_TS)"
while true; do
    if ! alive; then say "job ended before the checkpoint was rewritten; exiting without action"; exit 0; fi
    mt=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
    if [ "$mt" -ge "$MARK_TS" ]; then say "phase 2 done: mtime=$(date -d @"$mt" '+%F %T')"; break; fi
    sleep "$POLL"
done

# ---- 3. wait for the write to settle ----------------------------------------
say "phase 3: waiting for size >= $MIN_SIZE and ${STABLE_SECS}s without change"
last_sig=""
stable_since=0
while true; do
    if ! alive; then say "job ended during the write; exiting without action"; exit 0; fi
    sz=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    mt=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
    sig="$sz:$mt"
    now=$(date +%s)
    if [ "$sig" != "$last_sig" ]; then
        last_sig="$sig"; stable_since=$now
        say "  still moving: size=$sz"
    elif [ "$sz" -ge "$MIN_SIZE" ] && [ $((now - stable_since)) -ge "$STABLE_SECS" ]; then
        say "phase 3 done: size=$sz stable for $((now - stable_since))s"
        break
    fi
    sleep "$POLL"
done

# ---- 4. preserve, verify, then cancel ---------------------------------------
FINAL_SIZE=$(stat -c %s "$CKPT")
if [ "$KEEP_COPY" = "1" ]; then
    DEST="$CKPT_DIR/${TARGET_ITER}.pth"
    say "copying $CKPT -> $DEST"
    if cp -f "$CKPT" "$DEST"; then
        COPY_SIZE=$(stat -c %s "$DEST" 2>/dev/null || echo 0)
        if [ "$COPY_SIZE" != "$FINAL_SIZE" ]; then
            say "COPY SIZE MISMATCH ($COPY_SIZE != $FINAL_SIZE) - NOT cancelling, leaving training alive"
            exit 1
        fi
        say "copy verified: $COPY_SIZE bytes"
    else
        say "copy FAILED - NOT cancelling, leaving training alive"
        exit 1
    fi
fi

# Re-check the source did not start changing again while we copied.
if [ "$(stat -c %s "$CKPT")" != "$FINAL_SIZE" ]; then
    say "$CKPT changed during the copy - NOT cancelling, leaving training alive"
    exit 1
fi

say "cancelling training job $TARGET_JOB"
scancel "$TARGET_JOB" && say "scancel issued" || say "scancel returned non-zero"
sleep 30
say "final state: $(squeue -h -j "$TARGET_JOB" -o '%T' 2>/dev/null || echo GONE)"
say "checkpoint: $CKPT ($FINAL_SIZE bytes)"
[ "$KEEP_COPY" = "1" ] && say "copy: $CKPT_DIR/${TARGET_ITER}.pth"
say "done"
