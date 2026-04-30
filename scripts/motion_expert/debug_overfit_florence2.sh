#!/bin/bash

set -euo pipefail

REPO_ROOT=/data/user_data/ziyinx/analogical_manipulation_clean
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/motion_expert/configs/calvin_florence2_overfit.yaml}
FLORENCE2_CACHE_SNAPSHOT=/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [ -d "$FLORENCE2_CACHE_SNAPSHOT" ]; then
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-$FLORENCE2_CACHE_SNAPSHOT}
else
  export FLORENCE2_MODEL_PATH=${FLORENCE2_MODEL_PATH:-microsoft/Florence-2-large}
fi

cd "$REPO_ROOT"
python3 -m motion_expert.train --config "$CONFIG_PATH" "$@"
