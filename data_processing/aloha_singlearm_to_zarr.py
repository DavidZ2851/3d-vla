#!/usr/bin/env python3
"""
Convert the processed single-arm ALOHA LeRobot dataset into the multi-camera zarr
consumed by the flower_actor_3d family (3D front + 3D back + 2D wrist).

This is a thin wrapper over ``aloha_to_zarr.py``: the zarr schema, rot6d / anchor-
relative-delta conventions, calibration handling and per-episode tensor builders
are reused verbatim, so the single-arm zarr is byte-for-byte compatible with the
bimanual one the trainer already expects. The only differences are:

  * Source root      -> /data/user_data/ziyinx/datasets/aloha-singlearm-processed
  * Task layout      -> each of the 5 tasks is a SINGLE folder (no <task>1 split),
                        and gets its own instr_id 0..4 (sorted task order).
  * Output zarr dir  -> /data/user_data/ziyinx/datasets/zarr_datasets/aloha_singlearm_zarr

The parquet columns and video keys are identical to the bimanual set
(observation/action.right_eef_pose, front/back color+transformed_depth, cam_wrist),
so no other change is needed. Camera calibration defaults to the same multiview
front/back rig file; override with --calib if the rig differs.

After this writes train.zarr / val.zarr, run data_processing/rechunk.py with
--chunk_size 8 (and --data_type motion, since the zarr has rel_action, not action)
to produce the train_rechunked8.zarr / val_rechunked8.zarr the trainer loads.
The package_aloha_singlearm.sbatch CPU job does both steps in order.

Usage:
    python aloha_singlearm_to_zarr.py [--overwrite] [--val-per-folder N] [...]
    # all flags from aloha_to_zarr.py are accepted unchanged.
"""

import sys
from pathlib import Path

# Reuse every building block from the bimanual converter; only the dataset's
# task/folder map, default source root and default output dir change.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import aloha_to_zarr as a2z  # noqa: E402


# Override the module globals that aloha_to_zarr.{parse_args,enumerate_episodes,
# load_instructions} read at call time. Each single-arm task is one folder.
a2z.DEFAULT_ROOT = Path("/data/user_data/ziyinx/datasets/aloha-singlearm-processed")
a2z.DEFAULT_OUT = Path("/data/user_data/ziyinx/datasets/zarr_datasets/aloha_singlearm_zarr")
a2z.TASKS = {
    # logical task -> (instr_id, source folders)
    "bottle_upright": (0, ("bottle_upright",)),
    "bottles_triangle": (1, ("bottles_triangle",)),
    "marker_cup": (2, ("marker_cup",)),
    "rank_color": (3, ("rank_color",)),
    "uncap_marker": (4, ("uncap_marker",)),
}


if __name__ == "__main__":
    a2z.main()
