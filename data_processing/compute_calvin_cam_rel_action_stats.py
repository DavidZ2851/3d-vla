"""Measure the camera-frame relative-action distribution for CALVIN.

The world-frame CALVIN policy uses a hardcoded workspace normalizer of
+/-0.02 (xyz) and +/-0.05 (Euler), because ``calvin_to_zarr.py`` writes
``rel_actions[:3] * 0.02`` / ``rel_actions[3:6] * 0.05`` from CALVIN's own
already-clipped-to-[-1, 1] deltas -- so +/-0.02 / +/-0.05 IS the exact range.

Re-anchoring to the static-camera frame changes that range:

  * positions rotate rigidly, so the exact per-axis max is analytic:
    ``0.02 * sum_j |R_cam_world[i, j]|`` (see ``position_delta_bound_cam``);
  * Euler deltas do NOT rotate rigidly -- the camera-frame TCP orientations sit
    closer to the ``e_y = +/-pi/2`` Euler singularity, which amplifies the
    angular rates -- so the rotation range has to be measured.

Usage:
    python data_processing/compute_calvin_cam_rel_action_stats.py \
        --zarr /data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/train_rechunked8.zarr \
        --num_rows 200000 --out datasets/calvin_cam_rel_action_stats.json
"""

import argparse
import json

import numpy as np
import torch
import zarr

from utils.calvin_camera_frame import (
    euler_to_matrix,
    get_frame_transforms,
    position_delta_bound_frame,
    pose_world_to_frame,
    rel_action_frame_to_abs_world,
    rel_action_world_to_frame,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--zarr', type=str,
        default='/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/train_rechunked8.zarr'
    )
    parser.add_argument('--num_rows', type=int, default=200000)
    parser.add_argument('--block', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--margin', type=float, default=1.05)
    parser.add_argument('--out', type=str, default='datasets/calvin_cam_rel_action_stats.json')
    parser.add_argument(
        '--frame', type=str, default='calvin_static',
        help=(
            "Fixed frame to measure. 'calvin_static' reproduces the original "
            "camera-frame numbers; 'calvin_base_yaw45' measures the robot-base "
            "yaw ablation frame."
        )
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    world2frame, frame2world = get_frame_transforms(args.frame)
    if world2frame is None:
        raise SystemExit(
            "--frame world is the identity; its normalizer is the hardcoded "
            "+/-0.02 / +/-0.05 in CALVINTrainTester.get_workspace_normalizer."
        )
    group = zarr.open(args.zarr, 'r')
    total = group['rel_action'].shape[0]
    num_rows = min(args.num_rows, total)

    # The rechunked8 store is row-shuffled, so contiguous blocks are already a
    # random sample -- and contiguous reads are orders of magnitude faster than
    # fancy indexing over NFS.
    rng = np.random.default_rng(args.seed)
    num_blocks = max(1, num_rows // args.block)
    starts = rng.choice(
        max(1, total - args.block), size=num_blocks, replace=False
    )

    abs_pos, abs_rot = [], []
    max_round_trip_pos, max_round_trip_rot = 0.0, 0.0
    seen = 0
    for start in np.sort(starts):
        stop = min(int(start) + args.block, total)
        rel = torch.from_numpy(
            np.asarray(group['rel_action'][int(start):stop])
        ).double()
        prop = torch.from_numpy(
            np.asarray(group['proprioception'][int(start):stop])
        ).double()
        rel_cam = rel_action_world_to_frame(rel, prop, world2frame)

        # Validate the transform is exactly invertible on every block we touch.
        abs_world = torch.cat([
            prop[:, -1:][..., :6] + torch.cumsum(rel[..., :6], 1),
            rel[..., 6:],
        ], -1)
        abs_world_rt = rel_action_frame_to_abs_world(
            rel_cam, pose_world_to_frame(prop[:, -1:], world2frame), frame2world
        )
        max_round_trip_pos = max(max_round_trip_pos, float(
            (abs_world_rt[..., :3] - abs_world[..., :3]).abs().max()
        ))
        max_round_trip_rot = max(max_round_trip_rot, float(
            (euler_to_matrix(abs_world_rt[..., 3:6])
             - euler_to_matrix(abs_world[..., 3:6])).abs().max()
        ))

        flat = rel_cam[..., :6].reshape(-1, 6).abs().numpy()
        abs_pos.append(flat[:, :3])
        abs_rot.append(flat[:, 3:])
        seen += stop - int(start)
        print(f"  {seen}/{num_rows} rows", flush=True)

    abs_pos = np.concatenate(abs_pos)
    abs_rot = np.concatenate(abs_rot)

    pos_bound = position_delta_bound_frame(world2frame)
    rot_max = abs_rot.max(0)
    stats = {
        "zarr": args.zarr,
        "frame": args.frame,
        "rows_sampled": int(seen),
        "steps_sampled": int(abs_pos.shape[0]),
        "round_trip_max_err": {
            "position_m": max_round_trip_pos,
            "rotation_matrix": max_round_trip_rot,
        },
        "position": {
            "measured_p99.9": np.percentile(abs_pos, 99.9, axis=0).tolist(),
            "measured_max": abs_pos.max(0).tolist(),
            "analytic_bound": pos_bound.tolist(),
        },
        "rotation": {
            "measured_p99.9": np.percentile(abs_rot, 99.9, axis=0).tolist(),
            "measured_max": rot_max.tolist(),
        },
        # What CALVINCamTrainTester.get_workspace_normalizer should return.
        # Positions use the analytic bound (a guarantee, not a sample estimate);
        # rotations use the measured max with a small margin.
        "workspace_normalizer_hi": (
            pos_bound.tolist() + (rot_max * args.margin).tolist()
        ),
    }
    print(json.dumps(stats, indent=2))
    with open(args.out, 'w') as handle:
        json.dump(stats, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
