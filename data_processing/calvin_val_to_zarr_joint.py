"""Build the CALVIN joint+EE validation zarr.

Same fields/conventions as the train converter (calvin_to_zarr_joint.py), but:
  * Coverage, not head: instead of keeping the first N datapoints, we stride
    evenly across the WHOLE validation set so the ~target_rows kept points are a
    representative sample of every episode region.
  * No shards: the validation set is small, so we do a single pass.
  * Shuffled, chunk_size=8, written straight to a local --out/val.zarr.

Only episodes that contribute at least one strided datapoint are loaded, and
within those episodes only the selected frames pay the rgb/depth + env-reset
cost; robot_obs/rel_actions (tiny) are read for every frame to build the action
horizon.
"""
import argparse
import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
from tqdm import tqdm

from calvin_to_zarr_joint import (
    IM_SIZE, ACTION_LEN, ACTION_DIM, JOINT_DIM, NHAND, SPLIT_CONFIG,
    get_env, get_view_matrix,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', type=str,
        default='/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D')
    parser.add_argument(
        '--out', type=str,
        default='/data/user_data/ziyinx/datasets/calvin_zarr_joint',
        help='Directory to write val.zarr to.')
    # Target number of validation datapoints, spread across the whole set.
    parser.add_argument('--target_rows', type=int, default=8000)
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--store_instructions', type=int, default=1)
    return parser.parse_args()


def select_strided(indx, target_rows):
    """Pick local datapoint indices on a global stride so the union evenly covers
    every episode. Returns ({ann_id: [local_idx, ...]}, total_datapoints)."""
    numels = [max(0, end + 1 - start - ACTION_LEN) for (start, end) in indx]
    total = int(sum(numels))
    stride = max(1, total // max(1, target_rows))
    selected, offset = {}, 0
    for ann_id, n in enumerate(numels):
        if n > 0:
            first = (-offset) % stride          # smallest i>=0 with (offset+i)%stride==0
            sel = list(range(first, n, stride))
            if sel:
                selected[ann_id] = sel
        offset += n
    return selected, total


def process_episode(env, suffix_dir, start, end, sel):
    """Extract the selected datapoints of one episode into per-key row lists,
    mirroring calvin_to_zarr_joint's field construction exactly."""
    n_frames = end + 1 - start
    # Small fields for every frame (needed for the action horizon / anchors).
    robot_obs = np.empty((n_frames, 15), dtype=np.float32)
    rel_raw = np.empty((n_frames, 7), dtype=np.float32)
    for k, ep_id in enumerate(range(start, end + 1)):
        d = np.load(f'{suffix_dir}/episode_{ep_id:07d}.npz')
        robot_obs[k] = d['robot_obs']
        rel_raw[k] = d['rel_actions']

    prop_all = np.concatenate([
        robot_obs[:, :6], (robot_obs[:, [-1]] > 0).astype(np.float32)
    ], axis=-1)                                            # (L, ACTION_DIM)
    pj_all = np.concatenate([
        robot_obs[:, 7:14], (robot_obs[:, [-1]] > 0).astype(np.float32)
    ], axis=-1)                                            # (L, JOINT_DIM)
    rel_act_all = np.concatenate([
        rel_raw[:, :3] * 0.02, rel_raw[:, 3:6] * 0.05,
        (rel_raw[:, [-1]] > 0).astype(np.float32)
    ], axis=-1)                                            # (L, ACTION_DIM)

    rows = {k: [] for k in (
        'rgb_front', 'rgb_wrist', 'depth_front', 'depth_wrist',
        'extrinsics_wrist', 'proprioception', 'proprioception_joint',
        'rel_action', 'rel_joint_action',
    )}
    for i in sel:
        ep_id = start + i
        d = np.load(f'{suffix_dir}/episode_{ep_id:07d}.npz')
        rows['rgb_front'].append(d['rgb_static'].transpose(2, 0, 1)[None])      # (1,3,H,W)
        rows['rgb_wrist'].append(d['rgb_gripper'].transpose(2, 0, 1)[None])     # (1,3,84,84)
        rows['depth_front'].append(d['depth_static'].astype(np.float16)[None])  # (1,H,W)
        rows['depth_wrist'].append(d['depth_gripper'].astype(np.float16)[None]) # (1,84,84)
        env.robot.reset(robot_obs[i])
        ext = np.linalg.inv(
            np.array(get_view_matrix(env)).reshape((4, 4)).T
        ).astype(np.float16)
        rows['extrinsics_wrist'].append(ext)                                    # (4,4)
        rows['proprioception'].append(prop_all[i][None, None])                  # (1,1,ACTION_DIM)
        rows['proprioception_joint'].append(pj_all[i][None, None])             # (1,1,JOINT_DIM)
        rows['rel_action'].append(rel_act_all[i:i + ACTION_LEN][:, None])       # (ACTION_LEN,1,ACTION_DIM)
        rj = pj_all[i + 1:i + 1 + ACTION_LEN].copy()
        rj[:, :7] -= pj_all[i, :7]                                              # anchor-relative joints
        rows['rel_joint_action'].append(rj[:, None])                           # (ACTION_LEN,1,JOINT_DIM)
    return rows


def main():
    args = parse_arguments()
    suffix = 'validation'
    annos = np.load(
        f'{args.root}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    indx = annos['info']['indx']
    selected, total = select_strided(indx, args.target_rows)
    n_sel = sum(len(v) for v in selected.values())
    print(f"[val] total datapoints={total}, target={args.target_rows}, "
          f"selecting {n_sel} across {len(selected)} episodes")

    out = f"{args.out}/val.zarr"
    if os.path.exists(out):
        print(f"{out} already exists, skipping.")
        return
    os.makedirs(args.out, exist_ok=True)

    env = get_env(SPLIT_CONFIG['val'], show_gui=False)
    suffix_dir = f'{args.root}/{suffix}'

    acc = None
    instr_ids = []
    for ann_id in tqdm(sorted(selected), desc="val episodes"):
        start, end = indx[ann_id]
        rows = process_episode(env, suffix_dir, start, end, selected[ann_id])
        if acc is None:
            acc = {k: [] for k in rows}
        for k, v in rows.items():
            acc[k].extend(v)
        instr_ids.extend([ann_id] * len(rows['rgb_front']))
    env.close()

    # Stack, shuffle once, write chunk_size=8.
    data = {k: np.stack(v, axis=0) for k, v in acc.items()}
    data['instr_id'] = np.asarray(instr_ids, dtype=np.int64)
    n = data['instr_id'].shape[0]
    perm = np.random.default_rng(args.seed).permutation(n)
    data = {k: v[perm] for k, v in data.items()}

    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    dst = zarr.open_group(out, mode='w')
    for k, v in data.items():
        dst.create_dataset(
            k, shape=v.shape, chunks=(args.chunk_size,) + v.shape[1:],
            compressor=compressor, dtype=v.dtype,
        )
        dst[k][:] = v
    assert all(dst[k].shape[0] == n for k in dst.array_keys())
    print(f"[val] wrote {n} shuffled rows -> {out}")

    if bool(args.store_instructions):
        os.makedirs('instructions/calvin', exist_ok=True)
        with open('instructions/calvin/val_instructions.json', 'w') as fid:
            json.dump(annos['language']['ann'], fid)


if __name__ == '__main__':
    main()
