import argparse
import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, open_dict


IM_SIZE = 200
ACTION_LEN = 10
ACTION_DIM = 7   # EE proprio / rel_action: [tcp_pos(3), tcp_euler(3), gripper(1)]
JOINT_DIM = 8    # joint proprio / rel_joint_action: [arm_joints(7), gripper(1)]
NHAND = 1

# CALVIN robot_obs (15-dim) layout:
#   [0:3]   tcp position (x, y, z)
#   [3:6]   tcp orientation (euler)
#   [6]     gripper width
#   [7:14]  7 arm joint angles      <-- joint-space fields below
#   [14]    gripper action (binary, -1/1)  == robot_obs[-1]

SPLIT_CONFIG = {
    "train": "online_evaluation_calvin/configs/merged_config_train_abc_d.yaml",
    "val": "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', type=str,
        default='/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D')
    parser.add_argument(
        '--tgt', type=str,
        default='/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr_joint')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--store_instructions', type=int, default=1)
    # val: keep only the first `val_max_rows` datapoints across ALL shards (<=0 -> no cap).
    parser.add_argument('--val_max_rows', type=int, default=5000)
    return parser.parse_args()


def get_env(dataset_config, show_gui=True):
    render_conf = OmegaConf.load(dataset_config)
    # The tactile camera instantiates TactileSensor -> tacto -> pyrender -> pyglet,
    # which needs libGLU (unavailable on headless CPU nodes). We only use the gripper
    # camera (cameras[1]) for the wrist extrinsics and never render/use tactile, so
    # drop it before instantiate to avoid that import path entirely.
    cams = render_conf.env.cameras
    if cams is not None and "tactile" in cams:
        with open_dict(cams):
            del cams["tactile"]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(
        render_conf.env,
        show_gui=show_gui, use_vr=False, use_scene_info=True
    )
    return env


def get_view_matrix(env):
    camera_ls = env.p.getLinkState(
        bodyUniqueId=env.cameras[1].robot_uid,
        linkIndex=env.cameras[1].gripper_cam_link,
        physicsClientId=env.cameras[1].cid
    )
    cam_pos, cam_orn = camera_ls[:2]
    cam_rot = env.p.getMatrixFromQuaternion(cam_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    return env.p.computeViewMatrix(cam_pos, cam_pos + cam_rot_y, -cam_rot_z)


def compute_keep_budget(annos, max_rows):
    """Per-episode row budget so the union over all shards == the first `max_rows`
    datapoints (in ann_id order). Deterministic from annotations, so every shard
    computes the same budget without coordinating. Returns {ann_id: keep_rows},
    or None for 'keep everything'."""
    if max_rows is None or max_rows <= 0:
        return None
    keep = {}
    cum = 0
    for ann_id, (start, end) in enumerate(annos['info']['indx']):
        numel = max(0, end + 1 - start - ACTION_LEN)
        if cum >= max_rows:
            keep[ann_id] = 0
        else:
            take = min(numel, max_rows - cum)
            keep[ann_id] = take
            cum += take
    return keep


def all_tasks_main(env, split, root, store_path, num_shards, shard_id, keep_budget):
    os.makedirs(store_path, exist_ok=True)
    filename = f"{store_path}/{split}_shard_{shard_id:02d}_of_{num_shards:02d}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{root}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    n_written = 0
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field, shape=(0,) + shape, chunks=(1,) + shape,
                compressor=compressor, dtype=dtype
            )

        _create("rgb_front", (1, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("rgb_wrist", (1, 3, 84, 84), "uint8")
        _create("depth_front", (1, IM_SIZE, IM_SIZE), "float16")
        _create("depth_wrist", (1, 84, 84), "float16")
        _create("proprioception", (1, NHAND, ACTION_DIM), "float32")
        _create("proprioception_joint", (1, NHAND, JOINT_DIM), "float32")
        _create("rel_action", (ACTION_LEN, NHAND, ACTION_DIM), "float32")
        _create("rel_joint_action", (ACTION_LEN, NHAND, JOINT_DIM), "float32")
        _create("extrinsics_wrist", (4, 4), "float16")
        _create("instr_id", (), "int")

        for ann_id, (start, end) in tqdm(
            enumerate(annos['info']['indx']), desc=f"{split} shard {shard_id}"
        ):
            # Shard assignment: each annotation/episode handled by exactly one job.
            if ann_id % num_shards != shard_id:
                continue
            ep_keep = None
            if keep_budget is not None:
                ep_keep = keep_budget.get(ann_id, 0)
                if ep_keep == 0:
                    continue

            rgb_front, rgb_wrist, depth_front, depth_wrist = [], [], [], []
            ext_wrist = []
            prop, rel_actions, prop_joint = [], [], []
            # Each episode is split in multiple files
            for ep_id in range(start, end + 1):
                episode = 'episode_{:07d}.npz'.format(ep_id)
                data = np.load(f'{root}/{suffix}/{episode}')
                # Store proprio/actions in EVERY timestep
                prop.append(np.concatenate([
                    data['robot_obs'][:6],
                    (data['robot_obs'][[-1]] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1))
                # Joint-space proprio: 7 arm joints + binary gripper.
                prop_joint.append(np.concatenate([
                    data['robot_obs'][7:14],
                    (data['robot_obs'][[-1]] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1))
                rel_actions.append(np.concatenate([
                    data['rel_actions'][:3] * 0.02,
                    data['rel_actions'][3:6] * 0.05,
                    (data['rel_actions'][[-1]] > 0).astype(np.float32)
                ], axis=-1))
                # Only if we're not at the end of episode
                if ep_id >= end + 1 - ACTION_LEN:
                    continue
                rgb_front.append(data['rgb_static'].transpose(2, 0, 1))
                rgb_wrist.append(data['rgb_gripper'].transpose(2, 0, 1))
                depth_front.append(data['depth_static'].astype(np.float16))
                depth_wrist.append(data['depth_gripper'].astype(np.float16))
                # additional logic for storing wrist camera extrinsics
                env.robot.reset(data['robot_obs'])
                ext_wrist.append(np.linalg.inv(
                    np.array(get_view_matrix(env)).reshape((4, 4)).T
                ).astype(np.float16))

            # Merge
            rgb_front = np.ascontiguousarray(rgb_front)[:, None]
            rgb_wrist = np.ascontiguousarray(rgb_wrist)[:, None]
            depth_front = np.ascontiguousarray(depth_front)[:, None]
            depth_wrist = np.ascontiguousarray(depth_wrist)[:, None]
            ext_wrist = np.ascontiguousarray(ext_wrist)
            instr_id = np.ascontiguousarray([ann_id] * len(rgb_front))
            _numel = end + 1 - start - ACTION_LEN
            prop = np.ascontiguousarray(prop, dtype=np.float32)[:_numel, None, None]
            # Relative EE actions (same as the original calvin pipeline)
            rel_actions = np.ascontiguousarray(rel_actions, dtype=np.float32)
            rel_actions = np.ascontiguousarray([
                rel_actions[i:i+ACTION_LEN] for i in range(_numel)
            ])[:, :, None]

            # --- Joint-space fields -------------------------------------------
            pj = np.ascontiguousarray(prop_joint, dtype=np.float32)   # (L, JOINT_DIM)
            proprioception_joint = pj[:_numel, None, None]            # anchor at each step
            # Absolute future joint states over the horizon: rows i+1 .. i+ACTION_LEN.
            rel_joint_action = np.ascontiguousarray([
                pj[i + 1:i + 1 + ACTION_LEN] for i in range(_numel)
            ])                                                        # (_numel, ACTION_LEN, JOINT_DIM)
            # Anchor-relative joints: subtract the chunk-start joint state, the SAME
            # anchor for the whole horizon (NOT incremental). Gripper (dim 7) stays
            # absolute binary, mirroring rel_action.
            rel_joint_action[..., :7] -= pj[:_numel, None, :7]
            rel_joint_action = rel_joint_action[:, :, None]           # (_numel, ACTION_LEN, 1, JOINT_DIM)

            keep = _numel if ep_keep is None else min(_numel, ep_keep)

            # Write
            zarr_file['rgb_front'].append(rgb_front[:keep])
            zarr_file['rgb_wrist'].append(rgb_wrist[:keep])
            zarr_file['depth_front'].append(depth_front[:keep])
            zarr_file['depth_wrist'].append(depth_wrist[:keep])
            zarr_file['extrinsics_wrist'].append(ext_wrist[:keep])
            zarr_file['instr_id'].append(instr_id[:keep])
            zarr_file['proprioception'].append(prop[:keep])
            zarr_file['proprioception_joint'].append(proprioception_joint[:keep])
            zarr_file['rel_action'].append(rel_actions[:keep])
            zarr_file['rel_joint_action'].append(rel_joint_action[:keep])
            n_written += keep
            assert all(
                zarr_file['rel_action'].shape[0] == zarr_file[key].shape[0]
                for key in zarr_file.keys()
            )

    print(f"[{split} shard {shard_id}/{num_shards}] wrote {n_written} rows -> {filename}")


def store_instructions(split, root):
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{root}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    return annos['language']['ann']


if __name__ == "__main__":
    args = parse_arguments()
    assert 0 <= args.shard_id < args.num_shards, "shard_id out of range"

    for split in args.splits:
        suffix = 'training' if split == 'train' else 'validation'
        annos = np.load(
            f'{args.root}/{suffix}/lang_annotations/auto_lang_ann.npy',
            allow_pickle=True
        ).item()
        # Train: whole set. Val: only the first val_max_rows datapoints.
        keep_budget = compute_keep_budget(annos, args.val_max_rows) if split == 'val' else None

        env = get_env(SPLIT_CONFIG[split], show_gui=False)
        all_tasks_main(
            env, split, args.root, args.tgt,
            args.num_shards, args.shard_id, keep_budget,
        )
        env.close()
        del env

    # Instructions json: write once (shard 0 only) to avoid duplicate/racing writes.
    if bool(args.store_instructions) and args.shard_id == 0:
        os.makedirs('instructions/calvin', exist_ok=True)
        for split in args.splits:
            instr = store_instructions(split, args.root)
            name = 'train_instructions.json' if split == 'train' else 'val_instructions.json'
            with open(f'instructions/calvin/{name}', 'w') as fid:
                json.dump(instr, fid)
