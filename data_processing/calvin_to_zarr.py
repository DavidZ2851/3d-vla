import argparse
import json
import os

from numcodecs import Blosc
import zarr
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf


IM_SIZE = 200
ACTION_LEN = 10
ACTION_DIM = 7
NHAND = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/CALVIN_zarr'),
        ('store_every', int, 1),
        ('store_val', int, 1),
        ('store_instructions', int, 1)
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def get_env(dataset_config, show_gui=True):
    render_conf = OmegaConf.load(dataset_config)
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


def all_tasks_main(env, split):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    # All CALVIN episodes
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{ROOT}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb_front", (1, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("rgb_wrist", (1, 3, 84, 84), "uint8")
        _create("depth_front", (1, IM_SIZE, IM_SIZE), "float16")
        _create("depth_wrist", (1, 84, 84), "float16")
        _create("proprioception", (1, NHAND, ACTION_DIM), "float32")
        _create("action", (ACTION_LEN, NHAND, ACTION_DIM), "float32")
        _create("rel_action", (ACTION_LEN, NHAND, ACTION_DIM), "float32")
        _create("extrinsics_wrist", (4, 4), "float16")
        _create("instr_id", (), "int")

        # Loop through episodes
        for ann_id, (start, end) in tqdm(enumerate(annos['info']['indx'])):
            if ann_id % STORE_EVERY != 0:
                continue
            rgb_front, rgb_wrist, depth_front, depth_wrist = [], [], [], []
            ext_wrist = []
            prop, actions, rel_actions = [], [], []
            # Each episode is split in multiple files
            for ep_id in range(start, end + 1):
                episode = 'episode_{:07d}.npz'.format(ep_id)
                data = np.load(f'{ROOT}/{suffix}/{episode}')
                # Store proprio/actions in EVERY timestep
                prop.append(np.concatenate([
                    data['robot_obs'][:6],
                    (data['robot_obs'][[-1]] > 0).astype(np.float32)  # [0, 1]
                ], axis=-1))
                actions.append(np.concatenate([
                    data['actions'][:6],
                    (data['actions'][[-1]] > 0).astype(np.float32)  # [0, 1]
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
            # Pad actions
            actions = np.ascontiguousarray(actions, dtype=np.float32)
            actions = np.ascontiguousarray([
                actions[i:i+ACTION_LEN] for i in range(_numel)
            ])[:, :, None]
            # Relative actions
            rel_actions = np.ascontiguousarray(rel_actions, dtype=np.float32)
            rel_actions = np.ascontiguousarray([
                rel_actions[i:i+ACTION_LEN] for i in range(_numel)
            ])[:, :, None]

            # Write
            zarr_file['rgb_front'].append(rgb_front)
            zarr_file['rgb_wrist'].append(rgb_wrist)
            zarr_file['depth_front'].append(depth_front)
            zarr_file['depth_wrist'].append(depth_wrist)
            zarr_file['extrinsics_wrist'].append(ext_wrist)
            zarr_file['instr_id'].append(instr_id)
            zarr_file['proprioception'].append(prop)
            zarr_file['action'].append(actions)
            zarr_file['rel_action'].append(rel_actions)
            assert all(
                zarr_file['action'].shape[0] == zarr_file[key].shape[0]
                for key in zarr_file.keys()
            )


def store_instructions(split):
    # All CALVIN episodes
    suffix = 'training' if split == 'train' else 'validation'
    annos = np.load(
        f'{ROOT}/{suffix}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()
    return annos['language']['ann']


if __name__ == "__main__":
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    STORE_EVERY = args.store_every

    env = get_env(
        "online_evaluation_calvin/configs/merged_config_train_abc_d.yaml",
        show_gui=False
    )
    all_tasks_main(env, 'train')
    env.close()
    del env

    if bool(args.store_val):
        env = get_env(
            "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml",
            show_gui=False
        )
        all_tasks_main(env, 'val')
        env.close()
        del env

    # Store instructions as json (can be run independently)
    if bool(args.store_instructions):
        os.makedirs('instructions/calvin', exist_ok=True)
        instr = store_instructions('train')
        with open('instructions/calvin/train_instructions.json', 'w') as fid:
            json.dump(instr, fid)
        instr = store_instructions('val')
        with open('instructions/calvin/val_instructions.json', 'w') as fid:
            json.dump(instr, fid)
