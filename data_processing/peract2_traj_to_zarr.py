import argparse
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import image_to_float_array


NCAM = 3
NHAND = 2
ACTION_LEN = 10
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/peract2_raw_squash/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/Peract2traj_zarr/')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def all_tasks_main(split, tasks):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    cameras = ["front", "wrist_left", "wrist_right"]
    task2id = {task: t for t, task in enumerate(tasks)}

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

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (1, NHAND, 8), "float32")
        _create("action", (ACTION_LEN, NHAND, 8), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        for task in tasks:
            print(task)
            task_folder = f'{ROOT}/{split}/{task}/all_variations/episodes'
            n = len(os.listdir(task_folder)) if split == 'train' else 5
            episodes = sorted(os.listdir(task_folder))[:n]
            for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                indices = np.arange(len(demo) - ACTION_LEN)

                # Loop through indices and store:
                # RGB (frames, cameras, 3, 256, 256)
                rgb = np.ascontiguousarray([
                    np.stack([
                        np.array(Image.open(
                            f"{task_folder}/{ep}/{cam}_rgb/rgb_{_num2id(k)}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in indices
                ])
                rgb = rgb.transpose(0, 1, 4, 2, 3)

                # Depth (frames, cameras, 256, 256)
                depth_list = []
                for k in indices:
                    cam_d = []
                    for cam in cameras:
                        depth = image_to_float_array(Image.open(
                            f"{task_folder}/{ep}/{cam}_depth/depth_{_num2id(k)}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        depth = near + depth * (far - near)
                        cam_d.append(depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                depth = np.ascontiguousarray(depth_list)

                # Proprioception (frames, 1, 2, 8)
                states = np.ascontiguousarray([np.concatenate([
                    demo[k].left.gripper_pose, [demo[k].left.gripper_open],
                    demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                ]) for k in range(len(demo))]).astype(np.float32)
                prop = states.reshape(len(states), 1, NHAND, 8)[:len(rgb)]
                # Action (frames, ACTION_LEN, 2, 8)
                actions = np.ascontiguousarray([
                    states[i+1:i+1+ACTION_LEN] for i in range(len(rgb))
                ]).reshape(len(rgb), ACTION_LEN, NHAND, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                extrinsics = np.ascontiguousarray([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in indices
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                intrinsics = np.ascontiguousarray([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in indices
                ])

                # Task id (keyframes,)
                task_id = np.ascontiguousarray([task2id[task]] * len(indices))
                task_id = task_id.astype(np.uint8)

                # Variation (keyframes,)
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    var_ = pickle.load(f)
                var_ = np.ascontiguousarray([int(var_)] * len(indices))
                var_ = var_.astype(np.uint8)

                # Write
                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(var_)


def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


if __name__ == "__main__":
    tasks = [
        'bimanual_push_box',
        'bimanual_lift_ball',
        'bimanual_dual_push_buttons',
        'bimanual_pick_plate',
        'bimanual_put_item_in_drawer',
        'bimanual_put_bottle_in_fridge',
        'bimanual_handover_item',
        'bimanual_pick_laptop',
        'bimanual_straighten_rope',
        'bimanual_sweep_to_dustpan',
        'bimanual_lift_tray',
        'bimanual_handover_item_easy',
        'bimanual_take_tray_out_of_oven'
    ]
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks)
