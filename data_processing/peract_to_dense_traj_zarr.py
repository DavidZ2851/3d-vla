import argparse
import json
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageDraw

from data_processing.rlbench_utils import (
    image_to_float_array,
    keypoint_discovery,
    store_instructions
)


NCAM = 2
NHAND = 1
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1


front_camera_extrinsics = np.array([
    [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01, 1.34999919e+00],
    [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07, 3.71546562e-08],
    [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01, 1.57999933e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
])
front_camera_intrinsics = np.array([
    [-351.6771208,    0.       ,  128.       ],
    [   0.       , -351.6771208,  128.       ],
    [   0.       ,    0.       ,    1.       ]
])


def inverse_depth_batched(xyz):
    """Convert point cloud in world frame to depth."""
    intrinsics = front_camera_intrinsics
    extrinsics = front_camera_extrinsics
    # Construct camera projection
    C = extrinsics[:3, 3][None].T  # (3, 1)
    R = extrinsics[:3, :3]  # (3, 3)
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)  # (3, 1)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)  # (3, 4)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)  # (3, 4)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]]
    )  # (4, 4)
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)  # (4, 4)

    # World space to pixel space
    h, w = xyz.shape[:2]
    xyz = np.concatenate([xyz, np.ones((h, w, 1))], -1)
    xyz = np.reshape(xyz, (h * w, -1)).T
    xyz = np.matmul(np.linalg.inv(cam_proj_mat_inv), xyz)
    xyz = np.reshape(xyz.T, (h, w, -1))[..., :3]
    
    # Isolate the influence of depth
    depth = xyz[..., -1]
    depth[depth == 0] = 1e-8
    xy = xyz[..., :2] / depth[..., None]
    return xy


def interpolate_color(start_color, end_color, t):
    return tuple([
        int(start + (end - start) * t)
        for start, end in zip(start_color, end_color)
    ])


def draw_trajectory(img, trajectory, gripper_state, line_width=3, circle_radius=6):
    """
    Draw trajectory as a gradient line from blue to red with a colored circle at the final point.
    
    Args:
        img: PIL Image or numpy array
        trajectory: List of (x, y) tuples representing the trajectory points
        gripper_state: "open", "close", or "UNKNOWN"
        line_width: Width of the trajectory lines
        circle_radius: Radius of the final point circle
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    
    # Convert trajectory points to pixel coordinates (here assumed already scaled)
    pixel_trajectory = [(int(x), int(y)) for x, y in trajectory]
    
    num_segments = len(pixel_trajectory) - 1
    if num_segments < 1:
        return img  # Nothing to draw

    # Gradient colors from blue (0,0,255) to red (255,0,0)
    start_color = (0, 0, 255)
    end_color = (255, 0, 0)

    for i in range(num_segments):
        x1, y1 = pixel_trajectory[i]
        x2, y2 = pixel_trajectory[i + 1]
        t = i / num_segments  # Gradient progress
        color = interpolate_color(start_color, end_color, t)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

    # Draw circle at final point based on gripper state
    if gripper_state != "UNKNOWN":
        final_x, final_y = pixel_trajectory[-1]
        
        if gripper_state == "open":
            circle_colour = "green"
        elif gripper_state == "close":
            circle_colour = "red"
        else:
            circle_colour = None
        
        if circle_colour:
            draw.ellipse(
                (final_x - circle_radius, final_y - circle_radius,
                 final_x + circle_radius, final_y + circle_radius),
                outline=circle_colour,
                width=line_width,
            )

    return img


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/datasets/hiveformer_clean/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/traj_ontop_zarr/')
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

    cameras = ["front", "wrist"]
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
        _create("action", (1, NHAND, 8), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        for task in tasks:
            print(task)
            task_folder = f'{ROOT}/{task}/'
            for var_f in os.listdir(task_folder):
                if var_f.endswith('pkl'):
                    continue
                int_var_ = int(var_f[len('variation'):])
                episode_folder = f'{task_folder}/{var_f}/episodes/'
                n_ = 1 if split != 'train' else 100
                for ep in tqdm(sorted(os.listdir(episode_folder))[:n_]):
                    # Read low-dim file from RLBench
                    folder = f"{episode_folder}/{ep}"
                    ld_file = f"{folder}/low_dim_obs.pkl"
                    with open(ld_file, 'rb') as f:
                        demo = pickle.load(f)

                    # Keypose discovery
                    key_frames = keypoint_discovery(demo, bimanual=False)
                    next_keypose = []
                    k_ind = 0
                    for i in range(len(demo) - 10):
                        if i == key_frames[k_ind]:
                            k_ind += 1
                        next_keypose.append(key_frames[k_ind])
                    next_keypose = np.array(next_keypose)

                    indices = np.arange(len(demo) - 10)

                    # Loop through indices and store:
                    # RGB (frames, cameras, 3, 256, 256)
                    rgb = np.ascontiguousarray([
                        np.stack([
                            np.array(Image.open(
                                f"{folder}/{cam}_rgb/{k}.png"
                            ))
                            for cam in cameras
                        ])
                        for k in indices
                    ])
                    rgb = rgb.transpose(0, 1, 4, 2, 3)

                    # Inpaint the trajectory on the front image
                    states = np.ascontiguousarray([np.concatenate([
                        demo[k].gripper_pose, [demo[k].gripper_open]
                    ]) for k in range(len(demo))]).astype(np.float32)
                    states = states.reshape(-1, 8)
                    for k in range(len(rgb)):
                        xy = inverse_depth_batched(states[k:next_keypose[k], :3][None])[0]
                        # Add small noise
                        xy = xy + np.random.randint(0, 3, size=xy.shape) - 1
                        # Drop some points
                        if len(xy) > 15:
                            xy = xy[np.where(np.random.random((len(xy),)) > 0.12)]
                        if len(xy) > 15:
                            xy = np.concatenate((xy[::15], xy[-1:]))
                        front = draw_trajectory(
                            rgb[k, 0].transpose(1, 2, 0), xy,
                            "open" if states[next_keypose[k], -1] == 1 else "close"
                        )
                        rgb[k, 0] = np.array(front).transpose(2, 0, 1)

                    # Depth (frames, cameras, 256, 256)
                    depth_list = []
                    for k in indices:
                        cam_d = []
                        for cam in cameras:
                            depth = image_to_float_array(Image.open(
                                f"{folder}/{cam}_depth/{k}.png"
                            ), DEPTH_SCALE)
                            near = demo[k].misc[f'{cam}_camera_near']
                            far = demo[k].misc[f'{cam}_camera_far']
                            depth = near + depth * (far - near)
                            cam_d.append(depth)
                        depth_list.append(np.stack(cam_d).astype(np.float16))
                    depth = np.ascontiguousarray(depth_list)

                    # Proprioception (frames, 1, 1, 8)
                    prop = states.reshape(len(states), 1, NHAND, 8)[:len(rgb)]
                    # Action (frames, ACTION_LEN, 1, 8)
                    actions = states[next_keypose].reshape(len(rgb), 1, NHAND, 8).astype(np.float32)

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
                    var_ = np.ascontiguousarray([int(int_var_)] * len(indices))
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


if __name__ == "__main__":
    tasks = [
        'insert_onto_square_peg',
        'meat_off_grill',
        'place_shape_in_shape_sorter',
        # 'place_wine_at_rack_location',
        # 'push_buttons',
        # 'put_groceries_in_cupboard',
        # 'put_money_in_safe',
        # 'reach_and_drag',
        # 'stack_cups',
        # 'turn_tap',
        # 'place_cups'
    ]
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks)
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/peract', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks)
    with open('instructions/rlbench/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)
