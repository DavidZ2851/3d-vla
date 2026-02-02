import argparse,json, os, zarr, numpy as np, cv2, imageio.v3 as iio, av
from pathlib import Path
from typing import Dict, List, Optional
from numcodecs import Blosc
import pyarrow.parquet as pq
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

IM_SIZE = 256  # Resize images to this size
ACTION_LEN = 12  # Number of future action steps
ACTION_DIM = 8  # [x, y, z, qx, qy, qz, qw, gripper]
NHAND = 2  # Bimanual
NCAM = 3  # head_front, hand_left, hand_right
DOWNSAMPLE_FACTOR = 3  # Downsample from 30fps to 10fps (keep every 3rd frame)


def parse_arguments():
    parser = argparse.ArgumentParser()
    arguments = [
        ('root', str, '/data/group_data/katefgroup/datasets/ziyinx/geniesim/bimanual_hold_ball/g2'),
        ('tgt', str, '/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr'),
        ('store_every', int, 1),
        ('store_val', int, 1),
        ('store_instructions', int, 1),
        ('split', str, 'train'),
        ('train_ratio', float, 0.66),  # Ratio for train/val split if split not in info.json
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])
    return parser.parse_args()


def load_video(video_path: Path) -> Optional[np.ndarray]:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IM_SIZE, IM_SIZE))
        frames.append(frame)
    cap.release()
    return np.array(frames) if frames else None


def load_depth_video(video_path: Path):
    if not video_path.exists():
        return None
    frames = []
    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            raw_buffer = frame.planes[0]
            np_frame = np.frombuffer(raw_buffer, dtype='>u2')
            np_frame = np_frame.reshape((frame.height, frame.width))
            depth_meters = np_frame.astype(np.float32) / 1000.0
            depth_meters = cv2.resize(depth_meters, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_NEAREST)
            frames.append(depth_meters)
        container.close()
        if not frames:
            return None
        return np.array(frames)
    except Exception as e:
        print(f"Error reading depth video with PyAV: {e}")
        return None


def quaternion_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format."""
    if len(q_xyzw.shape) == 1:
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    else:
        return np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=-1)


def extract_ee_pose_from_state(state: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract end-effector poses from state vector.
    - Position: 14-19 (left: 14-16, right: 17-19)
    - Orientation: 32-39 (left: 32-35, right: 36-39) - 8 dims (2 quaternions in xyzw format)
    """
    left_openness = state[0]
    right_openness = state[1]
    left_pos = state[14:17]  # x, y, z
    right_pos = state[17:20]  # x, y, z
    base_pos = state[118:121]  # x, y, z
    # Format: [qx_l, qy_l, qz_l, qw_l, qx_r, qy_r, qz_r, qw_r]
    left_quat_xyzw = state[32:36]  # qx, qy, qz, qw
    right_quat_xyzw = state[36:40]  # qx, qy, qz, qw
    base_quat_xyzw = state[121:125]  # qx, qy, qz, qw
    left_rot = R.from_quat(left_quat_xyzw).as_matrix()
    right_rot = R.from_quat(right_quat_xyzw).as_matrix()
    R_ee_tool = (R.from_euler('y', -90, degrees=True).as_matrix()
            @ R.from_euler('x', -90, degrees=True).as_matrix())
    left_rot = left_rot @ R_ee_tool
    right_rot = right_rot @ R_ee_tool
    left_quat_xyzw = R.from_matrix(left_rot).as_quat()
    right_quat_xyzw = R.from_matrix(right_rot).as_quat()
    # Convert to wxyz format for consistency with the codebase
    left_quat_wxyz = quaternion_xyzw_to_wxyz(left_quat_xyzw)
    right_quat_wxyz = quaternion_xyzw_to_wxyz(right_quat_xyzw)
    base_quat_wxyz = quaternion_xyzw_to_wxyz(base_quat_xyzw)
    left_quat_wxyz /= np.linalg.norm(left_quat_wxyz)
    right_quat_wxyz /= np.linalg.norm(right_quat_wxyz)
    base_quat_wxyz /= np.linalg.norm(base_quat_wxyz)
    # Combine gripper position and quaternion (8 dims: x, y, z, w, x, y, z, openness)
    # Convert scalars to 1D arrays for concatenation
    left_openness = np.array([left_openness])
    right_openness = np.array([right_openness])
    left_pose = np.concatenate([left_pos, left_quat_wxyz, left_openness])
    right_pose = np.concatenate([right_pos, right_quat_wxyz, right_openness])
    base_pose = np.concatenate([base_pos, base_quat_wxyz])  # (7 dims: x, y, z, w, x, y, z)
    return {
        'left': left_pose,  # [x, y, z, w, x, y, z, openness]
        'right': right_pose,  # [x, y, z, w, x, y, z, openness]
        'base': base_pose  # [x, y, z, w, x, y, z]
    }


def extract_action_from_action(action: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract end-effector actions from action vector.
    - Position: 2-7 (left: 2-4, right: 5-7)
    - Orientation: 8-15 (left: 8-11, right: 12-15) - 8 dims (2 quaternions in xyzw format)
    """
    left_pos = action[2:5]  # x, y, z
    right_pos = action[5:8]  # x, y, z
    left_quat_xyzw = action[8:12]  # qx, qy, qz, qw
    right_quat_xyzw = action[12:16]  # qx, qy, qz, qw
    left_rot = R.from_quat(left_quat_xyzw).as_matrix()
    right_rot = R.from_quat(right_quat_xyzw).as_matrix()
    R_ee_tool = (R.from_euler('y', -90, degrees=True).as_matrix()
            @ R.from_euler('x', -90, degrees=True).as_matrix())
    left_rot = left_rot @ R_ee_tool
    right_rot = right_rot @ R_ee_tool
    left_quat_xyzw = R.from_matrix(left_rot).as_quat()
    right_quat_xyzw = R.from_matrix(right_rot).as_quat()
    # Convert to wxyz format
    left_quat_wxyz = quaternion_xyzw_to_wxyz(left_quat_xyzw)
    right_quat_wxyz = quaternion_xyzw_to_wxyz(right_quat_xyzw)
    # Gripper: Use action/left_effector/position (index 0) and action/right_effector/position (index 1)
    left_gripper_val = action[0]
    right_gripper_val = action[1]
    # If values are outside [0, 1], use sigmoid or clamp
    if left_gripper_val < 0 or left_gripper_val > 1:
        left_gripper_val = 1.0 / (1.0 + np.exp(-left_gripper_val))
    if right_gripper_val < 0 or right_gripper_val > 1:
        right_gripper_val = 1.0 / (1.0 + np.exp(-right_gripper_val))
    left_gripper = np.array([float(left_gripper_val)])
    right_gripper = np.array([float(right_gripper_val)])
    # Combine: [x, y, z, w, x, y, z, gripper]
    left_action = np.concatenate([left_pos, left_quat_wxyz, left_gripper])
    right_action = np.concatenate([right_pos, right_quat_wxyz, right_gripper])
    base_velocity = action[38:40]  # x, y
    return {
        'left': left_action,
        'right': right_action,
        'base': base_velocity
    }


def extract_camera_extrinsics(state: np.ndarray) -> Dict[str, np.ndarray]:
    # Hand left camera
    hand_left_rot = state[126:135].reshape(3, 3)
    hand_left_trans = state[171:174]
    hand_left_T = np.eye(4, dtype=np.float32)
    hand_left_T[:3, :3] = hand_left_rot
    hand_left_T[:3, 3] = hand_left_trans
    # Hand right camera
    hand_right_rot = state[135:144].reshape(3, 3)
    hand_right_trans = state[174:177]
    hand_right_T = np.eye(4, dtype=np.float32)
    hand_right_T[:3, :3] = hand_right_rot
    hand_right_T[:3, 3] = hand_right_trans
    # Head front camera
    head_front_rot = state[162:171].reshape(3, 3)
    head_front_trans = state[183:186]
    head_front_T = np.eye(4, dtype=np.float32)
    head_front_T[:3, :3] = head_front_rot
    head_front_T[:3, 3] = head_front_trans
    return {
        'hand_left': hand_left_T,
        'hand_right': hand_right_T,
        'head_front': head_front_T
    }


def get_camera_intrinsics(info: Dict, camera_key: str, episode_idx: int = 0) -> Optional[np.ndarray]:
    """Get camera intrinsics from info.json. Camera parameters are stored per episode."""
    camera_params = info.get('camera_parameters', {})
    
    # Try to get intrinsics from the specified episode, or first available episode
    episode_key = str(episode_idx)
    if episode_key not in camera_params:
        # Try first available episode
        if len(camera_params) > 0:
            episode_key = list(camera_params.keys())[0]
        else:
            return None
    
    episode_params = camera_params[episode_key]
    intrinsic_key = f'intrinsic_{camera_key}_rgb'
    if intrinsic_key not in episode_params:
        intrinsic_key = f'intrinsic_{camera_key}_depth'
    if intrinsic_key not in episode_params:
        return None
    
    params = episode_params[intrinsic_key]
    if isinstance(params, dict):
        fx = params.get('Fx', 306)
        fy = params.get('Fy', 306)
        cx = params.get('Cx', 320)
        cy = params.get('Cy', 200)
    else:
        # Fallback to defaults
        fx = fy = 306
        cx = 320
        cy = 200
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def process_episode(episode_path: Path, task_path: Path, info: Dict, 
                   chunk_idx: int, episode_idx: int) -> Optional[Dict]:
    """Process a single episode and return data dictionary."""
    df = pq.read_table(episode_path).to_pandas()
    
    # Extract state and action arrays
    states = np.stack([np.array(s) for s in df['observation.state'].values])
    actions = np.stack([np.array(a) for a in df['action'].values])
    num_frames = len(states)

    # Load videos
    video_base = task_path / "videos" / f"chunk-{chunk_idx:03d}"
    # RGB videos
    head_rgb_path = video_base / "observation.images.top_head" / f"episode_{episode_idx:06d}.mp4"
    hand_left_rgb_path = video_base / "observation.images.hand_left" / f"episode_{episode_idx:06d}.mp4"
    hand_right_rgb_path = video_base / "observation.images.hand_right" / f"episode_{episode_idx:06d}.mp4"
    head_rgb = load_video(head_rgb_path)
    hand_left_rgb = load_video(hand_left_rgb_path)
    hand_right_rgb = load_video(hand_right_rgb_path)
    # Depth videos
    head_depth_path = video_base / "observation.images.head_depth" / f"episode_{episode_idx:06d}.mp4"
    hand_left_depth_path = video_base / "observation.images.hand_left_depth" / f"episode_{episode_idx:06d}.mp4"
    hand_right_depth_path = video_base / "observation.images.hand_right_depth" / f"episode_{episode_idx:06d}.mp4"
    head_depth = load_depth_video(head_depth_path)
    hand_left_depth = load_depth_video(hand_left_depth_path)
    hand_right_depth = load_depth_video(hand_right_depth_path)
    
    # Ensure all videos have the same number of frames
    min_frames = min(
        len(head_rgb) if head_rgb is not None else num_frames,
        len(hand_left_rgb) if hand_left_rgb is not None else num_frames,
        len(hand_right_rgb) if hand_right_rgb is not None else num_frames,
        len(head_depth) if head_depth is not None else num_frames,
        len(hand_left_depth) if hand_left_depth is not None else num_frames,
        len(hand_right_depth) if hand_right_depth is not None else num_frames,
        num_frames
    )
    # Truncate to minimum frames
    states = states[:min_frames]
    actions = actions[:min_frames]
    if head_rgb is not None:
        head_rgb = head_rgb[:min_frames]
    if hand_left_rgb is not None:
        hand_left_rgb = hand_left_rgb[:min_frames]
    if hand_right_rgb is not None:
        hand_right_rgb = hand_right_rgb[:min_frames]
    if head_depth is not None:
        head_depth = head_depth[:min_frames]
    if hand_left_depth is not None:
        hand_left_depth = hand_left_depth[:min_frames]
    if hand_right_depth is not None:
        hand_right_depth = hand_right_depth[:min_frames]
    
    # Downsample from 30fps to 10fps (keep every 3rd frame)
    # Create indices for downsampling: [0, 3, 6, 9, ...]
    downsample_indices = np.arange(0, min_frames, DOWNSAMPLE_FACTOR)
    
    # Apply downsampling to all data
    states = states[downsample_indices]
    actions = actions[downsample_indices]
    if head_rgb is not None:
        head_rgb = head_rgb[downsample_indices]
    if hand_left_rgb is not None:
        hand_left_rgb = hand_left_rgb[downsample_indices]
    if hand_right_rgb is not None:
        hand_right_rgb = hand_right_rgb[downsample_indices]
    if head_depth is not None:
        head_depth = head_depth[downsample_indices]
    if hand_left_depth is not None:
        hand_left_depth = hand_left_depth[downsample_indices]
    if hand_right_depth is not None:
        hand_right_depth = hand_right_depth[downsample_indices]
    
    # Update min_frames after downsampling
    min_frames = len(downsample_indices)
    
    # Extract end-effector poses and actions and camera extrinsics
    ee_poses = []
    base_poses = []
    action_poses = []
    base_velocities = []
    extrinsics_list = []
    for i in range(min_frames):
        # Extract poses
        poses = extract_ee_pose_from_state(states[i])
        ee_poses.append([poses['left'], poses['right']])
        base_poses.append(poses['base'])
        # Extract actions
        act_poses = extract_action_from_action(actions[i])
        action_poses.append([act_poses['left'], act_poses['right']])
        base_velocities.append(act_poses['base'])
        # Extract extrinsics
        ext = extract_camera_extrinsics(states[i])
        extrinsics_list.append([
            ext['head_front'],
            ext['hand_left'],
            ext['hand_right']
        ])
    # Convert to numpy arrays
    ee_poses = np.array(ee_poses, dtype=np.float32)  # (T, 2, 8) - [x, y, z, w, x, y, z, openness] for each hand
    base_poses = np.array(base_poses, dtype=np.float32)  # (T, 7) - [x, y, z, w, x, y, z] for base
    action_poses = np.array(action_poses, dtype=np.float32)  # (T, 2, 8) - [x, y, z, w, x, y, z, gripper] for each hand
    base_velocities = np.array(base_velocities, dtype=np.float32)  # (T, 2)
    extrinsics = np.array(extrinsics_list, dtype=np.float32)  # (T, 3, 4, 4)
    
    # Prepare RGB: (T, 3, H, W, 3) -> (T, 3, 3, H, W)
    rgb_head = head_rgb.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    rgb_hand_left = hand_left_rgb.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    rgb_hand_right = hand_right_rgb.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    rgb = np.stack([rgb_head, rgb_hand_left, rgb_hand_right], axis=1)  # (T, 3, 3, H, W)
    # Prepare depth: (T, H, W) -> (T, 3, H, W)
    depth_head = head_depth if head_depth is not None else np.zeros((min_frames, IM_SIZE, IM_SIZE), dtype=np.float16)
    depth_hand_left = hand_left_depth if hand_left_depth is not None else np.zeros((min_frames, IM_SIZE, IM_SIZE), dtype=np.float16)
    depth_hand_right = hand_right_depth if hand_right_depth is not None else np.zeros((min_frames, IM_SIZE, IM_SIZE), dtype=np.float16)
    depth = np.stack([depth_head, depth_hand_left, depth_hand_right], axis=1)  # (T, 3, H, W)
    depth = depth.astype(np.float16)
    
    # Get camera intrinsics (use episode 0 as reference, intrinsics should be constant)
    intrinsics_head = get_camera_intrinsics(info, 'head_front', episode_idx=episode_idx)
    intrinsics_hand_left = get_camera_intrinsics(info, 'hand_left', episode_idx=episode_idx)
    intrinsics_hand_right = get_camera_intrinsics(info, 'hand_right', episode_idx=episode_idx)
    h_head, w_head = info["features"]["observation.images.top_head"]["shape"][:2]
    h_hand_right, w_hand_right = info["features"]["observation.images.hand_right"]["shape"][:2]
    h_hand_left, w_hand_left = info["features"]["observation.images.hand_left"]["shape"][:2]

    def rescale_intrinsics(K, W0, H0):
        sx = IM_SIZE / W0
        sy = IM_SIZE / H0
        K = K.copy()
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        K[0, 2] *= sx  # cx
        K[1, 2] *= sy  # cy
        return K
    
    intrinsics_head = rescale_intrinsics(intrinsics_head, w_head, h_head)
    intrinsics_hand_left = rescale_intrinsics(intrinsics_hand_left, w_hand_left, h_hand_left)
    intrinsics_hand_right = rescale_intrinsics(intrinsics_hand_right, w_hand_right, h_hand_right)
    # Ensure all intrinsics are 3x3 matrices
    assert intrinsics_head.shape == (3, 3), f"Expected (3, 3), got {intrinsics_head.shape}"
    assert intrinsics_hand_left.shape == (3, 3), f"Expected (3, 3), got {intrinsics_hand_left.shape}"
    assert intrinsics_hand_right.shape == (3, 3), f"Expected (3, 3), got {intrinsics_hand_right.shape}"
    
    # Stack intrinsics: (3, 3, 3) -> (3, 3, 3)
    intrinsics = np.stack([intrinsics_head, intrinsics_hand_left, intrinsics_hand_right], axis=0)  # (3, 3, 3)
    # Tile to match number of frames: (T, 3, 3, 3)
    intrinsics = np.tile(intrinsics[None, :, :, :], (min_frames, 1, 1, 1))  # (T, 3, 3, 3)
    
    # Create proprioception: (T, nhist, 3, 8)
    # Store current pose and two previous poses as history
    prop = np.zeros((min_frames, 3, NHAND+1, ACTION_DIM), dtype=np.float32)
    for i in range(min_frames):
        # Current pose (most recent)
        prop[i, 2, 0] = ee_poses[i, 0]  # left
        prop[i, 2, 1] = ee_poses[i, 1]  # right
        prop[i, 2, 2, :7] = base_poses[i]  # base
        # Previous poses (if available)
        if i > 0:
            prop[i, 1, 0] = ee_poses[i-1, 0]
            prop[i, 1, 1] = ee_poses[i-1, 1]
            prop[i, 1, 2, :7] = base_poses[i-1]  # base
        else:
            prop[i, 1, 0] = ee_poses[i, 0]  # Use current if no previous
            prop[i, 1, 1] = ee_poses[i, 1]
            prop[i, 1, 2, :7] = base_poses[i]  # base
        if i > 1:
            prop[i, 0, 0] = ee_poses[i-2, 0]
            prop[i, 0, 1] = ee_poses[i-2, 1]
            prop[i, 0, 2, :7] = base_poses[i-2]  # base
        else:
            prop[i, 0, 0] = ee_poses[i, 0]  # Use current if no previous
            prop[i, 0, 1] = ee_poses[i, 1]
            prop[i, 0, 2, :7] = base_poses[i]  # base
    
    # Create action sequences: (T, ACTION_LEN, 3, 8)
    action_seqs = np.zeros((min_frames, ACTION_LEN, NHAND+1, ACTION_DIM), dtype=np.float32)
    for i in range(min_frames):
        for j in range(ACTION_LEN):
            idx = min(i + j, min_frames - 1)
            action_seqs[i, j, :2] = action_poses[idx]
            action_seqs[i, j, 2, :2] = base_velocities[idx]
    # Only return frames that have enough future actions
    valid_frames = min_frames - ACTION_LEN + 1
    if valid_frames <= 0:
        return None
    
    return {
        'rgb': rgb[:valid_frames],  # (T, 3, 3, H, W)
        'depth': depth[:valid_frames],  # (T, 3, H, W)
        'proprioception': prop[:valid_frames],  # (T, 3, 3, 8)
        'action': action_seqs[:valid_frames],  # (T, ACTION_LEN, 3, 8)
        'extrinsics': extrinsics[:valid_frames],  # (T, 3, 4, 4)
        'intrinsics': intrinsics[:valid_frames],  # (T, 3, 3, 3)
    }


def split_episodes(episodes: List[Dict], train_ratio: float = 0.9, random_seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Args:
        episodes: List of episode dictionaries
        train_ratio: Ratio of episodes to use for training (default 0.9 = 90%)
        random_seed: Random seed for reproducibility
    Returns: Dictionary with 'train' and 'val' keys containing split episodes
    """
    np.random.seed(random_seed)
    num_episodes = len(episodes)
    num_train = int(num_episodes * train_ratio)
    
    # Shuffle episodes
    indices = np.arange(num_episodes)
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]
    
    print(f"Split {num_episodes} episodes into {len(train_episodes)} train and {len(val_episodes)} val")
    
    return {
        'train': train_episodes,
        'val': val_episodes
    }


def load_all_episodes(root: Path) -> List[Dict]:
    """Load all episodes info from the dataset."""
    episodes = []
    data_dir = root / "data"
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        chunk_idx = int(chunk_dir.name.split('-')[1])
        for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
            episode_idx = int(parquet_file.stem.split('_')[1])
            episodes.append({
                'episode_index': episode_idx,
                'chunk_index': chunk_idx,
                'parquet_path': str(parquet_file)
            })
    return episodes


def all_episodes_main(root: Path, split: str, info: Dict, episodes: Optional[List[Dict]] = None, train_ratio: float = 0.9):
    """Process all episodes and save to zarr."""
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None
    # Load episodes info
    if episodes is None:
        episodes = load_all_episodes(root)
    # Filter by split from info.json
    splits = info.get('splits', {})
    split_range = splits[split]
    start, end = map(int, split_range.split(':'))
    episodes = episodes[start:end]
        
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
        _create("proprioception", (3, NHAND+1, ACTION_DIM), "float32")
        _create("action", (ACTION_LEN, NHAND+1, ACTION_DIM), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (1,), "int")
        _create("instr_id", (1,), "int")  # Instruction index for mapping to instructions list
        
        # Process episodes
        episode_counter = 0  # Track episode index within the filtered split
        for ep_info in tqdm(episodes):
            if ep_info.get('episode_index') % STORE_EVERY != 0:
                continue
            episode_idx = ep_info.get('episode_index', 0)
            chunk_idx = ep_info.get('chunk_index', 0)
            if 'parquet_path' in ep_info:
                parquet_path = Path(ep_info['parquet_path'])
            else:
                parquet_path = root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
            if not parquet_path.exists():
                continue
            # Process episode
            data = process_episode(parquet_path, root, info, chunk_idx, episode_idx)
            if data is None:
                continue
            
            # Append to zarr
            num_frames = len(data['rgb'])
            # Get instruction index for this episode
            # episode_counter tracks the episode index within the filtered split
            instr_idx = episode_counter
            for i in range(num_frames):
                zarr_file['rgb'].append(data['rgb'][i:i+1])  # (1, 3, 3, H, W)
                zarr_file['depth'].append(data['depth'][i:i+1])  # (1, 3, H, W)
                zarr_file['proprioception'].append(data['proprioception'][i:i+1])  # (1, 3, 3, 8)
                zarr_file['action'].append(data['action'][i:i+1])  # (1, ACTION_LEN, 3, 8)
                zarr_file['extrinsics'].append(data['extrinsics'][i:i+1])  # (1, 3, 4, 4)
                zarr_file['intrinsics'].append(data['intrinsics'][i:i+1])  # (1, 3, 3, 3)
                zarr_file['task_id'].append(np.array([[0]]))  # Placeholder task_id
                zarr_file['instr_id'].append(np.array([[instr_idx]]))  # Instruction index for this episode
            episode_counter += 1  # Increment after processing episode
            # Verify consistency
            assert all(
                zarr_file['action'].shape[0] == zarr_file[key].shape[0]
                for key in ['rgb', 'depth', 'proprioception', 'action', 'extrinsics', 'intrinsics', 'task_id', 'instr_id']
            )


def store_instructions(split: str, info: Dict):
    """Store instructions as JSON, matching calvin format."""
    instruction_segments = info.get('instruction_segments', {})
    splits = info.get('splits', {})
    split_range = splits[split]
    start, end = map(int, split_range.split(':'))
    # Extract instructions for episodes in this split
    instructions = []
    for episode_idx in range(start, end):
        episode_key = str(episode_idx)
        if episode_key in instruction_segments:
            segments = instruction_segments[episode_key]
            if segments and len(segments) > 0:
                instruction_text = segments[0].get('instruction', '')
                instructions.append(instruction_text)
            else:
                instructions.append('')
        else:
            instructions.append('')
    return instructions


if __name__ == "__main__":
    args = parse_arguments()
    ROOT = Path(args.root)
    STORE_PATH = args.tgt
    STORE_EVERY = args.store_every
    
    # Load info.json and create output directory
    info_path = ROOT / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    os.makedirs(STORE_PATH, exist_ok=True)
    
    # Load all episodes once
    all_episodes = load_all_episodes(ROOT)
    
    # Check if split is defined in info.json
    splits = info.get('splits', {})
    has_predefined_split = 'train' in splits or 'val' in splits
    
    # Process splits
    if args.split == 'all' or (args.split == 'train' and bool(args.store_val)):
        # Process both train and val
        if has_predefined_split:
            all_episodes_main(ROOT, 'train', info, episodes=all_episodes, train_ratio=args.train_ratio)
            if bool(args.store_val):
                all_episodes_main(ROOT, 'val', info, episodes=all_episodes, train_ratio=args.train_ratio)
        else:
            # Create consistent split once
            print(f"No split defined in info.json. Creating train/val split with ratio {args.train_ratio}")
            episode_splits = split_episodes(all_episodes, train_ratio=args.train_ratio)
            
            # Process train
            all_episodes_main(ROOT, 'train', info, episodes=episode_splits['train'], train_ratio=args.train_ratio)
            
            # Process val
            if bool(args.store_val):
                all_episodes_main(ROOT, 'val', info, episodes=episode_splits['val'], train_ratio=args.train_ratio)
    else:
        # Process specified split only
        if has_predefined_split:
            # Use predefined split
            all_episodes_main(ROOT, args.split, info, episodes=all_episodes, train_ratio=args.train_ratio)
        else:
            # Create split and use the requested one
            print(f"No split defined in info.json. Creating train/val split with ratio {args.train_ratio}")
            episode_splits = split_episodes(all_episodes, train_ratio=args.train_ratio)
            if args.split in episode_splits:
                all_episodes_main(ROOT, args.split, info, episodes=episode_splits[args.split], train_ratio=args.train_ratio)
            else:
                print(f"Warning: Split '{args.split}' not found. Using all episodes.")
                all_episodes_main(ROOT, args.split, info, episodes=all_episodes, train_ratio=args.train_ratio)
    
    # Store instructions
    if bool(args.store_instructions):
        os.makedirs('instructions/geniesim3', exist_ok=True)
        instr = store_instructions('train', info)
        with open('instructions/geniesim3/train_instructions.json', 'w') as fid:
            json.dump(instr, fid)
        instr = store_instructions('val', info)
        with open('instructions/geniesim3/val_instructions.json', 'w') as fid:
            json.dump(instr, fid)

