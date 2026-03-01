import numpy as np, open3d as o3d, matplotlib, torch, cv2, av
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datasets.geniesim3 import GenieSim3Dataset
from pathlib import Path
import json
import pyarrow.parquet as pq
import imageio.v3 as iio

class RobotSceneVisualizer:
    def __init__(self):
        self.world_pcds = []
        self.grippers = {'left': None, 'right': None}
        self.robot_pos = None
        self.robot_rot = None
        self.trajectory_grippers = {'left': [], 'right': []}  # List of {'pos', 'rot', 'color', 'alpha'}

    def _get_cam_proj_mat_inv(self, extrinsics, intrinsics):
        C = extrinsics[:3, 3:4]
        R = extrinsics[:3, :3]
        R_inv = R.T
        t_inv = -R_inv @ C
        ext_inv = np.concatenate([R_inv, t_inv], axis=-1)
        cam_proj = intrinsics @ ext_inv
        cam_proj_homo = np.vstack([
            cam_proj,
            np.array([[0, 0, 0, 1]], dtype=cam_proj.dtype)
        ])
        cam_proj_inv = np.linalg.inv(cam_proj_homo)[:3]
        return cam_proj_inv

    def _create_uniform_pixel_coords_image(self, resolution):
        h, w = resolution
        u = np.tile(np.arange(w), (h, 1))
        v = np.tile(np.arange(h), (w, 1)).T
        ones = np.ones_like(u)
        return np.stack((u, v, ones), axis=-1)

    def _unproject_depth(self, depth, extrinsics, intrinsics):
        h, w = depth.shape
        uv1 = self._create_uniform_pixel_coords_image((h, w))
        pc = uv1 * depth[:, :, None]
        pc = np.concatenate([pc, np.ones((h, w, 1))], axis=-1)
        pc = pc.reshape(-1, 4).T
        cam_proj_inv = self._get_cam_proj_mat_inv(extrinsics, intrinsics)
        world_pc = cam_proj_inv @ pc
        return world_pc.T.reshape(h, w, 3)

    def add_camera_data(self, rgb_image, depth_image, intrinsic_matrix, extrinsic_matrix, depth_scale=1000.0):
        if depth_scale != 1.0:
            depth_image = depth_image / depth_scale
        depth_image = depth_image.astype(np.float32)
        valid_mask = (depth_image > 0.01) & (depth_image < 10.0)
        T_world_to_cam = extrinsic_matrix.astype(np.float64)
        world_pc = self._unproject_depth(depth_image, T_world_to_cam, intrinsic_matrix.astype(np.float64))
        xyz_world = world_pc[valid_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_world.astype(np.float64))
        if rgb_image is not None:
            rgb_flat = rgb_image.reshape(-1, 3)
            rgb_flat = rgb_flat[valid_mask.flatten()]
            pcd.colors = o3d.utility.Vector3dVector(rgb_flat.astype(np.float64) / 255.0)
        self.world_pcds.append(pcd)

    def visualize(self, filename="debug_vis.png"):
        if not self.world_pcds:
            print("No point clouds to visualize.")
            return
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd = self.world_pcds[0]
        if len(combined_pcd.points) > 500000:
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)
        points = np.asarray(combined_pcd.points)
        colors = np.asarray(combined_pcd.colors)
        if len(points) == 0:
            print("Warning: No points to visualize.")
            return
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        idx = np.random.choice(len(points), min(len(points), 20000), replace=False)
        if len(colors) > 0:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c=colors[idx], s=1, alpha=0.6)
        else:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c='gray', s=1, alpha=0.6)
        # Draw trajectory grippers first (translucent)
        for hand_key in ['left', 'right']:
            for traj_data in self.trajectory_grippers[hand_key]:
                self._draw_gripper_wireframe(ax, traj_data['pos'], traj_data['rot'], 
                                            traj_data['color'], alpha=traj_data['alpha'])
        
        # Draw current grippers (opaque)
        for hand_key, data in self.grippers.items():
            if data is not None:
                color_to_use = 'red' if hand_key == 'left' else 'blue'
                self._draw_gripper_wireframe(ax, data['pos'], data['rot'], color_to_use, alpha=1.0)
        center_point = np.array([0.8, 0, 0.4])
        eye_point = np.array([1.5, 0, 0.8])
        view_dir = center_point - eye_point
        view_dir = view_dir / np.linalg.norm(view_dir)
        elev = np.arcsin(-view_dir[2]) * 180 / np.pi
        azim = np.arctan2(view_dir[1], view_dir[0]) * 180 / np.pi
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Point Cloud in World Frame")
        all_min = points.min(axis=0)
        all_max = points.max(axis=0)
        center = (all_min + all_max) / 2
        max_range = (all_max - all_min).max() / 2
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_box_aspect([1,1,1])
        plt.savefig(filename, dpi=100)
        print(f"Visualization saved to {filename}")

    def set_gripper_poses(self, left_pos=None, left_rot=None, right_pos=None, right_rot=None, robot_pos=None, robot_rot=None):
        if left_pos is not None and left_rot is not None:
            self.grippers['left'] = {'pos': left_pos, 'rot': left_rot, 'color': 'blue'} # Image uses blue for left, red for right in bottom plots
        if right_pos is not None and right_rot is not None:
            self.grippers['right'] = {'pos': right_pos, 'rot': right_rot, 'color': 'red'}
        if robot_pos is not None and robot_rot is not None:
            self.robot_pos = robot_pos
            self.robot_rot = robot_rot
    
    def add_trajectory_gripper(self, hand_key, pos, rot, color, alpha):
        """Add a trajectory gripper pose with color and transparency."""
        self.trajectory_grippers[hand_key].append({
            'pos': pos,
            'rot': rot,
            'color': color,
            'alpha': alpha
        })

    def _draw_gripper_wireframe(self, ax, pos, rot_matrix, color, alpha=1.0):
        local_points = np.array([ [0.0, 0.0, 0.0], # 0: Base Center 
                                 [-0.08, 0.0, 0.0], # 1: Wrist back 
                                 [0.02, 0.04, 0.0], # 2: Left Finger Base 
                                 [0.10, 0.04, 0.0], # 3: Left Finger Tip 
                                 [0.02, -0.04, 0.0],# 4: Right Finger Base 
                                 [0.10, -0.04, 0.0] # 5: Right Finger Tip 
                                 ]).T # 3xN 
        
        rot_matrix = rot_matrix
        world_points = rot_matrix @ local_points + pos.reshape(3, 1) 
        wp = world_points
        lines = [ [wp[:,1], wp[:,0]], # Wrist to base 
                 [wp[:,0], wp[:,2]], # Base to Left Finger Base 
                 [wp[:,2], wp[:,3]], # Left Finger to Tip 
                 [wp[:,0], wp[:,4]], # Base to Right Finger Base 
                 [wp[:,4], wp[:,5]], # Right Finger to Tip 
                 [wp[:,2], wp[:,4]] # Connecting finger bases (palm) 
        ]
        for line_start, line_end in lines: 
            ax.plot( [line_start[0], line_end[0]], 
                    [line_start[1], line_end[1]], 
                    [line_start[2], line_end[2]], color=color, linewidth=10, alpha=alpha
            )

    def _draw_base_axis(self, ax, pos, rot_matrix):
        axis_length = 0.3
        origin = pos.reshape(3)
        x_axis_end = origin + rot_matrix[:, 0] * axis_length
        y_axis_end = origin + rot_matrix[:, 1] * axis_length
        z_axis_end = origin + rot_matrix[:, 2] * axis_length
        ax.plot([origin[0], x_axis_end[0]],
                [origin[1], x_axis_end[1]],
                [origin[2], x_axis_end[2]],
                color='r', linewidth=3)
        ax.plot([origin[0], y_axis_end[0]],
                [origin[1], y_axis_end[1]],
                [origin[2], y_axis_end[2]],
                color='g', linewidth=3)
        ax.plot([origin[0], z_axis_end[0]],
                [origin[1], z_axis_end[1]],
                [origin[2], z_axis_end[2]],
                color='b', linewidth=3)
                
def load_video(video_path):
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames) if frames else None

def load_depth_video(video_path: Path, im_size=256):
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
            frames.append(depth_meters)
        container.close()
        if not frames:
            return None
        return np.array(frames)
    except Exception as e:
        print(f"Error reading depth video with PyAV: {e}")
        return None

def extract_ee_pose_from_state(state):
    left_pos = state[14:17]
    right_pos = state[17:20]
    left_quat_xyzw = state[32:36]
    right_quat_xyzw = state[36:40]
    left_pose = np.concatenate([left_pos, left_quat_xyzw])
    right_pose = np.concatenate([right_pos, right_quat_xyzw])
    robot_pos = state[118:121]
    robot_quat = state[121:125]
    robot_pose = np.concatenate([robot_pos, robot_quat])
    return {
        'left': left_pose,
        'right': right_pose,
        'robot': robot_pose
    }

def transform_pose_to_robot_frame(pose_pos, pose_rot, source_robot_pos, source_robot_rot, target_robot_pos, target_robot_rot):
    # Transform pose from source robot frame to world frame
    pose_in_world_pos = source_robot_rot @ pose_pos + source_robot_pos
    pose_in_world_rot = source_robot_rot @ pose_rot
    # Transform from world frame to target robot frame
    target_robot_rot_inv = target_robot_rot.T
    transformed_pos = target_robot_rot_inv @ (pose_in_world_pos - target_robot_pos)
    transformed_rot = target_robot_rot_inv @ pose_in_world_rot
    return transformed_pos, transformed_rot

def extract_camera_extrinsics(state):
    hand_left_rot = state[126:135].reshape(3, 3)
    hand_left_trans = state[171:174]
    hand_left_T = np.eye(4, dtype=np.float32)
    hand_left_T[:3, :3] = hand_left_rot
    hand_left_T[:3, 3] = hand_left_trans
    hand_right_rot = state[135:144].reshape(3, 3)
    hand_right_trans = state[174:177]
    hand_right_T = np.eye(4, dtype=np.float32)
    hand_right_T[:3, :3] = hand_right_rot
    hand_right_T[:3, 3] = hand_right_trans
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

def get_camera_intrinsics(info, camera_key, episode_idx=0):
    camera_params = info.get('camera_parameters', {})
    episode_key = str(episode_idx)
    if episode_key not in camera_params:
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
        fx = fy = 306
        cx = 320
        cy = 200
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

def load_geniesim3_raw_data(root_path, chunk_idx=0, episode_idx=0, timestep=0, im_size=256):
    root = Path(root_path)
    info_path = root / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    parquet_path = root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = pq.read_table(parquet_path).to_pandas()
    states = np.stack([np.array(s) for s in df['observation.state'].values])
    
    if timestep >= len(states):
        raise ValueError(f"Timestep {timestep} out of range. Episode has {len(states)} frames.")
    
    state = states[timestep]
    
    video_base = root / "videos" / f"chunk-{chunk_idx:03d}"
    head_rgb_path = video_base / "observation.images.top_head" / f"episode_{episode_idx:06d}.mp4"
    hand_left_rgb_path = video_base / "observation.images.hand_left" / f"episode_{episode_idx:06d}.mp4"
    hand_right_rgb_path = video_base / "observation.images.hand_right" / f"episode_{episode_idx:06d}.mp4"
    head_depth_path = video_base / "observation.images.head_depth" / f"episode_{episode_idx:06d}.mp4"
    hand_left_depth_path = video_base / "observation.images.hand_left_depth" / f"episode_{episode_idx:06d}.mp4"
    hand_right_depth_path = video_base / "observation.images.hand_right_depth" / f"episode_{episode_idx:06d}.mp4"
    head_rgb = load_video(head_rgb_path)
    hand_left_rgb = load_video(hand_left_rgb_path)
    hand_right_rgb = load_video(hand_right_rgb_path)
    head_depth = load_depth_video(head_depth_path, im_size=im_size)
    hand_left_depth = load_depth_video(hand_left_depth_path, im_size=im_size)
    hand_right_depth = load_depth_video(hand_right_depth_path, im_size=im_size)
    
    min_frames = min(
        len(head_rgb) if head_rgb is not None else len(states),
        len(hand_left_rgb) if hand_left_rgb is not None else len(states),
        len(hand_right_rgb) if hand_right_rgb is not None else len(states),
        len(states)
    )
    
    if timestep >= min_frames:
        timestep = min_frames - 1
    
    vis = RobotSceneVisualizer()
    
    extrinsics = extract_camera_extrinsics(state)
    intrinsics_head = get_camera_intrinsics(info, 'head_front', episode_idx=episode_idx)
    intrinsics_hand_left = get_camera_intrinsics(info, 'hand_left', episode_idx=episode_idx)
    intrinsics_hand_right = get_camera_intrinsics(info, 'hand_right', episode_idx=episode_idx)
    
    default_intrinsics = np.array([[im_size, 0, im_size/2], [0, im_size, im_size/2], [0, 0, 1]], dtype=np.float32)
    if intrinsics_head is None:
        intrinsics_head = default_intrinsics
    if intrinsics_hand_left is None:
        intrinsics_hand_left = default_intrinsics
    if intrinsics_hand_right is None:
        intrinsics_hand_right = default_intrinsics
    
    camera_data = [
        (head_rgb, head_depth, extrinsics['head_front'], intrinsics_head, 'head_front'),
        (hand_left_rgb, hand_left_depth, extrinsics['hand_left'], intrinsics_hand_left, 'hand_left'),
        (hand_right_rgb, hand_right_depth, extrinsics['hand_right'], intrinsics_hand_right, 'hand_right')
    ]
    
    for rgb_video, depth_video, extrinsic, intrinsic, cam_name in camera_data:
        if rgb_video is None or depth_video is None:
            print(f"Warning: Missing data for {cam_name}, skipping...")
            continue
        
        if timestep >= len(rgb_video) or timestep >= len(depth_video):
            print(f"Warning: Timestep {timestep} out of range for {cam_name}, skipping...")
            continue
        
        rgb_img = rgb_video[timestep]
        # if rgb_img.shape[1] != im_size or rgb_img.shape[2] != im_size:
        #     rgb_img = cv2.resize(rgb_img, (im_size, im_size))
        
        depth_img = depth_video[timestep].copy()
        if depth_img.dtype == np.float16:
            depth_img = depth_img.astype(np.float32)
        
        depth_valid = depth_img[depth_img > 0]
        if len(depth_valid) > 0:
            depth_max = np.max(depth_valid)
            depth_mean = np.mean(depth_valid)
            if depth_mean > 100 or depth_max > 1000:
                depth_scale = 1000.0
            else:
                depth_scale = 1.0
        else:
            depth_scale = 1.0
        
        vis.add_camera_data(
            rgb_image=rgb_img,
            depth_image=depth_img,
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            depth_scale=depth_scale
        )
    
    poses = extract_ee_pose_from_state(state)
    left_pos = poses['left'][:3]
    left_quat_xyzw = poses['left'][3:7]
    right_pos = poses['right'][:3]
    right_quat_xyzw = poses['right'][3:7]
    robot_pos = poses['robot'][:3]
    robot_quat_xyzw = poses['robot'][3:7]
    left_rot = R.from_quat(left_quat_xyzw).as_matrix()
    right_rot = R.from_quat(right_quat_xyzw).as_matrix()
    robot_rot = R.from_quat(robot_quat_xyzw).as_matrix()
    
    current_robot_pos = robot_pos.copy()
    current_robot_rot = robot_rot.copy()
    history_steps = 50
    future_steps = 50
    
    for hist_idx in range(1, history_steps + 1, 10):
        t_idx = timestep - hist_idx
        if t_idx >= 0 and t_idx < len(states):
            hist_state = states[t_idx]
            hist_poses = extract_ee_pose_from_state(hist_state)
            # Extract robot base frame at history timestep
            hist_robot_pos = hist_poses['robot'][:3]
            hist_robot_quat_xyzw = hist_poses['robot'][3:7]
            hist_robot_rot = R.from_quat(hist_robot_quat_xyzw).as_matrix()
            # Extract eef poses in their robot base frame
            hist_left_pos = hist_poses['left'][:3]
            hist_left_quat_xyzw = hist_poses['left'][3:7]
            hist_left_rot = R.from_quat(hist_left_quat_xyzw).as_matrix()
            
            hist_right_pos = hist_poses['right'][:3]
            hist_right_quat_xyzw = hist_poses['right'][3:7]
            hist_right_rot = R.from_quat(hist_right_quat_xyzw).as_matrix()
            # Transform to current robot base frame
            trans_left_pos, trans_left_rot = transform_pose_to_robot_frame(
                hist_left_pos, hist_left_rot,
                hist_robot_pos, hist_robot_rot,
                current_robot_pos, current_robot_rot
            )
            trans_right_pos, trans_right_rot = transform_pose_to_robot_frame(
                hist_right_pos, hist_right_rot,
                hist_robot_pos, hist_robot_rot,
                current_robot_pos, current_robot_rot
            )
            alpha_base = 1
            gradient_factor = (hist_idx - 1) / (history_steps - 1) if history_steps > 1 else 0
            left_pink_intensity = 0.5 + gradient_factor * 0.3  # 0.5 (darker) at t-1, 0.8 (lighter) at t-2
            right_green_intensity = 0.5 + gradient_factor * 0.3  # 0.5 (darker) at t-1, 0.8 (lighter) at t-2
            left_color = (1.0, 0.75 * left_pink_intensity, 0.85 * left_pink_intensity)  # RGB for pink
            right_color = (0.75 * right_green_intensity, 1.0, 0.75 * right_green_intensity)  # RGB for green
            vis.add_trajectory_gripper('left', trans_left_pos, trans_left_rot, left_color, alpha_base)
            vis.add_trajectory_gripper('right', trans_right_pos, trans_right_rot, right_color, alpha_base)
    
    for fut_idx in range(1, future_steps + 1, 10):
        t_idx = timestep + fut_idx
        if t_idx < len(states):
            fut_state = states[t_idx]
            fut_poses = extract_ee_pose_from_state(fut_state)
            # Extract robot base frame at future timestep
            fut_robot_pos = fut_poses['robot'][:3]
            fut_robot_quat_xyzw = fut_poses['robot'][3:7]
            fut_robot_rot = R.from_quat(fut_robot_quat_xyzw).as_matrix()
            # Extract eef poses in their robot base frame
            fut_left_pos = fut_poses['left'][:3]
            fut_left_quat_xyzw = fut_poses['left'][3:7]
            fut_left_rot = R.from_quat(fut_left_quat_xyzw).as_matrix()
            fut_right_pos = fut_poses['right'][:3]
            fut_right_quat_xyzw = fut_poses['right'][3:7]
            fut_right_rot = R.from_quat(fut_right_quat_xyzw).as_matrix()
            # Transform to current robot base frame
            trans_left_pos, trans_left_rot = transform_pose_to_robot_frame(
                fut_left_pos, fut_left_rot,
                fut_robot_pos, fut_robot_rot,
                current_robot_pos, current_robot_rot
            )
            trans_right_pos, trans_right_rot = transform_pose_to_robot_frame(
                fut_right_pos, fut_right_rot,
                fut_robot_pos, fut_robot_rot,
                current_robot_pos, current_robot_rot
            )
            alpha_base = 0.4
            gradient_factor = (fut_idx - 1) / (future_steps - 1) if future_steps > 1 else 0
            purple_intensity = 0.5 + gradient_factor * 0.3  # 0.5 at t+1, 0.8 at t+9
            left_color = (0.7 * purple_intensity, 0.5 * purple_intensity, 0.9 * purple_intensity)  # RGB for purple
            blue_intensity = 0.5 + gradient_factor * 0.3  # 0.5 at t+1, 0.8 at t+9
            right_color = (0.5 * blue_intensity, 0.7 * blue_intensity, 0.9 * blue_intensity)  # RGB for blue
            vis.add_trajectory_gripper('left', trans_left_pos, trans_left_rot, left_color, alpha_base)
            vis.add_trajectory_gripper('right', trans_right_pos, trans_right_rot, right_color, alpha_base)
    vis.set_gripper_poses(
        left_pos=left_pos,
        left_rot=left_rot,
        right_pos=right_pos,
        right_rot=right_rot,
        robot_pos=robot_pos,
        robot_rot=robot_rot
    )
    return vis

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None, help='Path to raw GenieSim3.0 data root (for raw mode)')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Chunk index to visualize')
    parser.add_argument('--episode_idx', type=int, default=0, help='Episode index to visualize (for raw mode)')
    parser.add_argument('--timestep', type=int, default=0, help='Timestep to visualize (for raw mode)')
    parser.add_argument('--output', type=str, default='visualization_output.png', help='Output filename')
    args = parser.parse_args()
    vis = load_geniesim3_raw_data(args.root_path, args.chunk_idx, args.episode_idx, args.timestep)
    vis.visualize(args.output)