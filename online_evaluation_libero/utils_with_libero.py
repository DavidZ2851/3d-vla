import json, os, numpy as np, torch
from typing import Dict, Sequence, Tuple
# TODO: do "cd /data/user_data/ziyinx/LIBERO && pip install -e ." in 3dfavla conda env before evaluation
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv

CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
DEPTH_KEYS = ("agentview_depth", "robot0_eye_in_hand_depth")
flip_yz = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float32)

def build_benchmark(benchmark_name: str, task_order_index: int):
    benchmark = get_benchmark(benchmark_name)(task_order_index)
    return benchmark

def build_task_bddl_path(task) -> str:
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

def build_task_init_states_path(task) -> str:
    return os.path.join(
        get_libero_path("init_states"), task.problem_folder, task.init_states_file
    )

def load_init_states(task) -> torch.Tensor:
    init_states_path = build_task_init_states_path(task)
    return torch.load(init_states_path)

def create_vectorized_env(
    task,
    img_h: int,
    img_w: int,
    num_envs: int,
    controller: str = "OSC_POSE",
    controller_config_updates: Dict | None = None,
):
    env_args = {
        "bddl_file_name": build_task_bddl_path(task),
        "camera_heights": img_h,
        "camera_widths": img_w,
        "camera_names": list(CAMERA_NAMES),
        "camera_depths": True,
        "controller": controller,
        "controller_config_updates": controller_config_updates or {},
    }
    return DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)])

def depth_to_meters(sim, depth_buffer: np.ndarray) -> np.ndarray:
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    z = depth_buffer
    depth = (2.0 * near * far) / (far + near - (2.0 * z - 1.0) * (far - near))
    return depth.astype(np.float32)

def get_camera_intrinsics(sim, camera_name: str, height: int, width: int) -> np.ndarray:
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = np.deg2rad(sim.model.cam_fovy[cam_id])
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def get_camera_extrinsics_c2w(sim, camera_name: str) -> np.ndarray:
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cam_id].copy()
    cam_rot = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = cam_rot
    ext[:3, 3] = cam_pos
    ext = np.matmul(ext.astype(np.float32), flip_yz)
    return ext

def get_static_camera_matrices(env, img_h: int, img_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return cached static matrices:
      - intrinsics for both cameras: (B, 2, 3, 3)
      - extrinsics for agentview only: (B, 4, 4)
    """
    intrinsics, agent_extrinsics = [], []
    for worker in env.workers:
        sim = worker.env.sim
        int_i = []
        for camera_name in CAMERA_NAMES:
            int_i.append(get_camera_intrinsics(sim, camera_name, img_h, img_w))
        intrinsics.append(np.stack(int_i, axis=0))
        agent_extrinsics.append(get_camera_extrinsics_c2w(sim, "agentview"))
    return np.stack(intrinsics, axis=0), np.stack(agent_extrinsics, axis=0)

def get_wrist_extrinsics(env) -> np.ndarray:
    """Return wrist camera extrinsics for all envs as (B, 4, 4)."""
    wrist_extrinsics = []
    for worker in env.workers:
        sim = worker.env.sim
        wrist_extrinsics.append(get_camera_extrinsics_c2w(sim, "robot0_eye_in_hand"))
    return np.stack(wrist_extrinsics, axis=0)

def compose_extrinsics(
    agent_extrinsics: np.ndarray, wrist_extrinsics: np.ndarray
) -> np.ndarray:
    """Compose full camera extrinsics batch as (B, 2, 4, 4)."""
    return np.stack([agent_extrinsics, wrist_extrinsics], axis=1)

def extract_rgb_depth(obs_list: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    rgb_batch = []
    depth_batch = []
    for obs in obs_list:
        missing_rgb = [key for key in RGB_KEYS if key not in obs]
        missing_depth = [key for key in DEPTH_KEYS if key not in obs]
        if missing_rgb or missing_depth:
            raise KeyError(
                f"Missing LIBERO RGBD keys. rgb={missing_rgb}, depth={missing_depth}"
            )
        rgbs = np.stack([obs[k].transpose(2, 0, 1) for k in RGB_KEYS], axis=0)  # (ncam, 3, h, w)
        rgbs = np.ascontiguousarray(rgbs[:, :, ::-1, :])  # flip upside down
        depth_frames = []
        for key in DEPTH_KEYS:
            depth = np.asarray(obs[key])
            # Some LIBERO / robosuite setups expose depth as (H, W, 1).
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            depth_frames.append(depth)
        depths = np.stack(depth_frames, axis=0)  # (ncam, h, w)
        depths = np.ascontiguousarray(depths[:, ::-1, :])  # flip upside down
        if rgbs.shape[-2:] != (224, 224) or depths.shape[-2:] != (224, 224):
            raise ValueError(
                "LIBERO online evaluation expects 224x224 RGBD for agentview and wrist, "
                f"got rgb={rgbs.shape[-2:]}, depth={depths.shape[-2:]}."
            )
        rgb_batch.append(rgbs)
        depth_batch.append(depths)
    return np.stack(rgb_batch, axis=0), np.stack(depth_batch, axis=0)

def convert_depth_batch_to_meters(env, depth_batch: np.ndarray) -> np.ndarray:
    meter_depth = []
    for env_idx, worker in enumerate(env.workers):
        sim = worker.env.sim
        cams = []
        for cam_idx in range(depth_batch.shape[1]):
            cams.append(depth_to_meters(sim, depth_batch[env_idx, cam_idx]))
        meter_depth.append(np.stack(cams, axis=0))
    return np.stack(meter_depth, axis=0)

def save_results(output_file: str, payload: Dict):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
