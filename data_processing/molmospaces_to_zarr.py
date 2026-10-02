"""Convert MolmoSpaces (MuJoCo, Franka) demos with recorded depth to the LIBERO zarr layout.

Train GeomVLA on the result with ``--dataset Libero --relative_action true --rotation_format axisangle``
(see scripts/molmospaces/). Everything is expressed in the robot base frame.

Input: a datagen run dir (house_*/trajectories_batch_*.h5 plus episode_*_<camera>[_depth]_batch_*.mp4),
recorded with depth (e.g. the *OneHouseDepthConfig datagens, MLSPACES_DEPTH_MAX=4.0, lossless).

Output zarr (N rows = one per frame; arrays chunked by --chunk rows, like rechunk.py's output):
    rgb          (N, 2, 3, 224, 224) uint8    cameras: [exterior, wrist] (exterior first: the motion
                                              expert only queries camera 0)
    depth        (N, 2, 224, 224)    float32  metric z-depth (m), nearest-neighbour resized
    extrinsics   (N, 2, 4, 4)        float32  camera-to-robot-base, OpenCV axes (+z forward, y down)
    intrinsics   (N, 2, 3, 3)        float32  for the 224x224 images
    proprioception (N, 1, 1, 8)      float32  TCP xyz + quat_xyzw + gripper width (0.08 if the last
                                              command was open, else 0; LIBERO's threshold is 0.02)
    rel_action   (N, T, 1, 7)        float32  T per-step targets: delta xyz (cumulative from the
                                              current TCP) + delta axis-angle (left-multiplied:
                                              R_k = dR_k @ R_{k-1}) + gripper (+1 close, -1 open)
    action       (N, 1)              float32  dummy, only so data_processing/rechunk.py finds a length
    instr_id     (N,)                int32    index into the instructions json written next to it

The targets are the planner's commanded EEF poses (actions/ee_pose), action[t] pairing with state[t]
after dropping the padded first action, as in molmospaces' mlspaces_to_lerobot.py. Steps past the end
of a trajectory repeat its last target (zero delta).

Usage (MLSPACES_DEPTH_MAX must match the recording):
    MLSPACES_DEPTH_MAX=4.0 PYTHONPATH=/home/haotian/molmospaces python data_processing/molmospaces_to_zarr.py \
        <run_dir> <out.zarr> --exterior random_view_00 --wrist wrist_camera
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

from molmo_spaces.utils.depth_utils import load_depth_video

IMG = 224
GRIPPER_ACTION_SCALE = 255.0  # joint_pos gripper command range
OPEN_WIDTH = 0.08  # metres; LIBERO's proprio threshold is 0.02


def decode_json(row) -> dict:
    raw = row.tobytes() if isinstance(row, np.ndarray) else row
    return json.loads(raw.decode("utf-8").rstrip("\x00"))


def pose7_to_mat(p: np.ndarray) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] -> 4x4."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(p[3:7], scalar_first=True).as_matrix()
    m[:3, 3] = p[:3]
    return m


def video(h5: Path, traj: str, cam: str) -> Path:
    suffix = h5.stem.replace("trajectories", "")  # _batch_k_of_n
    return h5.parent / f"episode_{int(traj[5:]):08d}_{cam}{suffix}.mp4"


def resize_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (IMG, IMG), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)


def resize_depth(d: np.ndarray) -> np.ndarray:
    return cv2.resize(d, (IMG, IMG), interpolation=cv2.INTER_NEAREST)


def scaled_intrinsics(k: np.ndarray, w: int, h: int) -> np.ndarray:
    s = np.diag([IMG / w, IMG / h, 1.0])
    return (s @ k).astype(np.float32)


def convert_trajectory(h5: Path, traj: str, ext_cam: str, wrist_cam: str, horizon: int):
    with h5py.File(h5, "r") as f:
        tg = f[traj]
        n = int(tg["success"].shape[0]) - 2  # drop dummy first action / done sentinel (as LeRobot converter)
        if n < 1:
            return None
        task = decode_json(tg["obs_scene"][()])["task_description"]
        base = pose7_to_mat(tg["obs/extra/robot_base_pose"][0])
        base_inv = np.linalg.inv(base)
        tcp = tg["obs/extra/tcp_pose"][:n]  # robot base frame, [xyz, qwxyz]
        targets = [np.asarray(decode_json(tg["actions/ee_pose"][i + 1])["arm"], dtype=float) for i in range(n)]
        grip_cmd = [decode_json(tg["actions/joint_pos"][i + 1])["gripper"][0] / GRIPPER_ACTION_SCALE for i in range(n)]
        cams = {}
        for c in (ext_cam, wrist_cam):
            p = tg[f"obs/sensor_param/{c}"]
            cams[c] = (np.asarray(p["cam2world_gl"][:n]), np.asarray(p["intrinsic_cv"][:n]))

    rgb = {c: iio.imread(video(h5, traj, c), plugin="pyav")[:n] for c in (ext_cam, wrist_cam)}
    depth = {c: load_depth_video(video(h5, traj, f"{c}_depth"))[:n] for c in (ext_cam, wrist_cam)}
    h, w = rgb[ext_cam].shape[1:3]
    for c in (ext_cam, wrist_cam):
        if len(rgb[c]) < n or len(depth[c]) < n:
            raise RuntimeError(f"{h5.name}/{traj} {c}: {len(rgb[c])} rgb / {len(depth[c])} depth frames, need {n}")

    tgt_mats = [pose7_to_mat(t) for t in targets]
    tcp_mats = [pose7_to_mat(t) for t in tcp]
    out = {k: [] for k in ("rgb", "depth", "extrinsics", "intrinsics", "proprioception", "rel_action")}
    for t in range(n):
        out["rgb"].append(np.stack([resize_rgb(rgb[c][t]) for c in (ext_cam, wrist_cam)]))
        out["depth"].append(np.stack([resize_depth(depth[c][t]) for c in (ext_cam, wrist_cam)]))
        out["extrinsics"].append(np.stack([(base_inv @ cams[c][0][t]).astype(np.float32) for c in (ext_cam, wrist_cam)]))
        out["intrinsics"].append(np.stack([scaled_intrinsics(cams[c][1][t], w, h) for c in (ext_cam, wrist_cam)]))
        # Openness from the last gripper command, not the finger width: a held object keeps the
        # fingers apart, so the width would read "open" while grasping.
        closed = 1.0 if t > 0 and grip_cmd[t - 1] > 0.5 else 0.0
        q_xyzw = Rotation.from_matrix(tcp_mats[t][:3, :3]).as_quat()
        out["proprioception"].append(
            np.concatenate([tcp_mats[t][:3, 3], q_xyzw, [(1.0 - closed) * OPEN_WIDTH]]).astype(np.float32)[None, None]
        )
        prev = tcp_mats[t]
        chunk = []
        for k in range(horizon):
            i = min(t + k, n - 1)
            cur = tgt_mats[i]
            d_pos = cur[:3, 3] - prev[:3, 3]
            d_rot = Rotation.from_matrix(cur[:3, :3] @ prev[:3, :3].T).as_rotvec()
            g = 1.0 if grip_cmd[i] > 0.5 else -1.0
            chunk.append(np.concatenate([d_pos, d_rot, [g]]))
            prev = cur
        out["rel_action"].append(np.asarray(chunk, dtype=np.float32)[:, None, :])
    return task, {k: np.stack(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("out", type=Path)
    ap.add_argument("--exterior", default="random_view_00")
    ap.add_argument("--wrist", default="wrist_camera")
    ap.add_argument("--horizon", type=int, default=10, help="action chunk length (LIBERO eval uses 10)")
    ap.add_argument("--chunk", type=int, default=8, help="zarr rows per chunk (= training --chunk_size)")
    ap.add_argument("--episodes", type=str, default=None, help="e.g. 0:18 -- slice of the trajectory list")
    args = ap.parse_args()
    if "MLSPACES_DEPTH_MAX" not in os.environ:
        raise SystemExit("Set MLSPACES_DEPTH_MAX to the depth range the data was recorded with")

    trajs = []
    for d in args.run_dirs:
        for h5 in sorted(Path(p) for p in glob.glob(str(d / "house_*" / "trajectories_batch_*.h5"))):
            with h5py.File(h5, "r") as f:
                trajs += [(h5, k) for k in sorted((k for k in f if k.startswith("traj_")), key=lambda k: int(k[5:]))]
    if args.episodes:
        a, b = (int(x) if x else None for x in args.episodes.split(":"))
        trajs = trajs[a:b]
    print(f"{len(trajs)} trajectories -> {args.out}")

    root = zarr.open_group(str(args.out), mode="w")
    comp = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arrays, instructions = {}, []
    for h5, traj in trajs:
        res = convert_trajectory(h5, traj, args.exterior, args.wrist, args.horizon)
        if res is None:
            continue
        task, data = res
        if task not in instructions:
            instructions.append(task)
        n = len(data["rgb"])
        data["action"] = np.zeros((n, 1), np.float32)
        data["instr_id"] = np.full((n,), instructions.index(task), np.int32)
        for k, v in data.items():
            if k not in arrays:
                arrays[k] = root.zeros(k, shape=(0, *v.shape[1:]), chunks=(args.chunk, *v.shape[1:]), dtype=v.dtype, compressor=comp)
            arrays[k].append(v)
        print(f"  {h5.parent.name}/{h5.name}/{traj}: {n} frames, max |dxyz| {np.abs(data['rel_action'][..., :3]).max():.3f} m, "
              f"max |drot| {np.abs(data['rel_action'][..., 3:6]).max():.3f} rad", flush=True)
    json.dump(instructions, open(str(args.out).rstrip("/") + ".instructions.json", "w"), indent=1)
    print(f"done: {len(arrays['rgb'])} frames, instructions {instructions}")


if __name__ == "__main__":
    main()
