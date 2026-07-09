#!/usr/bin/env python3
"""SpaTracker2 + TAPIP3D-style spaghetti visualization for raw RGBD datasets.

Benchmarks:
  calvin:    CALVIN task_ABC_D/training language spans
  libero:    LIBERO-10 raw HDF5 demos
  robotwin2: RoboTwin2 pick_dual_bottles HDF5 episodes

For each selected rollout/demo/episode, the script:
  1. saves the corresponding raw RGB video,
  2. picks start / mid / end query windows,
  3. tracks a grid for 15 frames at raw stride 3 with SpaTracker2,
  4. saves each window in TAPIP3D-compatible NPZ format,
  5. renders one white-background 3D spaghetti image per start/mid/end window.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = Path("/data/user_data/ziyinx/vis_outputs/raw_spatracker_tapip3d_first_windows")
DEFAULT_SPATRACKER_ROOT = Path("/data/user_data/ziyinx/SpaTrackerV2")
DEFAULT_SPATRACKER_CHECKPOINT = DEFAULT_SPATRACKER_ROOT / "checkpoints" / "SpatialTrackerV2-Offline"

DEFAULT_CALVIN_ROOT = Path("/data/group_data/katefgroup/VLA/calvin_dataset/task_ABC_D/training")
DEFAULT_LIBERO_ROOT = Path("/data/group_data/katefgroup/datasets/ziyinx/libero/libero_10")
DEFAULT_ROBOTWIN2_ROOT = Path("/data/group_data/katefgroup/datasets/ziyinx/RoboTwin2/pick_dual_bottles")

CALVIN_CAMERA_FRAME_FIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
CALVIN_FRONT_CAMERA_VIEW_MATRIX = np.asarray(
    [
        0.7232445478439331,
        -0.030260592699050903,
        0.6899287700653076,
        0.0,
        0.5141233801841736,
        0.6906105875968933,
        -0.5086592435836792,
        0.0,
        -0.46107974648475647,
        0.7225935459136963,
        0.5150379538536072,
        0.0,
        0.21526622772216797,
        -0.26317155361175537,
        -4.399168968200684,
        1.0,
    ],
    dtype=np.float32,
).reshape((4, 4)).T
CALVIN_FRONT_CAMERA_C2W_OPENGL = np.linalg.inv(CALVIN_FRONT_CAMERA_VIEW_MATRIX).astype(np.float32)
CALVIN_FRONT_CAMERA_C2W = (CALVIN_FRONT_CAMERA_C2W_OPENGL @ CALVIN_CAMERA_FRAME_FIX).astype(np.float32)
CALVIN_FRONT_CAMERA_W2C = np.linalg.inv(CALVIN_FRONT_CAMERA_C2W).astype(np.float32)


@dataclass
class RawEpisode:
    benchmark: str
    rollout_name: str
    source: str
    instruction: str
    rgb: np.ndarray  # (T,H,W,3) uint8, RGB
    depth: np.ndarray  # (T,H,W) float32, meters
    intrinsics: np.ndarray  # (T,3,3) float32
    extrinsics_w2c: np.ndarray  # (T,4,4) float32
    raw_frame_indices: np.ndarray  # (T,) int32


@dataclass
class TrackWindow:
    window_name: str
    query_index: int
    raw_frame_indices: np.ndarray
    query_uv: np.ndarray
    coords_world: np.ndarray
    coords_world_raw: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray


class SpaTrackerOfflineRunner:
    def __init__(
        self,
        spatracker_root: Path,
        checkpoint: Path | str,
        device: str,
        track_num: int,
        iters_track: int,
    ):
        self.device = device
        self.iters_track = int(iters_track)
        if not str(device).startswith("cuda"):
            raise RuntimeError("SpaTracker2 requires CUDA for this pipeline.")
        if str(spatracker_root) not in sys.path:
            sys.path.insert(0, str(spatracker_root))

        try:
            import easydict  # pylint: disable=import-error
            import yaml
            from models.SpaTrackV2.models.predictor import Predictor  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                f"Could not import SpaTracker2 from {spatracker_root}. "
                "Use the SpaTrackerV2 environment or install its missing dependencies."
            ) from exc

        cfg_path = spatracker_root / "config" / "magic_infer_offline.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        cfg.model.track_num = max(int(track_num), 1)
        model_ref = str(checkpoint)
        try:
            self.model = Predictor.from_pretrained(model_ref, args=cfg["model"])
        except TypeError:
            self.model = Predictor.from_pretrained(model_ref, model_cfg=cfg["model"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def track(
        self,
        video_rgb: np.ndarray,
        video_depth: np.ndarray,
        video_intrinsics: np.ndarray,
        video_extrinsics_w2c: np.ndarray,
        query_uv: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        video_t = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        depth_t = torch.from_numpy(video_depth.astype(np.float32, copy=False)).to(self.device, dtype=torch.float32)
        intr_t = torch.from_numpy(video_intrinsics.astype(np.float32, copy=False)).to(self.device, dtype=torch.float32)
        c2w_t = torch.from_numpy(np.linalg.inv(video_extrinsics_w2c).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )

        queries = np.zeros((query_uv.shape[0], 3), dtype=np.float32)
        queries[:, 1:] = query_uv.astype(np.float32, copy=False)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            (
                _c2w_traj,
                _intrs_out,
                _point_map,
                _conf_depth,
                track3d_pred,
                track2d_pred,
                vis_pred,
                conf_pred,
                _video_out,
            ) = self.model.forward(
                video_t,
                depth=depth_t,
                intrs=intr_t,
                extrs=c2w_t,
                queries=queries,
                fps=1,
                full_point=False,
                iters_track=self.iters_track,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                support_frame=max(int(video_rgb.shape[0]) - 1, 0),
                replace_ratio=0.2,
            )

        visibility = vis_pred.float().cpu().numpy().astype(np.float32)
        confidence = conf_pred.float().cpu().numpy().astype(np.float32)
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        if confidence.ndim == 3 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        return {
            "track_cam_xyz": track3d_pred[..., :3].float().cpu().numpy().astype(np.float32),
            "track_uvd": track2d_pred[..., :3].float().cpu().numpy().astype(np.float32),
            "visibility": visibility,
            "confidence": confidence,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", default=["calvin", "libero", "robotwin2"],
                   choices=["calvin", "libero", "robotwin2"])
    p.add_argument("--calvin-root", type=Path, default=DEFAULT_CALVIN_ROOT)
    p.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    p.add_argument("--robotwin2-root", type=Path, default=DEFAULT_ROBOTWIN2_ROOT)
    p.add_argument("--robotwin2-variant", type=str, default="aloha_vla_clean50_tcp")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-rollouts", type=int, default=1)
    p.add_argument("--track-len", type=int, default=15)
    p.add_argument("--raw-stride", type=int, default=3, help="Track every Nth raw frame; 3 means one-third FPS.")
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--max-tracks-per-window", type=int, default=140)
    p.add_argument("--motion-threshold", type=float, default=0.01)
    p.add_argument("--visibility-threshold", type=float, default=0.5)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--smooth-passes", type=int, default=1)
    p.add_argument("--scene-width", type=int, default=256, help="TAPIP3D scene cloud width.")
    p.add_argument("--scene-height", type=int, default=192, help="TAPIP3D scene cloud height.")
    p.add_argument("--pcd-max-points", type=int, default=60000)
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--point-alpha", type=float, default=0.44)
    p.add_argument("--line-width", type=float, default=1.35)
    p.add_argument("--line-alpha", type=float, default=0.9)
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-62.0)
    p.add_argument("--zoom", type=float, default=1.45)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--raw-video-fps", type=float, default=15.0)
    p.add_argument("--spatracker-root", type=Path, default=DEFAULT_SPATRACKER_ROOT)
    p.add_argument("--spatracker-checkpoint", type=Path, default=DEFAULT_SPATRACKER_CHECKPOINT)
    p.add_argument("--spatracker-track-num", type=int, default=756)
    p.add_argument("--spatracker-iters-track", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Load episodes and planned windows, then exit.")
    return p.parse_args()


def natural_int_key(path: Path) -> Tuple[str, int]:
    stem = path.stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    return stem, int(digits[-1]) if digits else -1


def calvin_front_intrinsics() -> np.ndarray:
    focal = float(200.0 / (2.0 * np.tan(np.deg2rad(10.0) / 2.0)))
    return np.array([[focal, 0.0, 100.0], [0.0, focal, 100.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def build_grid_points(height: int, width: int, grid_size: int) -> np.ndarray:
    ys = np.linspace(0.0, float(height - 1), int(grid_size), endpoint=False, dtype=np.float32)
    xs = np.linspace(0.0, float(width - 1), int(grid_size), endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)


def transform_camera_xyz_to_world(cam_xyz: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    flat = cam_xyz.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
    out = (np.linalg.inv(extrinsic_w2c).astype(flat.dtype) @ homog.T).T[:, :3]
    return out.reshape(cam_xyz.shape)


def transform_world_xyz_to_camera(world_xyz: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    flat = world_xyz.reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)], axis=1)
    out = (extrinsic_w2c.astype(flat.dtype) @ homog.T).T[:, :3]
    return out.reshape(world_xyz.shape)


def unproject_depth_to_world(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    src_h, src_w = depth.shape
    rgb_rs = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    depth_rs = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    k = intrinsics.astype(np.float32).copy()
    k[0, :] *= float(width) / float(src_w)
    k[1, :] *= float(height) / float(src_h)

    yy, xx = np.mgrid[0:height, 0:width]
    z = depth_rs.reshape(-1).astype(np.float32)
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    u = xx.reshape(-1).astype(np.float32)[valid]
    v = yy.reshape(-1).astype(np.float32)[valid]
    z = z[valid]
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    cam = np.stack([x, y, z], axis=1)
    world = transform_camera_xyz_to_world(cam, extrinsics_w2c).astype(np.float32)
    colors = rgb_rs.reshape(-1, 3)[valid].astype(np.float32) / 255.0
    return world, colors


def _infer_libero_depth_scale(depth_img: np.ndarray) -> float:
    valid = depth_img[np.isfinite(depth_img) & (depth_img > 0)]
    if valid.size == 0:
        return 1.0
    if float(valid.mean()) > 100.0 or float(valid.max()) > 1000.0:
        return 1000.0
    return 1.0


def load_libero_episode(hdf5_path: Path, demo_key: str, camera_name: str) -> RawEpisode:
    import h5py

    ext_key = "robot0_eye_in_hand" if camera_name == "eye_in_hand" else "agentview"
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][demo_key]
        rgb = np.asarray(grp["obs"][f"{camera_name}_rgb"], dtype=np.uint8)
        depth = np.asarray(grp["obs"][f"{camera_name}_depth"], dtype=np.float32)
        intr_raw = np.asarray(grp["intrinsics"][ext_key], dtype=np.float32)
        ext_raw = np.asarray(grp["extrinsics"][ext_key], dtype=np.float32)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    t = min(rgb.shape[0], depth.shape[0])
    rgb = np.ascontiguousarray(rgb[:t, ::-1, :, :])
    depth = np.ascontiguousarray(depth[:t, ::-1, :])
    for i in range(t):
        scale = _infer_libero_depth_scale(depth[i])
        if scale != 1.0:
            depth[i] /= scale
    intr = np.tile(intr_raw[None], (t, 1, 1)) if intr_raw.ndim == 2 else intr_raw[:t]
    ext_c2w = np.tile(ext_raw[None], (t, 1, 1)) if ext_raw.ndim == 2 else ext_raw[:t]
    flip_yz = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32))
    ext_c2w = np.matmul(ext_c2w.astype(np.float32), flip_yz)
    task_name = hdf5_path.stem.replace("_demo", "")
    return RawEpisode(
        benchmark="libero",
        rollout_name=f"{task_name}_{demo_key}",
        source=f"{hdf5_path}:{demo_key}",
        instruction=task_name.replace("_", " "),
        rgb=rgb,
        depth=depth.astype(np.float32),
        intrinsics=intr.astype(np.float32),
        extrinsics_w2c=np.linalg.inv(ext_c2w).astype(np.float32),
        raw_frame_indices=np.arange(t, dtype=np.int32),
    )


def discover_libero(root: Path, num_rollouts: int, camera_name: str) -> List[RawEpisode]:
    episodes: List[RawEpisode] = []
    for h5_path in sorted(root.glob("*.hdf5")):
        import h5py

        with h5py.File(h5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        if not demo_keys:
            continue
        episodes.append(load_libero_episode(h5_path, demo_keys[0], camera_name))
        if len(episodes) >= num_rollouts:
            break
    return episodes


def decode_robotwin2_rgb_frame(frame_bytes: bytes | np.bytes_) -> np.ndarray:
    raw = bytes(frame_bytes)
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode RoboTwin2 RGB bytes.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_robotwin2_instruction(variant_root: Path, episode_index: int) -> str:
    path = variant_root / "segment_annotations" / f"episode{episode_index}.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        segments = obj.get("segments", [])
        if segments:
            return " | ".join(str(seg.get("text", "")).strip() for seg in segments if seg.get("text"))
    path = variant_root / "instructions" / f"episode{episode_index}.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        seen = obj.get("seen", [])
        if seen:
            return str(seen[0])
    return "pick dual bottles"


def load_robotwin2_episode(hdf5_path: Path, variant_root: Path, camera_name: str) -> RawEpisode:
    import h5py

    episode_index = int("".join(ch if ch.isdigit() else " " for ch in hdf5_path.stem).split()[-1])
    with h5py.File(hdf5_path, "r") as f:
        prefix = f"observation/{camera_name}"
        rgb_ds = f[f"{prefix}/rgb"]
        depth = np.asarray(f[f"{prefix}/depth"], dtype=np.float32)
        intr = np.asarray(f[f"{prefix}/intrinsic_cv"], dtype=np.float32)
        extr34 = np.asarray(f[f"{prefix}/extrinsic_cv"], dtype=np.float32)
        rgb = np.asarray([decode_robotwin2_rgb_frame(rgb_ds[i]) for i in range(rgb_ds.shape[0])], dtype=np.uint8)
    t = min(rgb.shape[0], depth.shape[0], intr.shape[0], extr34.shape[0])
    extr = np.tile(np.eye(4, dtype=np.float32)[None], (t, 1, 1))
    extr[:, :3, :] = extr34[:t]
    return RawEpisode(
        benchmark="robotwin2",
        rollout_name=f"episode{episode_index:03d}",
        source=str(hdf5_path),
        instruction=load_robotwin2_instruction(variant_root, episode_index),
        rgb=np.ascontiguousarray(rgb[:t]),
        depth=np.ascontiguousarray(depth[:t].astype(np.float32)),
        intrinsics=np.ascontiguousarray(intr[:t].astype(np.float32)),
        extrinsics_w2c=extr,
        raw_frame_indices=np.arange(t, dtype=np.int32),
    )


def discover_robotwin2(root: Path, variant: str, num_rollouts: int, camera_name: str) -> List[RawEpisode]:
    variant_root = root / variant
    data_dir = variant_root / "data"
    files = sorted(data_dir.glob("episode*.hdf5"), key=natural_int_key)
    return [load_robotwin2_episode(p, variant_root, camera_name) for p in files[:num_rollouts]]


def build_calvin_rows(root: Path, min_len: int, num_rollouts: int) -> List[Dict]:
    ann = np.load(root / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True).item()
    bounds = np.asarray(np.load(root / "ep_start_end_ids.npy", allow_pickle=True), dtype=np.int64)
    rollout_starts = [int(x[0]) for x in bounds]
    rows: List[Dict] = []
    for ann_id, (instruction, span) in enumerate(zip(ann["language"]["ann"], ann["info"]["indx"])):
        start, end = int(span[0]), int(span[1])
        length = end - start + 1
        if length < min_len:
            continue
        rollout_idx = bisect.bisect_right(rollout_starts, start) - 1
        if rollout_idx < 0 or end > int(bounds[rollout_idx, 1]):
            continue
        rows.append(
            {
                "ann_id": int(ann_id),
                "instruction": str(instruction).strip(),
                "global_start": start,
                "global_end": end,
                "rollout_index": int(rollout_idx),
                "local_start": start - int(bounds[rollout_idx, 0]),
                "length": length,
            }
        )
    rows.sort(key=lambda r: (r["rollout_index"], r["local_start"], r["ann_id"]))
    return rows[:num_rollouts]


def load_calvin_episode(root: Path, row: Dict) -> RawEpisode:
    rgbs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    raw_ids = np.arange(int(row["global_start"]), int(row["global_end"]) + 1, dtype=np.int32)
    for frame_id in raw_ids.tolist():
        with np.load(root / f"episode_{frame_id:07d}.npz") as data:
            rgbs.append(np.asarray(data["rgb_static"], dtype=np.uint8))
            depths.append(np.asarray(data["depth_static"], dtype=np.float32))
    rgb = np.asarray(rgbs, dtype=np.uint8)
    depth = np.asarray(depths, dtype=np.float32)
    t = rgb.shape[0]
    return RawEpisode(
        benchmark="calvin",
        rollout_name=f"rollout{row['rollout_index']:03d}_ann{row['ann_id']:05d}",
        source=f"{root}:frames {int(row['global_start'])}-{int(row['global_end'])}",
        instruction=str(row["instruction"]),
        rgb=rgb,
        depth=depth,
        intrinsics=np.tile(calvin_front_intrinsics()[None], (t, 1, 1)).astype(np.float32),
        extrinsics_w2c=np.tile(CALVIN_FRONT_CAMERA_W2C[None], (t, 1, 1)).astype(np.float32),
        raw_frame_indices=raw_ids,
    )


def discover_calvin(root: Path, num_rollouts: int, min_len: int) -> List[RawEpisode]:
    return [load_calvin_episode(root, row) for row in build_calvin_rows(root, min_len, num_rollouts)]


def window_start_indices(total_frames: int, track_len: int, raw_stride: int) -> List[int]:
    span = (int(track_len) - 1) * int(raw_stride) + 1
    max_start = int(total_frames) - span
    if max_start < 0:
        return []
    return [int(x) for x in np.linspace(0, max_start, 3).round().astype(np.int64).tolist()]


def smooth_tracks(coords: np.ndarray, passes: int) -> np.ndarray:
    out = coords.astype(np.float32, copy=True)
    for _ in range(max(0, int(passes))):
        prev = np.concatenate([out[:1], out[:-1]], axis=0)
        nxt = np.concatenate([out[1:], out[-1:]], axis=0)
        stacked = np.stack([prev, out, nxt], axis=0)
        weights = np.array([0.25, 0.5, 0.25], dtype=np.float32)[:, None, None, None]
        finite = np.isfinite(stacked).all(axis=-1, keepdims=True)
        denom = np.sum(weights * finite, axis=0)
        numer = np.nansum(np.where(finite, stacked, 0.0) * weights, axis=0)
        out = np.where(denom > 0, numer / np.maximum(denom, 1e-6), np.nan).astype(np.float32)
    return out


def run_track_window(
    runner: SpaTrackerOfflineRunner,
    episode: RawEpisode,
    query_index: int,
    args: argparse.Namespace,
) -> TrackWindow:
    local_idx = query_index + np.arange(args.track_len, dtype=np.int32) * int(args.raw_stride)
    rgb = np.ascontiguousarray(episode.rgb[local_idx])
    depth = np.ascontiguousarray(episode.depth[local_idx])
    intr = np.ascontiguousarray(episode.intrinsics[local_idx])
    extr = np.ascontiguousarray(episode.extrinsics_w2c[local_idx])
    h, w = rgb.shape[1:3]
    query_uv = build_grid_points(h, w, int(args.grid_size))
    track = runner.track(rgb, depth, intr, extr, query_uv)

    cam = track["track_cam_xyz"]
    vis = np.asarray(track["visibility"], dtype=np.float32)
    conf = np.asarray(track["confidence"], dtype=np.float32)
    reliable = (
        np.isfinite(cam).all(axis=2)
        & np.isfinite(vis)
        & np.isfinite(conf)
        & (vis >= float(args.visibility_threshold))
        & (conf >= float(args.confidence_threshold))
    )
    world = np.full_like(cam, np.nan, dtype=np.float32)
    for t in range(cam.shape[0]):
        idx = np.where(reliable[t])[0]
        if idx.size:
            world[t, idx] = transform_camera_xyz_to_world(cam[t, idx], extr[t])
    world_smooth = smooth_tracks(world, args.smooth_passes)
    return TrackWindow(
        window_name=f"q{query_index:04d}",
        query_index=int(query_index),
        raw_frame_indices=episode.raw_frame_indices[local_idx],
        query_uv=query_uv,
        coords_world=world_smooth,
        coords_world_raw=world,
        visibility=vis,
        confidence=conf,
        rgb=rgb,
        depth=depth,
        intrinsics=intr,
        extrinsics_w2c=extr,
    )


def save_raw_video(path: Path, rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    for frame in rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def save_tapip3d_npz(path: Path, window: TrackWindow, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=window.coords_world.astype(np.float32),
        coords_raw=window.coords_world_raw.astype(np.float32),
        extrinsics=window.extrinsics_w2c.astype(np.float32),
        intrinsics=window.intrinsics.astype(np.float32),
        depths=window.depth.astype(np.float32),
        video=window.rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0,
        visibs=window.visibility.astype(np.float32),
        confs=window.confidence.astype(np.float32),
        query_uv=window.query_uv.astype(np.float32),
        raw_frame_indices=window.raw_frame_indices.astype(np.int32),
        metadata_json=json.dumps(metadata, indent=2),
    )


def select_track_indices(window: TrackWindow, args: argparse.Namespace) -> np.ndarray:
    coords = window.coords_world
    finite = np.isfinite(coords).all(axis=2)
    valid_count = finite.sum(axis=0)
    disp = np.zeros((coords.shape[1],), dtype=np.float32)
    for p in range(coords.shape[1]):
        idx = np.where(finite[:, p])[0]
        if idx.size >= 2:
            disp[p] = float(np.linalg.norm(coords[idx[-1], p] - coords[idx[0], p]))
    keep = (valid_count >= 3) & (disp >= float(args.motion_threshold))
    idx = np.where(keep)[0]
    if idx.size == 0:
        idx = np.where(valid_count >= 3)[0]
    if idx.size > int(args.max_tracks_per_window):
        idx = idx[np.argsort(disp[idx])[::-1][: int(args.max_tracks_per_window)]]
    return idx.astype(np.int64)


def contiguous_segments(valid: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(valid)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def build_render_tracks(windows: Sequence[TrackWindow], args: argparse.Namespace):
    line_segments: List[np.ndarray] = []
    scalars: List[float] = []
    current_pts: List[np.ndarray] = []
    all_pts: List[np.ndarray] = []
    for wi, window in enumerate(windows):
        for pidx in select_track_indices(window, args).tolist():
            traj = window.coords_world[:, pidx, :]
            valid = np.isfinite(traj).all(axis=1)
            for s0, s1 in contiguous_segments(valid):
                if s1 - s0 + 1 < 2:
                    continue
                seg = traj[s0 : s1 + 1]
                all_pts.append(seg)
                for k in range(seg.shape[0] - 1):
                    line_segments.append(seg[k : k + 2])
                    scalars.append(float(seg[0, 1]))
            if valid[-1]:
                current_pts.append(traj[-1:])
    if not line_segments:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    sv = np.asarray(scalars, dtype=np.float32)
    if float(sv.max() - sv.min()) < 1e-8:
        norm_vals = np.zeros_like(sv)
    else:
        norm_vals = matplotlib.colors.Normalize(vmin=float(sv.min()), vmax=float(sv.max()))(sv)
    rgba = np.asarray(matplotlib.colormaps["rainbow"](norm_vals), dtype=np.float32)
    rgba[:, 3] = float(args.line_alpha)
    pts = np.concatenate(all_pts + current_pts, axis=0) if current_pts else np.concatenate(all_pts, axis=0)
    return line_segments, rgba, pts


def build_scene_cloud(episode: RawEpisode, query_indices: Sequence[int], args: argparse.Namespace):
    pts_all: List[np.ndarray] = []
    cols_all: List[np.ndarray] = []
    for q in query_indices:
        pts, cols = unproject_depth_to_world(
            episode.rgb[q],
            episode.depth[q],
            episode.intrinsics[q],
            episode.extrinsics_w2c[q],
            width=int(args.scene_width),
            height=int(args.scene_height),
        )
        if pts.size:
            pts_all.append(pts)
            cols_all.append(cols)
    if not pts_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    if pts.shape[0] > int(args.pcd_max_points):
        rng = np.random.default_rng(int(args.seed))
        sel = rng.choice(pts.shape[0], int(args.pcd_max_points), replace=False)
        pts = pts[sel]
        cols = cols[sel]
    return pts.astype(np.float32), cols.astype(np.float32)


def set_equal_aspect(ax, pts: np.ndarray, zoom: float) -> None:
    finite = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
    if finite.size == 0:
        return
    lo = np.percentile(finite, 2.0, axis=0)
    hi = np.percentile(finite, 98.0, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(0.55 * float(np.max(hi - lo)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=float(zoom))
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def project_line_segments_to_figure(ax, segments: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Project 3D segments into figure coordinates for a frontmost 2D overlay."""
    projected: List[np.ndarray] = []
    proj = ax.get_proj()
    fig_inv = ax.figure.transFigure.inverted()
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float32)
        if seg.shape[0] < 2 or not np.isfinite(seg).all():
            continue
        xs, ys, _zs = proj3d.proj_transform(seg[:, 0], seg[:, 1], seg[:, 2], proj)
        xy_disp = ax.transData.transform(np.column_stack([xs, ys]))
        projected.append(fig_inv.transform(xy_disp))
    return projected


def add_frontmost_trajectory_overlay(
    fig,
    ax,
    polylines: Sequence[np.ndarray],
    rgba: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if not polylines:
        return
    fig.canvas.draw()
    projected = project_line_segments_to_figure(ax, polylines)
    if not projected:
        return
    colors = rgba[: len(projected)] if rgba.shape[0] >= len(projected) else rgba
    overlay = LineCollection(
        projected,
        colors=colors,
        linewidths=float(args.line_width),
        capstyle="round",
        joinstyle="round",
        transform=fig.transFigure,
        zorder=1000,
    )
    overlay.set_clip_on(False)
    fig.add_artist(overlay)


def render_spaghetti(path: Path, episode: RawEpisode, window: TrackWindow, args: argparse.Namespace) -> None:
    query_indices = [window.query_index]
    cloud_xyz, cloud_rgb = build_scene_cloud(episode, query_indices, args)
    polylines, rgba, track_pts = build_render_tracks([window], args)
    bound = cloud_xyz
    if track_pts.size:
        bound = np.concatenate([cloud_xyz, track_pts], axis=0) if cloud_xyz.size else track_pts

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 8.0), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
    if cloud_xyz.size:
        ax.scatter(
            cloud_xyz[:, 0],
            cloud_xyz[:, 1],
            cloud_xyz[:, 2],
            c=np.clip(cloud_rgb, 0.0, 1.0),
            s=float(args.point_size),
            alpha=float(args.point_alpha),
            depthshade=False,
            linewidths=0,
        )
    set_equal_aspect(ax, bound, args.zoom)
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    add_frontmost_trajectory_overlay(fig, ax, polylines, rgba, args)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def load_benchmark_episodes(benchmark: str, args: argparse.Namespace) -> List[RawEpisode]:
    min_len = (int(args.track_len) - 1) * int(args.raw_stride) + 1
    if benchmark == "calvin":
        return discover_calvin(args.calvin_root, args.num_rollouts, min_len=min_len)
    if benchmark == "libero":
        return discover_libero(args.libero_root, args.num_rollouts, camera_name="agentview")
    if benchmark == "robotwin2":
        return discover_robotwin2(
            args.robotwin2_root,
            variant=args.robotwin2_variant,
            num_rollouts=args.num_rollouts,
            camera_name="head_camera",
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


def process_episode(runner: Optional[SpaTrackerOfflineRunner], episode: RawEpisode, args: argparse.Namespace) -> Dict:
    starts = window_start_indices(episode.rgb.shape[0], args.track_len, args.raw_stride)
    out_dir = args.output_root / episode.benchmark
    raw_video_path = out_dir / "raw_video" / f"{episode.rollout_name}.mp4"
    save_raw_video(raw_video_path, episode.rgb, args.raw_video_fps)

    record = {
        "benchmark": episode.benchmark,
        "rollout_name": episode.rollout_name,
        "source": episode.source,
        "instruction": episode.instruction,
        "num_frames": int(episode.rgb.shape[0]),
        "window_starts": starts,
        "raw_video": str(raw_video_path),
        "spaghetti_images": {},
        "tapip3d_npz": [],
    }
    if args.dry_run:
        return record
    if runner is None:
        raise RuntimeError("runner is required unless --dry-run is set")
    if not starts:
        record["error"] = "episode too short for requested track_len/raw_stride"
        return record

    windows: List[TrackWindow] = []
    for tag, qidx in zip(["start", "mid", "end"], starts):
        print(f"[track] {episode.benchmark}/{episode.rollout_name}/{tag}: q={qidx}")
        window = run_track_window(runner, episode, qidx, args)
        window.window_name = tag
        windows.append(window)
        npz_path = out_dir / "tapip3d_npz" / episode.rollout_name / f"{tag}.npz"
        save_tapip3d_npz(
            npz_path,
            window,
            metadata={
                "benchmark": episode.benchmark,
                "rollout_name": episode.rollout_name,
                "source": episode.source,
                "instruction": episode.instruction,
                "window": tag,
                "query_index": int(qidx),
                "raw_stride": int(args.raw_stride),
                "track_len": int(args.track_len),
            },
        )
        record["tapip3d_npz"].append(str(npz_path))

    for window in windows:
        spaghetti_path = out_dir / "spaghetti" / episode.rollout_name / f"{window.window_name}.png"
        render_spaghetti(spaghetti_path, episode, window, args)
        record["spaghetti_images"][window.window_name] = str(spaghetti_path)
    return record


def main() -> None:
    args = parse_args()
    if args.track_len <= 1:
        raise ValueError("--track-len must be > 1")
    if args.raw_stride <= 0:
        raise ValueError("--raw-stride must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict] = []

    runner = None
    if not args.dry_run:
        print(f"[model] SpaTracker2 root={args.spatracker_root}")
        print(f"[model] checkpoint={args.spatracker_checkpoint}")
        runner = SpaTrackerOfflineRunner(
            spatracker_root=args.spatracker_root,
            checkpoint=args.spatracker_checkpoint,
            device=args.device,
            track_num=max(args.spatracker_track_num, args.grid_size * args.grid_size),
            iters_track=args.spatracker_iters_track,
        )

    for benchmark in args.benchmarks:
        episodes = load_benchmark_episodes(benchmark, args)
        print(f"[data] {benchmark}: selected {len(episodes)} rollout(s)")
        for idx, episode in enumerate(episodes):
            print(f"[episode] {benchmark} {idx + 1}/{len(episodes)} {episode.rollout_name} T={episode.rgb.shape[0]}")
            rec = process_episode(runner, episode, args)
            manifest.append(rec)

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
