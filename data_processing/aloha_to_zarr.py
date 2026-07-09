#!/usr/bin/env python3
"""
Convert the binary-gripper ALOHA LeRobot dataset into a multi-camera zarr for
the flower_actor_3d family (3D front + 3D back + 2D wrist).

Source
------
    /data/user_data/ziyinx/datasets/aloha_binary_gripper
      <task>[1]/data/chunk-000/episode_XXXXXX.parquet
      <task>[1]/videos/chunk-000/observation.images.<key>/episode_XXXXXX.{mp4,mkv}

    Each logical task is split across two folders (e.g. bottle_into_cup and
    bottle_into_cup1). The three logical tasks map to instr_id 0/1/2:
        0 = bottle_into_cup, 1 = insert_marker_into_cup, 2 = stack_two_blocks

    parquet columns used:
        observation.right_eef_pose : (10,) [rot6d(6), trans(3), gripper_deg(1)]   raw 0~100
        action.right_eef_pose      : (10,) [rot6d(6), trans(3), gripper_bin(1)]   already 0/1

    videos:
        cam_azure_kinect_front.color            h264   1280x720  -> rgb_front (3D)
        cam_azure_kinect_front.transformed_depth ffv1  gray16le  -> depth_front (3D, mm, lossless)
        cam_azure_kinect_back.color             h264   1280x720  -> rgb_back  (3D)
        cam_azure_kinect_back.transformed_depth ffv1  gray16le  -> depth_back  (3D, mm, lossless)
        cam_wrist                               h264   1280x720  -> rgb_wrist (2D, rgb only)

Output zarr schema (N = total frames; per-camera keys, robotwin2 convention)
----------------------------------------------------------------------------
    rgb_front      : (N, 1, 3, 224, 398) uint8
    rgb_back       : (N, 1, 3, 224, 398) uint8
    rgb_wrist      : (N, 1, 3, 224, 398) uint8     2D camera, RGB only
    depth_front    : (N, 1, 224, 398) float32      meters
    depth_back     : (N, 1, 224, 398) float32      meters
    proprioception : (N, 1, 10) float32            [rot6d(6), trans(3), gripper_bin(1)]
    rel_action     : (N, K=15, 1, 10) float32      [rot6d_delta(6), trans_delta(3), gripper_bin(1)]
    instr_id       : (N, 1) int32

Sidecars written next to the zarr:
    instructions.json        : ["<task0>", "<task1>", "<task2>"], indexed by instr_id
    camera_calibration.json  : front/back intrinsics (source 720x1280 + rescaled 224x398)
                               and world_from_cam extrinsics (4x4). Feeds the multiview
                               depth->world unprojection on the consumer side.
    xyz_delta_stats.json     : per-axis min/max/mean/abs-p99 of the rel_action xyz deltas
                               (anchor-relative grows over the horizon, so the trainer's
                               XYZ_MIN/MAX must be recomputed from this).

Conventions (decided with the data owner)
------------------------------------------
  * Layout matches datasets/aloha.py: zarr stores [rot6d(6), trans(3), grip(1)];
    the consumer reorders to [trans, rot6d, grip].
  * rel_action is ANCHOR-RELATIVE: every one of the K steps is measured against the
    single proprio anchor at t (NOT step-to-step incremental):
        trans_delta[k] = action[t+k].trans - proprio[t].trans
        rot6d_delta[k] = first_two_cols( R(action[t+k]) @ R(proprio[t]).T )
        gripper[k]     = action[t+k].gripper        (absolute binary)
    NOTE: deploy_aloha_policy._relative_to_absolute currently reconstructs
    INCREMENTALLY (cumsum / left-compose) and the trainer's fixed XYZ_MIN/MAX were
    fit on incremental deltas. Both must be updated to consume this zarr.
  * Static orientation maps to rot6d_delta = [1,0,0,0,1,0] (identity); we do NOT
    subtract identity, so the deploy Gram-Schmidt path stays valid.
  * proprio gripper reuses the dataset's already-binarized open/close signal
    (action.right_eef_pose[9], phase-labeled by binarize_aloha_gripper.py) sampled at
    the same frame t. We do NOT re-threshold the raw observation degrees, so proprio
    and action share one clean binary open/close signal (1=open, 0=closed).

The rot6d convention is Zhou et al. ortho6d = first two COLUMNS of R, identical to
modeling/utils/utils.compute_rotation_matrix_from_ortho6d (verified by self-test).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_ROOT = Path("/data/user_data/ziyinx/datasets/aloha_binary_gripper")
DEFAULT_OUT = Path("/data/group_data/katefgroup/VLA/zarr_datasets/aloha_zarr/")
DEFAULT_CALIB = Path(
    "/data/user_data/ziyinx/lerobot/lerobot/scripts/aloha_calibration/calibration_multiview.json"
)

# logical task -> (instr_id, source folders)
TASKS: dict[str, tuple[int, tuple[str, ...]]] = {
    "bottle_into_cup": (0, ("bottle_into_cup", "bottle_into_cup1")),
    "insert_marker_into_cup": (1, ("insert_marker_into_cup", "insert_marker_into_cup1")),
    "stack_two_blocks": (2, ("stack_two_blocks", "stack_two_blocks1")),
}

SOURCE_HW = (720, 1280)
TARGET_H, TARGET_W = 224, 398          # shorter side -> 224, preserves 720:1280
ACTION_HORIZON = 15                    # = train prediction_len / chunk_size

PROPRIO_COL = "observation.right_eef_pose"
ACTION_COL = "action.right_eef_pose"

CAM_KEYS = {
    "rgb_front": ("observation.images.cam_azure_kinect_front.color", "mp4", "rgb"),
    "rgb_back": ("observation.images.cam_azure_kinect_back.color", "mp4", "rgb"),
    "rgb_wrist": ("observation.images.cam_wrist", "mp4", "rgb"),
    "depth_front": ("observation.images.cam_azure_kinect_front.transformed_depth", "mkv", "depth"),
    "depth_back": ("observation.images.cam_azure_kinect_back.transformed_depth", "mkv", "depth"),
}


# --------------------------------------------------------------------------- #
# rot6d helpers (numpy port of modeling/utils/utils.py; first two columns of R)
# --------------------------------------------------------------------------- #
def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) with columns [x, y, z] (Zhou et al. ortho6d)."""
    x = _normalize(r6[..., 0:3])
    z = _normalize(np.cross(x, r6[..., 3:6]))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def matrix_to_rot6d(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6) = [first column, second column]."""
    return np.concatenate([m[..., 0], m[..., 1]], axis=-1)


def _selftest_rot6d() -> None:
    rng = np.random.default_rng(0)
    from numpy.linalg import svd

    mats = []
    for _ in range(64):
        a = rng.standard_normal((3, 3))
        u, _, vt = svd(a)
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        mats.append(R)
    R = np.stack(mats)
    r6 = matrix_to_rot6d(R)
    R2 = rot6d_to_matrix(r6)
    assert np.allclose(R, R2, atol=1e-5), "rot6d round-trip failed"
    # identity -> [1,0,0,0,1,0]
    eye6 = matrix_to_rot6d(np.eye(3))
    assert np.allclose(eye6, [1, 0, 0, 0, 1, 0]), eye6


# --------------------------------------------------------------------------- #
# ffmpeg decoding (raw frames; preserves 16-bit depth losslessly)
# --------------------------------------------------------------------------- #
def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found (install imageio-ffmpeg or system ffmpeg).")
        return exe


def _decode_video(path: Path, kind: str, ff: str) -> np.ndarray:
    """Decode an entire video to raw frames at source resolution.

    kind='rgb'   -> (T, H, W, 3) uint8
    kind='depth' -> (T, H, W)    uint16  (raw sensor units, mm)
    """
    h, w = SOURCE_HW
    if kind == "rgb":
        pix_fmt, dtype, ch = "rgb24", np.uint8, 3
    else:
        pix_fmt, dtype, ch = "gray16le", np.dtype("<u2"), 1
    cmd = [ff, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}:\n{proc.stderr.decode(errors='ignore')[:500]}")
    frame_elems = h * w * ch
    arr = np.frombuffer(proc.stdout, dtype=dtype)
    n = arr.size // frame_elems
    arr = arr[: n * frame_elems]
    return arr.reshape(n, h, w, 3) if ch == 3 else arr.reshape(n, h, w)


def _resize_rgb(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, 3, 224, 398) uint8 via INTER_AREA."""
    import cv2

    out = np.empty((frames.shape[0], 3, TARGET_H, TARGET_W), dtype=np.uint8)
    for i in range(frames.shape[0]):
        small = cv2.resize(frames[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        out[i] = small.transpose(2, 0, 1)
    return out


def _resize_depth(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) uint16 mm -> (T, 224, 398) float32 meters via nearest."""
    import cv2

    out = np.empty((frames.shape[0], TARGET_H, TARGET_W), dtype=np.float32)
    for i in range(frames.shape[0]):
        small = cv2.resize(frames[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        out[i] = small.astype(np.float32) / 1000.0
    return out


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def load_calibration(calib_json: Path) -> dict:
    """Load front/back intrinsics + extrinsics and rescale intrinsics to target HxW."""
    spec = json.loads(calib_json.read_text())
    base = calib_json.parent.parent  # paths inside are relative to .../scripts/
    src_h, src_w = SOURCE_HW
    sx, sy = TARGET_W / src_w, TARGET_H / src_h

    out: dict = {"target_hw": [TARGET_H, TARGET_W], "source_hw": list(SOURCE_HW), "cameras": {}}
    name_map = {"cam_azure_kinect_front": "front", "cam_azure_kinect_back": "back"}
    for cam_name, entry in spec.items():
        short = name_map.get(cam_name, cam_name)
        K = np.loadtxt(base / entry["intrinsics"]).astype(np.float64)
        T = np.loadtxt(base / entry["extrinsics"]).astype(np.float64)  # world_from_cam
        K_rs = K.copy()
        K_rs[0, 0] *= sx
        K_rs[0, 2] *= sx
        K_rs[1, 1] *= sy
        K_rs[1, 2] *= sy
        out["cameras"][short] = {
            "intrinsics_source": K.tolist(),
            "intrinsics_resized": K_rs.tolist(),
            "world_from_cam": T.tolist(),
        }
    out["wrist"] = "2D camera: RGB only, no depth/calibration used by the model."
    return out


# --------------------------------------------------------------------------- #
# Per-episode tensor construction
# --------------------------------------------------------------------------- #
def build_proprio(obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    """proprio (T,1,10): observation [rot6d(6), trans(3)] + the dataset's already-binary
    gripper (action.right_eef_pose[9]) at the same frame. No re-thresholding."""
    proprio = obs.copy().astype(np.float32)
    proprio[:, 9] = act[:, 9].astype(np.float32)  # already 0/1 (1=open, 0=closed)
    return proprio[:, None, :]


def build_rel_action(obs: np.ndarray, act: np.ndarray, horizon: int) -> np.ndarray:
    """Anchor-relative deltas. obs/act (T,10). Returns (T, horizon, 1, 10)."""
    t = obs.shape[0]
    obs_R = rot6d_to_matrix(obs[:, 0:6])              # (T,3,3)
    act_R = rot6d_to_matrix(act[:, 0:6])              # (T,3,3)
    obs_trans = obs[:, 6:9]                           # (T,3)
    act_trans = act[:, 6:9]                           # (T,3)
    act_grip = act[:, 9]                              # (T,) binary

    # gather targets for each (t, k), clamping the tail to the last frame
    k = np.arange(horizon)
    idx = np.minimum(np.arange(t)[:, None] + k[None, :], t - 1)   # (T,horizon)

    tgt_R = act_R[idx]                               # (T,horizon,3,3)
    obs_R_T = np.transpose(obs_R, (0, 2, 1))         # (T,3,3)
    dR = np.einsum("tkij,tjl->tkil", tgt_R, obs_R_T)  # R_target @ R_anchor^T
    rot6d_delta = matrix_to_rot6d(dR)                # (T,horizon,6)

    dtrans = act_trans[idx] - obs_trans[:, None, :]  # (T,horizon,3)
    grip = act_grip[idx][..., None]                  # (T,horizon,1)

    rel = np.concatenate([rot6d_delta, dtrans, grip], axis=-1).astype(np.float32)
    return rel[:, :, None, :]                        # (T,horizon,1,10)


# --------------------------------------------------------------------------- #
# Episode enumeration / split
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    task: str
    instr_id: int
    folder: str
    ep_index: int
    parquet: Path
    chunk: str
    n_frames: int


def enumerate_episodes(root: Path, tasks: list[str], limit: int | None) -> list[Episode]:
    eps: list[Episode] = []
    for task in tasks:
        instr_id, folders = TASKS[task]
        for folder in folders:
            fdir = root / folder
            if not fdir.exists():
                print(f"  [skip] missing folder {fdir}")
                continue
            parquets = sorted(fdir.glob("data/chunk-*/episode_*.parquet"))
            if limit is not None:
                parquets = parquets[:limit]
            for pq in parquets:
                chunk = pq.parent.name
                ep_index = int(pq.stem.split("_")[-1])
                n = pd.read_parquet(pq, columns=["frame_index"]).shape[0]
                eps.append(Episode(task, instr_id, folder, ep_index, pq, chunk, n))
    return eps


def video_path(root: Path, ep: Episode, zarr_key: str) -> Path:
    cam_key, ext, _ = CAM_KEYS[zarr_key]
    return root / ep.folder / "videos" / ep.chunk / cam_key / f"episode_{ep.ep_index:06d}.{ext}"


def split_episodes(eps: list[Episode], val_per_folder: int) -> tuple[list[Episode], list[Episode]]:
    """Hold out `val_per_folder` episode(s) (highest ep_index) from EACH source folder.

    With the default of 1, each logical task (2 folders) yields 2 val rollouts -> 6 total.
    Never empties a folder's train set.
    """
    train, val = [], []
    for task, folder in sorted({(e.task, e.folder) for e in eps}):
        fe = sorted([e for e in eps if e.task == task and e.folder == folder], key=lambda e: e.ep_index)
        k = min(val_per_folder, len(fe) - 1) if val_per_folder > 0 and len(fe) > 1 else 0
        val.extend(fe[len(fe) - k:] if k else [])
        train.extend(fe[: len(fe) - k] if k else fe)
    train.sort(key=lambda e: (e.instr_id, e.folder, e.ep_index))
    val.sort(key=lambda e: (e.instr_id, e.folder, e.ep_index))
    return train, val


# --------------------------------------------------------------------------- #
# Zarr writing
# --------------------------------------------------------------------------- #
def make_zarr(path: Path, n: int, horizon: int, chunk: int):
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    g = zarr.open_group(str(path), mode="w")
    h, w = TARGET_H, TARGET_W
    img = lambda: dict(chunks=(chunk, 1, 3, h, w), dtype="u1", compressor=comp)
    dep = lambda: dict(chunks=(chunk, 1, h, w), dtype="f4", compressor=comp)
    g.create_dataset("rgb_front", shape=(n, 1, 3, h, w), **img())
    g.create_dataset("rgb_back", shape=(n, 1, 3, h, w), **img())
    g.create_dataset("rgb_wrist", shape=(n, 1, 3, h, w), **img())
    g.create_dataset("depth_front", shape=(n, 1, h, w), **dep())
    g.create_dataset("depth_back", shape=(n, 1, h, w), **dep())
    g.create_dataset("proprioception", shape=(n, 1, 10), chunks=(chunk, 1, 10), dtype="f4", compressor=comp)
    g.create_dataset("rel_action", shape=(n, horizon, 1, 10), chunks=(chunk, horizon, 1, 10), dtype="f4", compressor=comp)
    g.create_dataset("instr_id", shape=(n, 1), chunks=(chunk, 1), dtype="i4", compressor=comp)
    return g


@dataclass
class XyzStats:
    lo: np.ndarray = field(default_factory=lambda: np.full(3, np.inf))
    hi: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf))
    abs_samples: list = field(default_factory=list)

    def update(self, dtrans: np.ndarray):
        flat = dtrans.reshape(-1, 3)
        self.lo = np.minimum(self.lo, flat.min(0))
        self.hi = np.maximum(self.hi, flat.max(0))
        # subsample abs values for p99 without keeping everything
        a = np.abs(flat)
        if a.shape[0] > 4000:
            a = a[np.random.default_rng(0).choice(a.shape[0], 4000, replace=False)]
        self.abs_samples.append(a)

    def summary(self) -> dict:
        allabs = np.concatenate(self.abs_samples, 0) if self.abs_samples else np.zeros((1, 3))
        return {
            "min": self.lo.tolist(),
            "max": self.hi.tolist(),
            "abs_p99": np.percentile(allabs, 99, axis=0).tolist(),
            "abs_mean": allabs.mean(0).tolist(),
        }


def fill_split(name: str, eps: list[Episode], root: Path, g, horizon: int,
               ff: str) -> XyzStats:
    stats = XyzStats()
    cursor = 0
    total = len(eps)
    for ei, ep in enumerate(eps):
        df = pd.read_parquet(ep.parquet, columns=[PROPRIO_COL, ACTION_COL])
        obs = np.stack([np.asarray(v, np.float32) for v in df[PROPRIO_COL].values])
        act = np.stack([np.asarray(v, np.float32) for v in df[ACTION_COL].values])
        n = obs.shape[0]
        s, e = cursor, cursor + n

        g["proprioception"][s:e] = build_proprio(obs, act)
        rel = build_rel_action(obs, act, horizon)
        g["rel_action"][s:e] = rel
        stats.update(rel[..., 6:9])
        g["instr_id"][s:e] = np.full((n, 1), ep.instr_id, np.int32)

        for key, (_, _, kind) in CAM_KEYS.items():
            vid = video_path(root, ep, key)
            if not vid.exists():
                raise FileNotFoundError(f"missing video {vid}")
            frames = _decode_video(vid, kind, ff)
            if frames.shape[0] != n:
                raise ValueError(
                    f"{vid}: {frames.shape[0]} frames != {n} parquet rows ({ep.folder} ep{ep.ep_index})"
                )
            if kind == "rgb":
                g[key][s:e] = _resize_rgb(frames)[:, None]
            else:
                g[key][s:e] = _resize_depth(frames)[:, None]
            del frames
        cursor = e
        print(f"  [{name}] {ei + 1}/{total} {ep.folder} ep{ep.ep_index:03d} "
              f"frames={n} (total={cursor})", flush=True)
    assert cursor == g["rel_action"].shape[0], (cursor, g["rel_action"].shape[0])
    return stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--horizon", type=int, default=ACTION_HORIZON)
    p.add_argument("--val-per-folder", type=int, default=1,
                   help="Held-out val episodes per source folder (default 1 -> 6 val total).")
    p.add_argument("--zarr-chunk", type=int, default=8)
    p.add_argument("--limit-episodes-per-folder", type=int, default=None,
                   help="Smoke-test: only convert the first N episodes of each folder.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_instructions(root: Path, tasks: list[str]) -> list[str]:
    """instr_id -> instruction string (from each task's meta/tasks.jsonl)."""
    texts: dict[int, str] = {}
    for task in tasks:
        instr_id, folders = TASKS[task]
        tj = root / folders[0] / "meta" / "tasks.jsonl"
        line = tj.read_text().strip().splitlines()[0]
        texts[instr_id] = json.loads(line)["task"]
    return [texts[i] for i in sorted(texts)]


def main() -> None:
    args = parse_args()
    _selftest_rot6d()
    ff = _ffmpeg_exe()

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} exists; pass --overwrite.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Enumerating episodes...")
    eps = enumerate_episodes(args.root, args.tasks, args.limit_episodes_per_folder)
    train, val = split_episodes(eps, args.val_per_folder)
    n_train = sum(e.n_frames for e in train)
    n_val = sum(e.n_frames for e in val)
    print(f"  {len(eps)} episodes -> train {len(train)} eps / {n_train} frames, "
          f"val {len(val)} eps / {n_val} frames")

    # sidecars
    (args.out / "instructions.json").write_text(
        json.dumps(load_instructions(args.root, args.tasks), indent=2)
    )
    (args.out / "camera_calibration.json").write_text(
        json.dumps(load_calibration(args.calib), indent=2)
    )

    summaries = {}
    for name, split in (("train", train), ("val", val)):
        if not split:
            continue
        n = sum(e.n_frames for e in split)
        zpath = args.out / f"{name}.zarr"
        print(f"Writing {zpath} ({n} frames)...")
        g = make_zarr(zpath, n, args.horizon, args.zarr_chunk)
        stats = fill_split(name, split, args.root, g, args.horizon, ff)
        summaries[name] = stats.summary()

    (args.out / "xyz_delta_stats.json").write_text(json.dumps(summaries, indent=2))
    print("\nDone. xyz_delta_stats (anchor-relative; recompute trainer XYZ_MIN/MAX from this):")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
