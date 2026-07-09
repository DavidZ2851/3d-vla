#!/usr/bin/env python3
"""Verify PerAct2 motion annotations against raw RLBench depth/camera metadata.

The strict check verifies that saved query_depth agrees with raw RLBench
front_depth PNG decoding and that RLBench front_camera_extrinsics are used as
camera-to-world matrices, which matches utils.depth2cloud.RLBenchDepth2Cloud.

The saved query_world_xyz values are also compared to direct raw-depth
unprojection, but this is diagnostic by default. Those values come from
SpaTracker's refined track3d_pred output, not from raw query_depth unprojection.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


DEPTH_SCALE = float(2**24 - 1)


class _RLBenchPickleCompatObject:
    def __init__(self, *args, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state

    def __len__(self):
        return len(getattr(self, "_observations", []))

    def __getitem__(self, idx):
        return getattr(self, "_observations")[idx]

    def __iter__(self):
        return iter(getattr(self, "_observations", []))


class _RLBenchCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if str(module).startswith("rlbench."):
            return _RLBenchPickleCompatObject
        return super().find_class(module, name)


def load_rlbench_pickle(path: Path):
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except (ModuleNotFoundError, AttributeError) as exc:
            if "rlbench" not in str(exc).lower() and "bimanual" not in str(exc).lower():
                raise
    with path.open("rb") as f:
        return _RLBenchCompatUnpickler(f).load()


def decode_rlbench_depth_png(path: Path, near: float, far: float) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        depth01 = np.sum(
            arr.astype(np.float32) * np.array([65536.0, 256.0, 1.0], dtype=np.float32),
            axis=2,
        ) / DEPTH_SCALE
    else:
        depth01 = arr.astype(np.float32) / DEPTH_SCALE
    return (float(near) + depth01 * (float(far) - float(near))).astype(np.float32)


def bilinear_depth_sample(depth: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u = uv[:, 0]
    v = uv[:, 1]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    in_bounds = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    if not np.any(in_bounds):
        return out, np.zeros_like(in_bounds, dtype=bool)

    idx = np.where(in_bounds)[0]
    x0v, x1v = x0[idx], x1[idx]
    y0v, y1v = y0[idx], y1[idx]
    du = u[idx] - x0v.astype(np.float32)
    dv = v[idx] - y0v.astype(np.float32)
    z00 = depth[y0v, x0v]
    z10 = depth[y0v, x1v]
    z01 = depth[y1v, x0v]
    z11 = depth[y1v, x1v]
    valid = (
        np.isfinite(z00)
        & np.isfinite(z10)
        & np.isfinite(z01)
        & np.isfinite(z11)
        & (z00 > 0.0)
        & (z10 > 0.0)
        & (z01 > 0.0)
        & (z11 > 0.0)
    )
    if np.any(valid):
        idx2 = idx[valid]
        du2 = du[valid]
        dv2 = dv[valid]
        out[idx2] = (
            (1.0 - du2) * (1.0 - dv2) * z00[valid]
            + du2 * (1.0 - dv2) * z10[valid]
            + (1.0 - du2) * dv2 * z01[valid]
            + du2 * dv2 * z11[valid]
        )
    return out, np.isfinite(out) & (out > 0.0)


def unproject_c2w(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics_c2w: np.ndarray) -> np.ndarray:
    uv1 = np.concatenate([uv.astype(np.float32), np.ones((uv.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_cam = (np.linalg.inv(intrinsics).astype(np.float32) @ uv1.T).T * depth[:, None]
    xyz_cam_h = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_world_h = (extrinsics_c2w.astype(np.float32) @ xyz_cam_h.T).T
    return xyz_world_h[:, :3]


def iter_annotation_files(root: Path, max_files: int) -> Iterable[Path]:
    count = 0
    for path in root.rglob("annotations/*.npz"):
        yield path
        count += 1
        if count >= max_files:
            return


def parse_annotation_ref(path: Path, annotations_root: Path) -> Tuple[str, str, int]:
    rel = path.relative_to(annotations_root)
    split = rel.parts[1] if rel.parts[0] == "front" else rel.parts[0]
    task = rel.parts[2] if rel.parts[0] == "front" else rel.parts[1]
    unit = rel.parts[3] if rel.parts[0] == "front" else rel.parts[2]
    match = re.search(r"episode_(\d+)", unit)
    if not match:
        raise ValueError(f"Could not parse episode index from unit directory: {unit}")
    return split, task, int(match.group(1))


def raw_episode_dir(raw_root: Path, split: str, task: str, episode_index: int) -> Path:
    return raw_root / split / task / "all_variations" / "episodes" / f"episode{episode_index}"


def check_file(
    npz_path: Path,
    annotations_root: Path,
    raw_root: Path,
    camera: str,
) -> Tuple[int, float, float, float, float]:
    split, task, episode_index = parse_annotation_ref(npz_path, annotations_root)
    episode_dir = raw_episode_dir(raw_root, split, task, episode_index)
    with np.load(npz_path, allow_pickle=False) as data:
        frame_id = int(np.asarray(data["frame_id_in_episode"]).item())
        query_uv = np.asarray(data["query_uv"], dtype=np.float32)
        saved_depth = np.asarray(data["query_depth"], dtype=np.float32)
        saved_world = np.asarray(data["query_world_xyz"], dtype=np.float32)

    demo = load_rlbench_pickle(episode_dir / "low_dim_obs.pkl")
    obs = demo[frame_id]
    misc = obs.misc
    near = float(misc[f"{camera}_camera_near"])
    far = float(misc[f"{camera}_camera_far"])
    intrinsics = np.asarray(misc[f"{camera}_camera_intrinsics"], dtype=np.float32)
    extrinsics_c2w = np.asarray(misc[f"{camera}_camera_extrinsics"], dtype=np.float32)
    depth_path = episode_dir / f"{camera}_depth" / f"depth_{frame_id:04d}.png"
    depth = decode_rlbench_depth_png(depth_path, near=near, far=far)
    sampled_depth, depth_valid = bilinear_depth_sample(depth, query_uv)
    valid = depth_valid & np.isfinite(saved_depth) & (saved_depth > 0)
    valid &= np.isfinite(saved_world).all(axis=1)
    if not np.any(valid):
        return 0, float("nan"), float("nan"), float("nan"), float("nan")

    recon_world = unproject_c2w(query_uv[valid], sampled_depth[valid], intrinsics, extrinsics_c2w)
    depth_err = np.abs(sampled_depth[valid] - saved_depth[valid])
    xyz_err = np.linalg.norm(recon_world - saved_world[valid], axis=1)
    return (
        int(valid.sum()),
        float(depth_err.mean()),
        float(depth_err.max()),
        float(xyz_err.mean()),
        float(xyz_err.max()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("/data/group_data/katefgroup/datasets/ziyinx/motion_annotations/peract2"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/group_data/katefgroup/VLA/peract2_raw_squash"),
    )
    parser.add_argument("--camera", type=str, default="front")
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument(
        "--max-depth-err",
        type=float,
        default=1e-4,
        help="Maximum allowed raw-depth-vs-saved-query_depth absolute error.",
    )
    parser.add_argument(
        "--max-saved-query-world-err",
        type=float,
        default=1e-4,
        help="Maximum saved query_world_xyz raw-unprojection L2 error used with --strict-saved-query-world.",
    )
    parser.add_argument(
        "--strict-saved-query-world",
        action="store_true",
        help="Fail if saved SpaTracker query_world_xyz differs from direct raw-depth unprojection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_root = args.annotations_root
    if not (annotations_root / "front").is_dir() and (annotations_root / args.camera).is_dir():
        annotations_root = annotations_root / args.camera

    total_points = 0
    depth_mean_errs: List[float] = []
    depth_max_errs: List[float] = []
    xyz_mean_errs: List[float] = []
    xyz_max_errs: List[float] = []
    checked_files = 0
    for path in iter_annotation_files(annotations_root, max_files=int(args.max_files)):
        points, depth_mean, depth_max, xyz_mean, xyz_max = check_file(
            path,
            annotations_root,
            args.raw_root,
            args.camera,
        )
        if points <= 0:
            continue
        checked_files += 1
        total_points += points
        depth_mean_errs.append(depth_mean)
        depth_max_errs.append(depth_max)
        xyz_mean_errs.append(xyz_mean)
        xyz_max_errs.append(xyz_max)

    if checked_files <= 0:
        raise RuntimeError("No valid annotation points were checked.")

    report = {
        "checked_files": checked_files,
        "checked_points": total_points,
        "mean_depth_abs_error": float(np.mean(depth_mean_errs)),
        "max_depth_abs_error": float(np.max(depth_max_errs)),
        "mean_xyz_l2_error": float(np.mean(xyz_mean_errs)),
        "max_xyz_l2_error": float(np.max(xyz_max_errs)),
        "extrinsics_convention": "front_camera_extrinsics is camera-to-world",
        "depth_units": "meters",
        "saved_query_world_note": (
            "query_world_xyz is SpaTracker refined track3d_pred transformed to world; "
            "it is not expected to exactly equal raw query_depth unprojection."
        ),
    }
    print(report)
    if report["max_depth_abs_error"] > float(args.max_depth_err):
        raise RuntimeError(
            f"PerAct2 depth decode check failed: max_depth_abs_error="
            f"{report['max_depth_abs_error']} > {args.max_depth_err}"
        )
    if args.strict_saved_query_world and report["max_xyz_l2_error"] > float(args.max_saved_query_world_err):
        raise RuntimeError(
            f"PerAct2 saved query_world check failed: max_xyz_l2_error="
            f"{report['max_xyz_l2_error']} > {args.max_saved_query_world_err}"
        )


if __name__ == "__main__":
    main()
