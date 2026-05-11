#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import pytorch3d_transforms
from utils.depth2cloud.robotwin2 import RoboTwinDepth2Cloud

DEFAULT_TRAIN_DATA_DIR = "/data/user_data/ziyinx/datasets/zarr_datasets/robotwin2_zarr/train_rechunked8.zarr"
DEFAULT_EVAL_DATA_DIR = "/data/group_data/katefgroup/datasets/ziyinx/robotwin2_zarr/val_rechunked8.zarr"
FIXED_XYZ_MIN = torch.tensor([-0.025, -0.022, -0.020], dtype=torch.float32)
FIXED_XYZ_MAX = torch.tensor([0.025, 0.022, 0.020], dtype=torch.float32)
FIXED_OPENNESS_MIN = torch.tensor(-0.07537692785263062, dtype=torch.float32)
FIXED_OPENNESS_MAX = torch.tensor(0.07537692785263062, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check RoboTwin2 val loader by visualizing one random datapoint: "
            "merged 3-cam point cloud + proprio + future action (rel->abs)."
        )
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=DEFAULT_TRAIN_DATA_DIR,
        help="Deprecated compatibility arg; ignored by this script.",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=DEFAULT_EVAL_DATA_DIR,
        help="Val zarr path to sample a datapoint from.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=str(Path(__file__).parent / "instructions/robotwin2/instructions_full.json"),
        help="Instruction metadata used by RoboTwin2Dataset.",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--val-mem-limit-gb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional fixed val chunk index (0 <= idx < len(val_dataset)).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Optional frame index inside chunk [0, chunk_size).",
    )
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).parent
            / "vis_outputs/robotwin2_val_loader_random_datapoint.png"
        ),
    )
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _candidate_val_paths(user_path: str) -> list[str]:
    return [
        user_path,
        "/data/group_data/katefgroup/datasets/ziyinx/robotwin2_zarr/val_rechunked8.zarr",
        "/data/group_data/katefgroup/datasets/ziyinx/robotwin2_zarr_rechunked_stride5/val_rechunked8.zarr",
        "/data/group_data/katefgroup/VLA/zarr_datasets/robotwin2_zarr/shards/shard_00/val.zarr",
    ]


def resolve_existing_zarr_path(candidates: list[str], tag: str) -> str:
    import zarr

    deduped: list[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    tried: list[str] = []
    for c in deduped:
        p = Path(c)
        tried.append(c)
        if not p.exists():
            continue
        try:
            zarr.open_group(str(p), mode="r")
            return str(p)
        except Exception:
            continue
    tried_lines = "\n".join([f"  - {x}" for x in tried])
    raise FileNotFoundError(
        f"Could not resolve a readable {tag} zarr group. Tried:\n{tried_lines}"
    )


def relative_to_absolute(action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """Mirror utils.trainers.base.relative_to_absolute for quat_wxyz."""
    pos = proprio[..., :3] + action[..., :3].cumsum(1)
    rel_orn = action[..., 3:7]
    curr_orn = proprio[..., 3:7]
    abs_orns = []
    for t in range(rel_orn.shape[1]):
        q_rel = rel_orn[:, t:t + 1]
        next_orn = pytorch3d_transforms.quaternion_multiply(q_rel, curr_orn)
        abs_orns.append(next_orn)
        curr_orn = next_orn
    orn = torch.cat(abs_orns, dim=1)
    return torch.cat([pos, orn, action[..., -1:]], dim=-1)


def normalize_rel_action_fixed_range(rel_action: torch.Tensor) -> torch.Tensor:
    """Normalize with fixed RoboTwin2 ranges used by the trainer."""
    out = rel_action.clone()
    xyz_min = FIXED_XYZ_MIN.to(out.device, out.dtype)
    xyz_max = FIXED_XYZ_MAX.to(out.device, out.dtype)
    xyz_diff = (xyz_max - xyz_min).clamp_min(1e-6)
    out[..., :3] = (out[..., :3] - xyz_min) / xyz_diff * 2.0 - 1.0

    open_min = FIXED_OPENNESS_MIN.to(out.device, out.dtype)
    open_max = FIXED_OPENNESS_MAX.to(out.device, out.dtype)
    open_diff = (open_max - open_min).clamp_min(1e-6)
    out[..., 7] = (out[..., 7] - open_min) / open_diff * 2.0 - 1.0
    return out


def depth_rgb_to_points(
    depth2cloud: RoboTwinDepth2Cloud,
    depth_hw: torch.Tensor,
    rgb_chw: torch.Tensor,
    extrinsic_w2c: torch.Tensor,
    point_stride: int,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        pcd = depth2cloud.deproject(
            depth_hw[None].float(),
            extrinsic_w2c[None].float()
        )[0]
    pcd_hwc = pcd.permute(1, 2, 0).cpu().numpy()
    rgb_hwc = rgb_chw.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
    depth_np = depth_hw.cpu().numpy()

    pcd_hwc = pcd_hwc[::point_stride, ::point_stride]
    rgb_hwc = rgb_hwc[::point_stride, ::point_stride]
    depth_np = depth_np[::point_stride, ::point_stride]

    valid = (
        np.isfinite(depth_np)
        & (depth_np > 1e-6)
        & (depth_np < depth_max)
        & np.isfinite(pcd_hwc).all(axis=-1)
    )
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    return pcd_hwc[valid].astype(np.float32), rgb_hwc[valid].astype(np.float32)


def quat_axis_wxyz(quat_wxyz: np.ndarray, axis: int = 0) -> np.ndarray:
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float32).reshape(1, 4)
    rot = pytorch3d_transforms.quaternion_to_matrix(quat)[0].cpu().numpy()
    vec = rot[:, axis]
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.6 * float(np.max(np.maximum(hi - lo, 1e-3)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from datasets.robotwin2 import RoboTwin2Dataset

    eval_data_dir = resolve_existing_zarr_path(
        _candidate_val_paths(args.eval_data_dir), tag="val"
    )
    if eval_data_dir != args.eval_data_dir:
        print(
            "[WARN] eval-data-dir fallback in use: "
            f"requested={args.eval_data_dir} resolved={eval_data_dir}"
        )

    val_ds = RoboTwin2Dataset(
        root=eval_data_dir,
        instructions=args.instructions,
        copies=1,
        relative_action=True,
        mem_limit=args.val_mem_limit_gb,
        chunk_size=args.chunk_size,
    )
    val_len = len(val_ds)
    if val_len <= 0:
        raise RuntimeError("Val dataset is empty.")

    chunk_idx = int(args.index) if args.index is not None else int(rng.integers(0, val_len))
    if chunk_idx < 0 or chunk_idx >= val_len:
        raise IndexError(f"chunk index {chunk_idx} out of bounds [0, {val_len})")

    sample = val_ds[chunk_idx]
    c = int(sample["action"].shape[0])
    frame_idx = int(args.frame_index) if args.frame_index is not None else int(rng.integers(0, c))
    if frame_idx < 0 or frame_idx >= c:
        raise IndexError(f"frame index {frame_idx} out of bounds [0, {c})")

    instr = sample["instr"][frame_idx]
    rel_action = sample["action"][frame_idx].float()  # (T, 2, 8)
    proprio = sample["proprioception"][frame_idx].float()  # (1, 2, 8)
    abs_action = relative_to_absolute(rel_action[None], proprio[None])[0]  # (T, 2, 8)

    rel_action_norm = normalize_rel_action_fixed_range(rel_action)

    depth2cloud = RoboTwinDepth2Cloud(source_hw=(224, 224))
    cam_names = ("head", "left", "right")
    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    extrinsics = sample["extrinsics"][frame_idx].float()  # (3, 4, 4)

    for cam_i, cam_name in enumerate(cam_names):
        depth = sample[f"depth_{cam_name}"][frame_idx, 0]
        rgb = sample[f"rgb_{cam_name}"][frame_idx, 0]
        pts, cols = depth_rgb_to_points(
            depth2cloud=depth2cloud,
            depth_hw=depth,
            rgb_chw=rgb,
            extrinsic_w2c=extrinsics[cam_i],
            point_stride=args.point_stride,
            depth_max=args.depth_max,
        )
        if pts.shape[0] > 0:
            points_all.append(pts)
            colors_all.append(cols)

    if points_all:
        points = np.concatenate(points_all, axis=0)
        colors = np.concatenate(colors_all, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    prop_np = proprio[0].cpu().numpy()  # (2,8) left,right
    abs_np = abs_action.cpu().numpy()  # (T,2,8)
    left_prop = prop_np[0]
    right_prop = prop_np[1]
    left_future = abs_np[:, 0]
    right_future = abs_np[:, 1]

    overlay_pts = [left_prop[:3], right_prop[:3], left_future[:, :3], right_future[:, :3]]
    overlay_pts = np.concatenate(
        [x[None] if x.ndim == 1 else x for x in overlay_pts],
        axis=0
    )
    all_for_limits = np.concatenate([points, overlay_pts], axis=0) if points.size else overlay_pts
    span = np.max(np.ptp(all_for_limits, axis=0)) if all_for_limits.size else 1.0
    arrow_len = max(0.02, 0.06 * float(span))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    front_rgb_path = out_path.with_name(f"{out_path.stem}_front_rgb{out_path.suffix}")

    front_rgb = sample["rgb_head"][frame_idx, 0].permute(1, 2, 0).cpu().numpy()
    plt.imsave(front_rgb_path, front_rgb)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=24, azim=-58)

    if points.shape[0] > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=float(args.point_size),
            alpha=0.85,
            linewidths=0,
        )

    ax.scatter(
        left_prop[0], left_prop[1], left_prop[2],
        c="dodgerblue", s=95, marker="o", label="left proprio"
    )
    ax.scatter(
        right_prop[0], right_prop[1], right_prop[2],
        c="orangered", s=95, marker="o", label="right proprio"
    )
    ax.plot(
        left_future[:, 0], left_future[:, 1], left_future[:, 2],
        c="springgreen", linewidth=2.3, label="left future abs(action)"
    )
    ax.scatter(left_future[:, 0], left_future[:, 1], left_future[:, 2], c="springgreen", s=24, marker="o")
    ax.plot(
        right_future[:, 0], right_future[:, 1], right_future[:, 2],
        c="violet", linewidth=2.3, label="right future abs(action)"
    )
    ax.scatter(right_future[:, 0], right_future[:, 1], right_future[:, 2], c="violet", s=24, marker="o")

    left_prop_vec = quat_axis_wxyz(left_prop[3:7], axis=0)
    right_prop_vec = quat_axis_wxyz(right_prop[3:7], axis=0)
    ax.quiver(
        left_prop[0], left_prop[1], left_prop[2],
        left_prop_vec[0], left_prop_vec[1], left_prop_vec[2],
        length=arrow_len, normalize=True, color="dodgerblue", linewidth=2.0
    )
    ax.quiver(
        right_prop[0], right_prop[1], right_prop[2],
        right_prop_vec[0], right_prop_vec[1], right_prop_vec[2],
        length=arrow_len, normalize=True, color="orangered", linewidth=2.0
    )

    for step in range(left_future.shape[0]):
        lv = quat_axis_wxyz(left_future[step, 3:7], axis=0)
        rv = quat_axis_wxyz(right_future[step, 3:7], axis=0)
        ax.quiver(
            left_future[step, 0], left_future[step, 1], left_future[step, 2],
            lv[0], lv[1], lv[2],
            length=arrow_len, normalize=True, color="springgreen", alpha=0.65, linewidth=1.1
        )
        ax.quiver(
            right_future[step, 0], right_future[step, 1], right_future[step, 2],
            rv[0], rv[1], rv[2],
            length=arrow_len, normalize=True, color="violet", alpha=0.65, linewidth=1.1
        )

    set_equal_axes(ax, all_for_limits)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"{instr}\nval_chunk={chunk_idx}/{val_len - 1}, frame_in_chunk={frame_idx}, "
        f"point_stride={args.point_stride}"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print(f"[DONE] Saved visualization to: {out_path}")
    print(f"[DONE] Saved front RGB to: {front_rgb_path}")
    print(f"[INFO] instruction: {instr}")
    print(f"[INFO] val_chunk_index={chunk_idx}, frame_index={frame_idx}, chunk_size={c}")
    print("[INFO] fixed normalization stats (same as RoboTwin2 trainer):")
    print(f"       xyz_min={FIXED_XYZ_MIN.tolist()}")
    print(f"       xyz_max={FIXED_XYZ_MAX.tolist()}")
    print(f"       openness_min={float(FIXED_OPENNESS_MIN):.8f}")
    print(f"       openness_max={float(FIXED_OPENNESS_MAX):.8f}")
    print(f"[RAW rel_action] shape={tuple(rel_action.shape)}")
    print(rel_action.cpu().numpy())
    print(f"[NORMALIZED rel_action] shape={tuple(rel_action_norm.shape)}")
    print(rel_action_norm.cpu().numpy())


if __name__ == "__main__":
    main()
