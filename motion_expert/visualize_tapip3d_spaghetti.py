#!/usr/bin/env python3
"""TAPIP3D-style point-cloud spaghetti visualization of the 3D MotionExpert.

For each benchmark (calvin / libero / robotwin2) we sample rollouts from the
*ordered* training zarr, run a trained MotionExpert checkpoint, and render the
predicted point trajectories ("spaghetti") on top of a photo of the colored
query point cloud -- mirroring the look of TAPIP3D's rerun visualizer
(https://github.com/zbw001/TAPIP3D/blob/main/utils/rerun_visualizer.py):
the cloud is colored by the source RGB and the trajectories use a `rainbow`
colormap normalized by a spatial coordinate.

To represent the manipulation *process*, each rollout uses 3 motion predictions
sampled at 3 timesteps along the same episode; their spaghettis are overlaid.

Outputs per rollout:
  pred/<ds>_rollout_XXX.png  -- point cloud + PREDICTION spaghetti (the "fancier" one)
  gt/<ds>_rollout_XXX.png    -- point cloud + GROUND-TRUTH spaghetti
  video/<ds>_rollout_XXX.mp4 -- rollout video (spaghetti grows over the process,
                                with a slow orbit), prediction by default.

Note on data: the calvin/libero/robotwin2 *motion* zarrs only store the 20x20
query points in 3D (query_world_xyz) -- there is no dense depth -- so the point
cloud defaults to the real colored query points rather than an interpolated
surface. The old interpolated surface can still be enabled with --dense.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_expert.visualize_calvin_pointworld import (  # noqa: E402
    _build_motion_expert,
    _make_motion_batch,
)
from motion_expert.zarr_dataset import InstructionTable, MotionExpertQwenTokenizer  # noqa: E402

GROUP_ZARR = Path("/data/group_data/katefgroup/VLA/zarr_datasets")

# Per-dataset defaults. The *ordered* (non-rechunked) train.zarr is used so that
# consecutive rows form a temporal rollout; the rechunked8 zarrs are shuffled.
DATASETS: Dict[str, Dict[str, str]] = {
    "calvin": {
        "zarr": str(GROUP_ZARR / "calvin_motion_zarr/train.zarr"),
        "config": str(REPO_ROOT / "motion_expert/configs/calvin_florence2.yaml"),
        "checkpoint": str(REPO_ROOT / "train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt"),
        "instructions": str(REPO_ROOT / "instructions/calvin/train_motion_instructions_full.json"),
    },
    "libero": {
        "zarr": str(GROUP_ZARR / "libero_motion_zarr/train.zarr"),
        "config": str(REPO_ROOT / "motion_expert/configs/libero_florence2.yaml"),
        "checkpoint": str(REPO_ROOT / "train_logs/motion_expert/libero_florence2/B32_18k.pt"),
        "instructions": str(REPO_ROOT / "instructions/libero/motion_instructions_full.json"),
    },
    "robotwin2": {
        "zarr": str(GROUP_ZARR / "robotwin2_motion_zarr/robotwin2_motion_zarr/train_rechunked8.zarr"),
        "config": str(REPO_ROOT / "motion_expert/configs/robotwin2_florence2.yaml"),
        "checkpoint": str(REPO_ROOT / "train_logs/motion_expert/robotwin2_florence2/checkpoint_step_20000.pt"),
        "instructions": str(REPO_ROOT / "instructions/robotwin2/instructions_full.json"),
    },
}

BG_COLOR = "#ffffff"
TITLE_COLOR = "#111111"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--zarr", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--instructions", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: /data/user_data/ziyinx/vis_outputs/motion_tapip3d/<dataset>")
    p.add_argument("--num-rollouts", type=int, default=20)
    p.add_argument("--frames-per-rollout", type=int, default=3,
                   help="Number of motion predictions (timesteps) per rollout.")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidate-pool", type=int, default=160,
                   help="Episodes to score before keeping the highest-motion --num-rollouts.")
    p.add_argument("--min-episode-len", type=int, default=6)
    p.add_argument("--calvin-block", type=int, default=24,
                   help="Pseudo-episode block size for datasets without frame_id_in_episode (calvin).")
    p.add_argument("--frame-fractions", type=str, default="0.12,0.5,0.85",
                   help="Where along the episode to take the timesteps.")
    p.add_argument("--motion-threshold", type=float, default=0.01,
                   help="Min trajectory displacement (m) for a track to be drawn.")
    p.add_argument("--max-points-per-chunk", type=int, default=140)
    p.add_argument("--random-select", action="store_true",
                   help="Pick rollouts at random instead of by GT motion magnitude.")
    # rendering
    p.add_argument("--color-by", choices=("y", "x", "z", "time", "chunk"), default="y",
                   help="Spaghetti color source. 'y' matches TAPIP3D (rainbow by world-y).")
    p.add_argument("--cmap", type=str, default="rainbow")
    p.add_argument("--cloud-from", choices=("first", "all"), default="all",
                   help="Render the point cloud of the first timestep or the union of all sampled timesteps.")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-72.0)
    p.add_argument("--zoom", type=float, default=1.8, help="3D camera zoom; >1 fills the frame.")
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--cloud-alpha", type=float, default=0.9)
    p.add_argument("--line-width", type=float, default=1.6)
    p.add_argument("--line-alpha", type=float, default=0.85)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--no-title", action="store_true")
    # GT-only / point-cloud mode
    p.add_argument("--pred", action="store_true",
                   help="Also run the motion expert and render predictions (default: GT only).")
    p.add_argument("--dense", dest="dense", action="store_true", default=False,
                   help="Interpolate the GxG query grid into a pseudo-dense surface (off by default).")
    p.add_argument("--no-dense", dest="dense", action="store_false",
                   help="Use the real sparse query points as the point cloud (default).")
    p.add_argument("--grid-size", type=int, default=20, help="Query grid side (400 points -> 20).")
    p.add_argument("--upsample", type=int, default=10,
                   help="Dense-cloud upsample factor when --dense is used (10 -> ~191x191).")
    p.add_argument("--edge-thresh-mult", type=float, default=3.0,
                   help="Drop dense cells whose 3D corner range exceeds this x the median (depth-edge masking).")
    # video
    p.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    p.add_argument("--no-video", dest="save_video", action="store_false")
    p.add_argument("--gt-video", action="store_true", help="Also render a GT rollout video.")
    p.add_argument("--video-fps", type=int, default=12)
    p.add_argument("--reveal-frames-per-chunk", type=int, default=8)
    p.add_argument("--video-hold-frames", type=int, default=6)
    p.add_argument("--video-orbit-deg", type=float, default=40.0)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _device(arg: Optional[str]) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Rollout / episode selection
# --------------------------------------------------------------------------- #
def build_episodes(z, calvin_block: int) -> List[np.ndarray]:
    """Return a list of row-index arrays, one per episode (ordered)."""
    n = int(z["instr_id"].shape[0])
    keys = set(z.array_keys())
    if "frame_id_in_episode" in keys:
        fid = np.asarray(z["frame_id_in_episode"][:]).astype(np.int64)
        # Episode starts where frame id resets (does not strictly increase).
        starts = [0] + [i for i in range(1, n) if fid[i] <= fid[i - 1]]
        bounds = starts + [n]
        return [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]
    # No episode metadata (calvin): split the ordered stream into fixed blocks.
    block = max(int(calvin_block), 3)
    return [np.arange(b, min(b + block, n)) for b in range(0, n - block + 1, block)]


def pick_frames(ep_rows: np.ndarray, fractions: Sequence[float]) -> Optional[np.ndarray]:
    length = int(ep_rows.shape[0])
    if length < len(fractions):
        return None
    pos = sorted({int(round(f * (length - 1))) for f in fractions})
    # Spread out if rounding collapsed positions.
    if len(pos) < len(fractions):
        pos = list(np.linspace(0, length - 1, len(fractions)).round().astype(int))
        pos = sorted(set(pos))
        if len(pos) < len(fractions):
            return None
    return ep_rows[np.asarray(pos, dtype=np.int64)]


def gt_motion_score(z, rows: np.ndarray) -> float:
    """Max valid trajectory displacement across the selected frames."""
    best = 0.0
    for r in rows.tolist():
        traj = np.asarray(z["traj_3d_xyz"][r], dtype=np.float32)  # (P,T,3) stepwise delta
        mask = np.asarray(z["valid_mask"][r], dtype=bool)  # (P,T)
        traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        traj = np.where(mask[..., None], traj, 0.0)
        disp = np.linalg.norm(np.cumsum(traj, axis=1), axis=-1)  # (P,T)
        valid_pts = mask.any(axis=1)
        if valid_pts.any():
            best = max(best, float(disp[valid_pts].max()))
    return best


def select_rollouts(z, args: argparse.Namespace, fractions: Sequence[float]) -> List[np.ndarray]:
    rng = np.random.default_rng(int(args.seed))
    episodes = [ep for ep in build_episodes(z, args.calvin_block) if ep.shape[0] >= args.min_episode_len]
    if not episodes:
        raise RuntimeError("No episodes of sufficient length found.")
    order = rng.permutation(len(episodes))
    pool = min(int(args.candidate_pool), len(episodes))
    candidates: List[Tuple[float, np.ndarray]] = []
    for idx in order[:pool]:
        rows = pick_frames(episodes[int(idx)], fractions)
        if rows is None:
            continue
        score = 0.0 if args.random_select else gt_motion_score(z, rows)
        candidates.append((score, rows))
    if not candidates:
        raise RuntimeError("No valid rollout candidates.")
    if args.random_select:
        rng.shuffle(candidates)
        chosen = candidates[: args.num_rollouts]
    else:
        candidates.sort(key=lambda t: t[0], reverse=True)
        chosen = candidates[: args.num_rollouts]
    return [rows for _, rows in chosen]


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def color_from_uv(rgb_chw: np.ndarray, query_uv: np.ndarray) -> np.ndarray:
    """Sample per-query-point colors (P,3) in [0,1] from the source RGB."""
    _, h, w = rgb_chw.shape
    u = np.clip(np.round(query_uv[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(query_uv[:, 1]).astype(np.int64), 0, h - 1)
    return rgb_chw[:, v, u].transpose(1, 0).astype(np.float32) / 255.0


@torch.no_grad()
def load_chunk(z, row: int, *, instr_table, gt_only: bool = True,
               pipeline=None, model_cfg=None, tokenizer=None, device=None,
               num_inference_steps: int = 10) -> Dict[str, np.ndarray]:
    rgb_chw = np.asarray(z["rgb"][row], dtype=np.uint8)  # (3,H,W)
    query_uv = np.asarray(z["query_uv"][row], dtype=np.float32)  # (P,2)
    query_xyz = np.asarray(z["query_world_xyz"][row], dtype=np.float32)  # (P,3)
    gt_delta = np.asarray(z["traj_3d_xyz"][row], dtype=np.float32)  # (P,T,3) stepwise delta
    mask = np.asarray(z["valid_mask"][row], dtype=bool)  # (P,T)
    instr_id = int(np.asarray(z["instr_id"][row]).reshape(-1)[0])
    instruction = instr_table.resolve(instr_id)

    gt_delta = np.nan_to_num(gt_delta, nan=0.0, posinf=0.0, neginf=0.0)
    gt_delta_masked = np.where(mask[..., None], gt_delta, 0.0)
    valid_pts = np.isfinite(query_xyz).all(axis=1) & (np.linalg.norm(query_xyz, axis=1) > 1e-6)
    gt_abs = query_xyz[:, None, :] + np.cumsum(gt_delta_masked, axis=1)  # (P,T,3)

    out = {
        "row": np.int64(row),
        "query_xyz": query_xyz,
        "query_uv": query_uv,
        "rgb_chw": rgb_chw,
        "colors": color_from_uv(rgb_chw, query_uv),
        "valid_pts": valid_pts,
        "valid_mask": mask,
        "gt_abs": gt_abs.astype(np.float32),
        "gt_disp": np.linalg.norm(np.cumsum(gt_delta_masked, axis=1), axis=-1).astype(np.float32),
        "instruction": instruction,
    }
    if not gt_only:
        batch = _make_motion_batch(
            model_cfg=model_cfg, tokenizer=tokenizer,
            rgb=rgb_chw.transpose(1, 2, 0), instruction=instruction,
            query_uv_source=query_uv, query_world_xyz=query_xyz,
        ).to(device)
        pred_delta = pipeline.predict(
            batch, num_inference_steps=int(num_inference_steps)
        )["pred_delta_xyz"].detach().cpu().float().numpy()[0]
        out["pred_abs"] = (query_xyz[:, None, :] + np.cumsum(pred_delta, axis=1)).astype(np.float32)
        out["pred_disp"] = np.linalg.norm(np.cumsum(pred_delta, axis=1), axis=-1).astype(np.float32)
    return out


def select_moving(chunk: Dict[str, np.ndarray], source: str, motion_threshold: float,
                  max_points: int) -> np.ndarray:
    valid_pts = chunk["valid_pts"]
    if source == "gt":
        moved = chunk["valid_mask"].any(axis=1) & valid_pts
        disp = chunk["gt_disp"].max(axis=1)
    else:
        moved = valid_pts
        disp = chunk["pred_disp"].max(axis=1)
    sel = moved & (disp > float(motion_threshold))
    idx = np.where(sel)[0]
    if idx.size == 0:
        idx = np.where(moved)[0]
    if idx.size > max_points:
        idx = idx[np.argsort(disp[idx])[::-1][:max_points]]
    return idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _color_axis_index(color_by: str) -> int:
    return {"x": 0, "y": 1, "z": 2}.get(color_by, 1)


def build_tracks(chunks: Sequence[Dict[str, np.ndarray]], source: str, args: argparse.Namespace,
                 reveal: Optional[Sequence[float]] = None):
    """Return (polylines, rgba_colors, all_track_points)."""
    cmap = matplotlib.colormaps[args.cmap]
    abs_key = "gt_abs" if source == "gt" else "pred_abs"

    polylines: List[np.ndarray] = []
    scalar_vals: List[float] = []  # for colormap normalization
    chunk_ids: List[int] = []
    time_vals: List[float] = []
    pts_accum: List[np.ndarray] = []

    for ci, chunk in enumerate(chunks):
        idx = select_moving(chunk, source, args.motion_threshold, args.max_points_per_chunk)
        abs_traj = chunk[abs_key]
        anchor = chunk["query_xyz"]
        n_steps = int(abs_traj.shape[1])
        frac = 1.0 if reveal is None else float(reveal[ci])
        # Reveal the first `show` steps; freeze the rest at the last revealed point
        # so every polyline has exactly n_steps+1 points (homogeneous -> a single
        # Line3DCollection; matplotlib's 3D autoscale chokes on ragged segments).
        show = int(round(frac * n_steps))
        show = max(0, min(show, n_steps))
        src_idx = np.minimum(np.arange(n_steps), max(show - 1, 0))
        for p in idx.tolist():
            anchor_p = anchor[p]
            if show <= 0:
                revealed = np.repeat(anchor_p[None, :], n_steps, axis=0)
            else:
                revealed = abs_traj[p][src_idx]
            poly = np.concatenate([anchor_p[None, :], revealed], axis=0)  # (n_steps+1, 3)
            if not np.isfinite(poly).all():
                continue
            polylines.append(poly)
            scalar_vals.append(float(anchor_p[_color_axis_index(args.color_by)]))
            chunk_ids.append(ci)
            time_vals.append(ci)
            pts_accum.append(poly)

    if not polylines:
        return [], np.zeros((0, 4)), np.zeros((0, 3))

    if args.color_by == "chunk":
        denom = max(len(chunks) - 1, 1)
        norm_vals = np.asarray(chunk_ids, dtype=np.float32) / denom
        rgba = cmap(norm_vals)
    elif args.color_by == "time":
        denom = max(len(chunks) - 1, 1)
        norm_vals = np.asarray(time_vals, dtype=np.float32) / denom
        rgba = cmap(norm_vals)
    else:
        sv = np.asarray(scalar_vals, dtype=np.float32)
        norm = matplotlib.colors.Normalize(vmin=float(sv.min()), vmax=float(sv.max()))
        rgba = cmap(norm(sv))
    rgba = np.asarray(rgba)
    rgba[:, 3] = float(args.line_alpha)
    all_pts = np.concatenate(pts_accum, axis=0)
    return polylines, rgba, all_pts


def interpolate_dense_cloud(chunk: Dict[str, np.ndarray], grid_size: int, upsample: int,
                            edge_thresh_mult: float):
    """Optionally densify the GxG query grid into a colored pseudo-cloud.

    The 400 query points form a regular GxG grid of real 3D world positions; bilinearly
    upsampling that grid yields a dense surface. Cells spanning a large depth discontinuity
    (object/background edges) and cells touching invalid points are dropped to avoid webbing.
    Colors are sampled from the full-resolution RGB at the upsampled pixel coordinates.
    """
    import torch.nn.functional as F

    g = int(grid_size)
    up = max(1, int(upsample))
    qxyz = chunk["query_xyz"].reshape(g, g, 3)
    quv = chunk["query_uv"].reshape(g, g, 2)
    valid = chunk["valid_pts"].reshape(g, g)
    rgb_chw = chunk["rgb_chw"]  # (3,H,W)
    _, H, W = rgb_chw.shape
    hf = (g - 1) * up + 1

    def _up(arr_hwc: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(arr_hwc.transpose(2, 0, 1)[None]).astype(np.float32))
        t = F.interpolate(t, size=(hf, hf), mode="bilinear", align_corners=True)
        return t[0].permute(1, 2, 0).numpy()

    xyz_f = _up(qxyz)            # (hf,hf,3)
    uv_f = _up(quv)             # (hf,hf,2)
    vmask_f = _up(valid.astype(np.float32)[..., None])[..., 0]  # (hf,hf)

    # Per-cell corner 3D range; mark depth-edge / invalid cells.
    cell_range = np.full((g - 1, g - 1), np.inf, dtype=np.float32)
    for a in range(g - 1):
        for b in range(g - 1):
            cv = valid[a:a + 2, b:b + 2]
            if not cv.all():
                continue
            corners = qxyz[a:a + 2, b:b + 2].reshape(4, 3)
            cell_range[a, b] = float(np.linalg.norm(corners.max(0) - corners.min(0)))
    finite = cell_range[np.isfinite(cell_range)]
    med = float(np.median(finite)) if finite.size else 0.0
    thresh = max(med * float(edge_thresh_mult), 1e-6)
    cell_bad = ~np.isfinite(cell_range) | (cell_range > thresh)

    ii, jj = np.mgrid[0:hf, 0:hf]
    cell_i = np.minimum(ii // up, g - 2)
    cell_j = np.minimum(jj // up, g - 2)
    keep = (~cell_bad[cell_i, cell_j]) & (vmask_f > 0.999)

    xyz_keep = xyz_f[keep]
    uv_keep = uv_f[keep]
    u = np.clip(np.round(uv_keep[:, 0]).astype(np.int64), 0, W - 1)
    v = np.clip(np.round(uv_keep[:, 1]).astype(np.int64), 0, H - 1)
    colors = rgb_chw[:, v, u].transpose(1, 0).astype(np.float32) / 255.0
    return xyz_keep.astype(np.float32), colors


def cloud_points(chunks: Sequence[Dict[str, np.ndarray]], args: argparse.Namespace):
    use = chunks[:1] if args.cloud_from == "first" else chunks
    xyz, rgb = [], []
    for chunk in use:
        if getattr(args, "dense", False):
            cx, cc = interpolate_dense_cloud(chunk, args.grid_size, args.upsample, args.edge_thresh_mult)
        else:
            v = chunk["valid_pts"]
            cx, cc = chunk["query_xyz"][v], chunk["colors"][v]
        xyz.append(cx)
        rgb.append(cc)
    return np.concatenate(xyz, axis=0), np.concatenate(rgb, axis=0)


def _equal_aspect(ax, pts: np.ndarray, zoom: float = 1.0, pct: float = 2.0) -> None:
    if pts.size == 0:
        return
    # Robust extent: ignore a few outlier points that would otherwise shrink the scene.
    mins = np.percentile(pts, pct, axis=0)
    maxs = np.percentile(pts, 100.0 - pct, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.6 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=float(zoom))
    except TypeError:  # older matplotlib without the zoom kwarg
        ax.set_box_aspect((1, 1, 1))


def _style_axes(ax) -> None:
    ax.set_axis_off()
    ax.set_facecolor(BG_COLOR)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))


def render_static(chunks, source: str, out_path: Path, args: argparse.Namespace, title: str) -> None:
    cloud_xyz, cloud_rgb = cloud_points(chunks, args)
    polylines, rgba, track_pts = build_tracks(chunks, source, args)

    fig = plt.figure(figsize=(8.0, 8.0), dpi=int(args.dpi))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax)

    if cloud_xyz.size:
        ax.scatter(cloud_xyz[:, 0], cloud_xyz[:, 1], cloud_xyz[:, 2],
                   c=np.clip(cloud_rgb, 0, 1), s=float(args.point_size),
                   alpha=float(args.cloud_alpha), depthshade=False, linewidths=0)
    if polylines:
        lc = Line3DCollection(polylines, colors=rgba, linewidths=float(args.line_width))
        ax.add_collection3d(lc)

    bound = cloud_xyz if track_pts.size == 0 else (
        np.concatenate([cloud_xyz, track_pts], axis=0) if cloud_xyz.size else track_pts)
    _equal_aspect(ax, bound, zoom=float(args.zoom))
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    if title and not args.no_title:
        ax.set_title(title, color=TITLE_COLOR, fontsize=10, pad=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_video(chunks, source: str, out_path: Path, args: argparse.Namespace, title: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[viz] ffmpeg not found; skipping video.")
        return False

    cloud_xyz, cloud_rgb = cloud_points(chunks, args)
    # Full-extent bounds (reveal everything) so the camera box is stable.
    _, _, full_pts = build_tracks(chunks, source, args, reveal=[1.0] * len(chunks))
    bound = full_pts if cloud_xyz.size == 0 else (
        np.concatenate([cloud_xyz, full_pts], axis=0) if full_pts.size else cloud_xyz)

    rpc = max(1, int(args.reveal_frames_per_chunk))
    n_reveal = len(chunks) * rpc
    n_total = n_reveal + max(0, int(args.video_hold_frames))

    fig = plt.figure(figsize=(7.6, 7.6), dpi=int(args.dpi))
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    tmp_dir = Path(tempfile.mkdtemp(prefix="motion_tapip3d_vid_"))
    try:
        for f in range(n_total):
            ax.cla()
            _style_axes(ax)
            fr = min(f, n_reveal)
            reveal = []
            for ci in range(len(chunks)):
                lo, hi = ci * rpc, (ci + 1) * rpc
                if fr >= hi:
                    reveal.append(1.0)
                elif fr <= lo:
                    reveal.append(0.0)
                else:
                    reveal.append((fr - lo) / float(rpc))
            polylines, rgba, _ = build_tracks(chunks, source, args, reveal=reveal)
            if cloud_xyz.size:
                ax.scatter(cloud_xyz[:, 0], cloud_xyz[:, 1], cloud_xyz[:, 2],
                           c=np.clip(cloud_rgb, 0, 1), s=float(args.point_size),
                           alpha=float(args.cloud_alpha), depthshade=False, linewidths=0)
            if polylines:
                ax.add_collection3d(Line3DCollection(polylines, colors=rgba, linewidths=float(args.line_width)))
            _equal_aspect(ax, bound, zoom=float(args.zoom))
            azim = float(args.azim) + float(args.video_orbit_deg) * (f / max(n_total - 1, 1))
            ax.view_init(elev=float(args.elev), azim=azim)
            if title and not args.no_title:
                ax.set_title(title, color=TITLE_COLOR, fontsize=10, pad=2)
            fig.savefig(tmp_dir / f"f_{f:04d}.png", facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-y", "-framerate", str(max(1, int(args.video_fps))),
               "-i", str(tmp_dir / "f_%04d.png"),
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
               "-vcodec", "libx264", str(out_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            print(f"[viz] ffmpeg failed for {out_path.name}:\n{proc.stderr[-800:]}")
            return False
        return True
    finally:
        plt.close(fig)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    snap = "/home/ziyinx/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594"
    if os.path.isdir(snap):
        os.environ.setdefault("FLORENCE2_MODEL_PATH", snap)
    args = parse_args()
    cfg = DATASETS[args.dataset]
    zarr_path = Path(args.zarr or cfg["zarr"])
    config_path = Path(args.config or cfg["config"])
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"])
    instructions_path = Path(args.instructions or cfg["instructions"])
    out_dir = Path(args.output_dir or f"/data/user_data/ziyinx/vis_outputs/motion_tapip3d/{args.dataset}")
    fractions = [float(x) for x in args.frame_fractions.split(",") if x.strip()]
    if len(fractions) != int(args.frames_per_rollout):
        fractions = list(np.linspace(0.12, 0.85, int(args.frames_per_rollout)))

    gt_only = not bool(args.pred)
    device = _device(args.device)
    print(f"[viz] dataset={args.dataset} device={device} gt_only={gt_only} dense={args.dense}")
    print(f"[viz] zarr={zarr_path}")

    z = zarr.open(str(zarr_path), mode="r")
    instr_table = InstructionTable(str(instructions_path))

    rollouts = select_rollouts(z, args, fractions)
    print(f"[viz] selected {len(rollouts)} rollouts ({args.frames_per_rollout} timesteps each)")

    pipeline = model_cfg = tokenizer = None
    if not gt_only:
        print(f"[viz] checkpoint={checkpoint_path}")
        _, model_cfg, pipeline = _build_motion_expert(config_path, checkpoint_path, device)
        backbone_type = str(model_cfg.backbone_type).lower()
        tokenizer = MotionExpertQwenTokenizer(
            backbone_type=backbone_type,
            model_name_or_path=(model_cfg.qwen.model_name_or_path if backbone_type == "qwen"
                                else model_cfg.florence.model_name_or_path),
            max_length=int(model_cfg.text_max_length),
        )

    for ri, rows in enumerate(rollouts):
        chunks = [
            load_chunk(z, int(r), instr_table=instr_table, gt_only=gt_only,
                       pipeline=pipeline, model_cfg=model_cfg, tokenizer=tokenizer, device=device,
                       num_inference_steps=args.num_inference_steps)
            for r in rows.tolist()
        ]
        instruction = chunks[0]["instruction"]
        title = f"{args.dataset} | rows={rows.tolist()} | {instruction}"
        stem = f"{args.dataset}_rollout_{ri:03d}"
        render_static(chunks, "gt", out_dir / "gt" / f"{stem}.png", args, title)
        if args.save_video:
            render_video(chunks, "gt", out_dir / "video" / f"{stem}.mp4", args, title)
        if not gt_only:
            render_static(chunks, "pred", out_dir / "pred" / f"{stem}.png", args, title)
            if args.save_video:
                render_video(chunks, "pred", out_dir / "video_pred" / f"{stem}.mp4", args, title)
        print(f"[viz] rollout {ri + 1}/{len(rollouts)} done  rows={rows.tolist()}  task='{instruction[:60]}'")

    print(f"[viz] all done -> {out_dir}")


if __name__ == "__main__":
    main()
