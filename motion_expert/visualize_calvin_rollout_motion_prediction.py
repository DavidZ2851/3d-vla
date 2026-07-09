#!/usr/bin/env python3
"""Predict and render MotionExpert trajectories from a saved CALVIN rollout folder.

The input folder is expected to contain the RGB/depth videos produced by
``online_evaluation_calvin/rollout_task_video.py``:

  seqXXXX_subY_<task>_static_rgb.mp4
  seqXXXX_subY_<task>_static_depth.mp4
  seqXXXX_subY_<task>_static_summary.json

The depth MP4 is a visualization video. If a raw depth NPZ exists in the summary
it is used; otherwise the script reconstructs metric depth from the summary's
``depth_min`` / ``depth_max`` values.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROLLOUT_DIR = (
    REPO_ROOT / "vis_outputs/calvin_task_rollouts/push_red_block_right"
)
DEFAULT_CONFIG = REPO_ROOT / "motion_expert/configs/calvin_florence2_viz.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt"
)

# Same static-camera parameters as utils/depth2cloud/calvin.py. That helper uses
# the OpenGL view matrix directly; MotionExpert's unprojection utilities use an
# OpenCV-style +Z camera convention, so we apply the same frame fix used by the
# existing PointWorld/TAPIP3D visualization code.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--rgb-video", type=Path, default=None)
    p.add_argument("--depth-video", type=Path, default=None)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--instruction", type=str, default="")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--pcd-stride", type=int, default=1)
    p.add_argument("--pcd-max-points", type=int, default=60000)
    p.add_argument("--motion-threshold", type=float, default=0.005)
    p.add_argument("--max-moving-points", type=int, default=100)
    p.add_argument("--scene-alpha", type=float, default=0.92)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--anchor-size", type=float, default=18.0)
    p.add_argument("--line-width", type=float, default=1.6)
    p.add_argument("--line-alpha", type=float, default=0.95)
    p.add_argument("--crop-padding", type=float, default=0.08)
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-68.0)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_existing(path: Optional[Path], base: Path) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if str(path).strip() == "":
        return None
    if path.exists():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _load_summary(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    summary_path = args.summary
    if summary_path is None:
        matches = sorted(args.rollout_dir.glob("*_summary.json"))
        if not matches:
            raise FileNotFoundError(f"No *_summary.json found in {args.rollout_dir}")
        summary_path = matches[0]
    summary_path = _resolve_existing(summary_path, args.rollout_dir)
    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return summary_path, json.load(f)


def _resolve_raw_depth_npz(summary: Dict[str, Any], rollout_dir: Path) -> Optional[Path]:
    raw_npz = summary.get("raw_depth_npz")
    if not raw_npz:
        return None
    raw_npz_path = _resolve_existing(Path(raw_npz), rollout_dir)
    if raw_npz_path is not None and raw_npz_path.exists():
        return raw_npz_path
    return None


def _resolve_rollout_files(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path], Dict[str, Any]]:
    summary_path, summary = _load_summary(args)
    rgb_video = _resolve_existing(args.rgb_video, args.rollout_dir)
    depth_video = _resolve_existing(args.depth_video, args.rollout_dir)
    if rgb_video is None:
        rgb_video = _resolve_existing(Path(summary.get("rgb_video", "")), args.rollout_dir)
    if depth_video is None:
        depth_video = _resolve_existing(Path(summary.get("depth_video", "")), args.rollout_dir)
    if rgb_video is None or not rgb_video.exists():
        matches = sorted(args.rollout_dir.glob("*_rgb.mp4"))
        if not matches:
            raise FileNotFoundError(f"No RGB video found in {args.rollout_dir}")
        rgb_video = matches[0]
    raw_depth_npz = _resolve_raw_depth_npz(summary, args.rollout_dir)
    if raw_depth_npz is None and (depth_video is None or not depth_video.exists()):
        matches = sorted(args.rollout_dir.glob("*_depth.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"No raw depth NPZ or depth video found in {args.rollout_dir}. "
                "Expected summary.raw_depth_npz to point to an existing .npz, "
                "or a *_depth.mp4 fallback."
            )
        depth_video = matches[0]
    return summary_path, rgb_video, depth_video, summary


def _read_rgb_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"RGB video has no frames: {path}")
    return np.asarray(frames, dtype=np.uint8)


def _read_depth_video_scaled(path: Path, depth_min: float, depth_max: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open depth video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(depth_min + gray * (depth_max - depth_min))
    cap.release()
    if not frames:
        raise RuntimeError(f"Depth video has no frames: {path}")
    return np.asarray(frames, dtype=np.float32)


def _load_depth(summary: Dict[str, Any], depth_video: Optional[Path], rollout_dir: Path) -> np.ndarray:
    raw_npz_path = _resolve_raw_depth_npz(summary, rollout_dir)
    if raw_npz_path is not None and raw_npz_path.exists():
        print(f"[rollout-motion-viz] loading raw metric depth: {raw_npz_path}")
        with np.load(raw_npz_path) as data:
            if "depth" not in data:
                raise KeyError(f"{raw_npz_path} does not contain key `depth`.")
            return np.asarray(data["depth"], dtype=np.float32)

    if depth_video is None or not depth_video.exists():
        raise FileNotFoundError(
            "Metric depth needs raw_depth_npz, or a depth video fallback with "
            "summary depth_min/depth_max. No usable depth video was found."
        )
    if "depth_min" not in summary or "depth_max" not in summary:
        raise KeyError(
            "Metric depth reconstruction needs raw_depth_npz or summary depth_min/depth_max."
        )
    print(
        "[rollout-motion-viz] raw depth NPZ not found; reconstructing metric "
        "depth from depth MP4 using summary depth_min/depth_max."
    )
    return _read_depth_video_scaled(
        depth_video,
        float(summary["depth_min"]),
        float(summary["depth_max"]),
    )


def _calvin_front_intrinsics_for_size(height: int, width: int) -> np.ndarray:
    focal_200 = float(200.0 / (2.0 * np.tan(np.deg2rad(10.0) / 2.0)))
    intr = np.array(
        [[focal_200, 0.0, 100.0], [0.0, focal_200, 100.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intr[0, :] *= float(width) / 200.0
    intr[1, :] *= float(height) / 200.0
    return intr


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_motion_predictor(args: argparse.Namespace, device: torch.device):
    from motion_expert.visualize_calvin_pointworld import _build_motion_expert
    from motion_expert.zarr_dataset import MotionExpertQwenTokenizer

    raw_cfg, model_cfg, pipeline = _build_motion_expert(args.config, args.checkpoint, device)
    backbone_type = str(model_cfg.backbone_type).lower()
    tokenizer_model_path = (
        model_cfg.qwen.model_name_or_path
        if backbone_type == "qwen"
        else model_cfg.florence.model_name_or_path
    )
    tokenizer = MotionExpertQwenTokenizer(
        backbone_type=backbone_type,
        model_name_or_path=tokenizer_model_path,
        max_length=int(model_cfg.text_max_length),
    )
    inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(raw_cfg.get("train", {}).get("val_num_inference_steps", 10))
    )
    return model_cfg, pipeline, tokenizer, inference_steps


@torch.no_grad()
def _predict(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    instruction: str,
    args: argparse.Namespace,
    device: torch.device,
):
    from motion_expert.visualize_calvin_pointworld import (
        _make_motion_batch,
        _query_world_xyz_from_rgbd,
    )

    height, width = depth.shape
    intr = _calvin_front_intrinsics_for_size(height, width)
    ext = CALVIN_FRONT_CAMERA_W2C
    query_uv, query_xyz, query_valid = _query_world_xyz_from_rgbd(
        depth,
        intr,
        ext,
        int(args.grid_size),
    )
    model_cfg, pipeline, tokenizer, inference_steps = _load_motion_predictor(args, device)
    batch = _make_motion_batch(
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        rgb=rgb,
        instruction=instruction,
        query_uv_source=query_uv,
        query_world_xyz=query_xyz,
    ).to(device)
    outputs = pipeline.predict(batch, num_inference_steps=inference_steps)
    pred_delta = outputs["pred_delta_xyz"].detach().cpu().float().numpy()[0]
    return {
        "intrinsics": intr,
        "extrinsics_w2c": ext,
        "query_uv": query_uv,
        "query_world_xyz": query_xyz,
        "query_valid": query_valid,
        "pred_delta_xyz": pred_delta,
        "inference_steps": inference_steps,
    }


def _future_abs_from_delta(anchor_xyz: np.ndarray, delta_xyz: np.ndarray) -> np.ndarray:
    return anchor_xyz[:, None, :] + np.cumsum(delta_xyz, axis=1)


def _select_motion_points(
    query_xyz: np.ndarray,
    pred_delta: np.ndarray,
    query_valid: np.ndarray,
    *,
    motion_threshold: float,
    max_points: int,
) -> np.ndarray:
    cumulative = np.cumsum(np.nan_to_num(pred_delta, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    disp = np.linalg.norm(cumulative, axis=-1).max(axis=1)
    valid = query_valid & np.isfinite(query_xyz).all(axis=1) & (disp > float(motion_threshold))
    idx = np.where(valid)[0]
    if idx.size == 0:
        idx = np.where(query_valid & np.isfinite(query_xyz).all(axis=1))[0]
    if idx.size > int(max_points):
        order = np.argsort(disp[idx])[::-1][: int(max_points)]
        idx = idx[order]
    return idx.astype(np.int64)


def _axis_limits_for_focus(points: np.ndarray, padding: float) -> Tuple[np.ndarray, float]:
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    if pts.size == 0:
        return np.zeros(3, dtype=np.float32), 0.5
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) + float(padding)
    return center.astype(np.float32), max(radius, 0.05)


def _set_3d_limits(ax, center: np.ndarray, radius: float, zoom: float) -> None:
    r = max(float(radius) / max(float(zoom), 1e-6), 1e-3)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def _build_line_segments(
    query_xyz: np.ndarray,
    pred_abs: np.ndarray,
    keep_idx: np.ndarray,
    hist_count: int,
    colors: np.ndarray,
    line_alpha: float,
) -> Tuple[list[np.ndarray], np.ndarray]:
    segments = []
    rgba = []
    for local_i, point_idx in enumerate(keep_idx.tolist()):
        if hist_count <= 0:
            continue
        poly = np.concatenate(
            [query_xyz[point_idx][None], pred_abs[point_idx, :hist_count]],
            axis=0,
        )
        if not np.isfinite(poly).all():
            continue
        for k in range(poly.shape[0] - 1):
            segments.append(poly[k : k + 2])
            color = np.asarray(colors[local_i], dtype=np.float32).copy()
            color[3] = float(line_alpha)
            rgba.append(color)
    if not rgba:
        return segments, np.zeros((0, 4), dtype=np.float32)
    return segments, np.asarray(rgba, dtype=np.float32)


def _render_frame(
    *,
    fig,
    ax,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    query_xyz: np.ndarray,
    pred_abs: np.ndarray,
    keep_idx: np.ndarray,
    hist_count: int,
    center: np.ndarray,
    radius: float,
    colors: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    from motion_expert.visualize_raw_spatracker_tapip3d import add_frontmost_trajectory_overlay

    for artist in getattr(fig, "_motion_overlay_artists", []):
        try:
            artist.remove()
        except ValueError:
            pass
    fig._motion_overlay_artists = []

    ax.cla()
    ax.set_facecolor("white")
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
    if scene_points.size:
        ax.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            c=np.clip(scene_colors, 0.0, 1.0),
            s=float(args.point_size),
            alpha=float(args.scene_alpha),
            linewidths=0,
            depthshade=False,
        )
    for local_i, point_idx in enumerate(keep_idx.tolist()):
        anchor = query_xyz[point_idx]
        if not np.isfinite(anchor).all():
            continue
        ax.scatter(
            anchor[0:1],
            anchor[1:2],
            anchor[2:3],
            color=colors[local_i : local_i + 1],
            s=float(args.anchor_size),
            edgecolors="black",
            linewidths=0.35,
            depthshade=False,
        )
    _set_3d_limits(ax, center, radius, float(args.zoom))
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.set_box_aspect((1, 1, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    segments, rgba = _build_line_segments(
        query_xyz,
        pred_abs,
        keep_idx,
        int(hist_count),
        colors,
        float(args.line_alpha),
    )
    artist_count = len(fig.artists)
    add_frontmost_trajectory_overlay(fig, ax, segments, rgba, args)
    fig._motion_overlay_artists = list(fig.artists)[artist_count:]
    fig.canvas.draw()
    rgba_img = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba_img[..., :3])


def _render_prediction_video(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    prediction: Dict[str, Any],
    output_video: Path,
    output_spaghetti: Path,
    args: argparse.Namespace,
) -> Tuple[Path, Path, int]:
    from motion_expert.visualize_raw_spatracker_tapip3d import unproject_depth_to_world

    scene_points, scene_colors = unproject_depth_to_world(
        rgb,
        depth,
        prediction["intrinsics"],
        prediction["extrinsics_w2c"],
        width=max(8, rgb.shape[1] // max(1, int(args.pcd_stride))),
        height=max(8, rgb.shape[0] // max(1, int(args.pcd_stride))),
    )
    if scene_points.shape[0] > int(args.pcd_max_points):
        rng = np.random.default_rng(0)
        keep = rng.choice(scene_points.shape[0], int(args.pcd_max_points), replace=False)
        scene_points, scene_colors = scene_points[keep], scene_colors[keep]

    query_xyz = np.asarray(prediction["query_world_xyz"], dtype=np.float32)
    pred_delta = np.asarray(prediction["pred_delta_xyz"], dtype=np.float32)
    query_valid = np.asarray(prediction["query_valid"], dtype=bool)
    keep_idx = _select_motion_points(
        query_xyz,
        pred_delta,
        query_valid,
        motion_threshold=float(args.motion_threshold),
        max_points=int(args.max_moving_points),
    )
    pred_abs = _future_abs_from_delta(query_xyz, np.nan_to_num(pred_delta))
    traj_points = np.concatenate([query_xyz[keep_idx, None, :], pred_abs[keep_idx]], axis=1)
    focus_points = traj_points.reshape(-1, 3)
    if focus_points.size == 0:
        focus_points = query_xyz[query_valid]
    center, radius = _axis_limits_for_focus(focus_points, float(args.crop_padding))

    scene_keep = np.isfinite(scene_points).all(axis=1)
    scene_keep &= np.linalg.norm(scene_points - center[None], axis=1) <= radius * 1.8
    if np.any(scene_keep):
        scene_points_plot = scene_points[scene_keep]
        scene_colors_plot = scene_colors[scene_keep]
    else:
        scene_points_plot = scene_points
        scene_colors_plot = scene_colors

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_spaghetti.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.4, 7.4), dpi=int(args.dpi))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(keep_idx), 1)))
    future_steps = int(pred_delta.shape[1])
    frames = []
    try:
        for hist_count in range(future_steps + 1):
            frame = _render_frame(
                fig=fig,
                ax=ax,
                scene_points=scene_points_plot,
                scene_colors=scene_colors_plot,
                query_xyz=query_xyz,
                pred_abs=pred_abs,
                keep_idx=keep_idx,
                hist_count=hist_count,
                center=center,
                radius=radius,
                colors=colors,
                args=args,
            )
            if hist_count == future_steps:
                plt.imsave(output_spaghetti, frame)
            frames.append(frame)
        _write_rgb_video_ffmpeg(output_video, frames, int(args.fps))
    finally:
        plt.close(fig)
    return output_video, output_spaghetti, int(len(keep_idx))


def _write_rgb_video_ffmpeg(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found in PATH; cannot write MP4.")
    if not frames:
        raise ValueError("No frames were rendered for the prediction video.")

    rgb_frames = [
        np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[..., :3])
        for frame in frames
    ]
    height, width = rgb_frames[0].shape[:2]
    for frame in rgb_frames:
        if frame.shape[:2] != (height, width) or frame.shape[-1] != 3:
            raise ValueError("All rendered frames must have the same HxWx3 shape.")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(1, int(fps))),
        "-i",
        "-",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for frame in rgb_frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
        returncode = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}:\n{stderr or stdout}")


def _save_prediction_npz(path: Path, prediction: Dict[str, Any], keep_meta: Dict[str, Any]) -> None:
    pred_abs = _future_abs_from_delta(
        np.asarray(prediction["query_world_xyz"], dtype=np.float32),
        np.nan_to_num(np.asarray(prediction["pred_delta_xyz"], dtype=np.float32)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        query_uv=np.asarray(prediction["query_uv"], dtype=np.float32),
        query_world_xyz=np.asarray(prediction["query_world_xyz"], dtype=np.float32),
        query_valid=np.asarray(prediction["query_valid"], dtype=bool),
        pred_delta_xyz=np.asarray(prediction["pred_delta_xyz"], dtype=np.float32),
        pred_abs_xyz=pred_abs.astype(np.float32),
        intrinsics=np.asarray(prediction["intrinsics"], dtype=np.float32),
        extrinsics_w2c=np.asarray(prediction["extrinsics_w2c"], dtype=np.float32),
        metadata_json=np.array(json.dumps(keep_meta, indent=2)),
    )


def _stem_from_summary(summary_path: Path, frame_index: int) -> str:
    stem = summary_path.name
    if stem.endswith("_summary.json"):
        stem = stem[: -len("_summary.json")]
    return f"{stem}_frame{frame_index:04d}"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    summary_path, rgb_video, depth_video, summary = _resolve_rollout_files(args)
    output_dir = args.output_dir or (args.rollout_dir / "motion_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_seq = _read_rgb_video(rgb_video)
    depth_seq = _load_depth(summary, depth_video, args.rollout_dir)
    total = min(rgb_seq.shape[0], depth_seq.shape[0])
    frame_idx = int(args.frame_index)
    if frame_idx < 0:
        frame_idx += total
    if frame_idx < 0 or frame_idx >= total:
        raise IndexError(f"frame_index={args.frame_index} outside [0, {total}).")
    instruction = args.instruction.strip() or str(summary.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("No instruction found. Pass --instruction or use a rollout summary with instruction.")

    rgb = np.ascontiguousarray(rgb_seq[frame_idx])
    depth = np.ascontiguousarray(depth_seq[frame_idx].astype(np.float32))
    print(
        "[rollout-motion-viz] "
        f"folder={args.rollout_dir} frame={frame_idx}/{total - 1} "
        f"instruction={instruction!r}"
    )
    print(f"[rollout-motion-viz] rgb={rgb_video}")
    if depth_video is not None and depth_video.exists():
        print(f"[rollout-motion-viz] depth_video={depth_video}")
    if args.dry_run:
        print("[rollout-motion-viz] dry run: loaded videos and summary only.")
        return

    device = _device_from_arg(args.device)
    prediction = _predict(
        rgb=rgb,
        depth=depth,
        instruction=instruction,
        args=args,
        device=device,
    )
    stem = _stem_from_summary(summary_path, frame_idx)
    output_video = output_dir / f"{stem}_pred_traj.mp4"
    output_spaghetti = output_dir / f"{stem}_spaghetti.png"
    output_npz = output_dir / f"{stem}_prediction.npz"
    output_video, output_spaghetti, moving_points = _render_prediction_video(
        rgb=rgb,
        depth=depth,
        prediction=prediction,
        output_video=output_video,
        output_spaghetti=output_spaghetti,
        args=args,
    )
    metadata = {
        "summary": str(summary_path),
        "rgb_video": str(rgb_video),
        "depth_video": str(depth_video),
        "frame_index": frame_idx,
        "instruction": instruction,
        "moving_points": int(moving_points),
        "grid_size": int(args.grid_size),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "inference_steps": int(prediction["inference_steps"]),
    }
    _save_prediction_npz(output_npz, prediction, metadata)
    print(f"[rollout-motion-viz] saved video: {output_video}")
    print(f"[rollout-motion-viz] saved final spaghetti: {output_spaghetti}")
    print(f"[rollout-motion-viz] saved query/prediction npz: {output_npz}")


if __name__ == "__main__":
    main()
