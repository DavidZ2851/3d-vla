from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MOTION_ANNOTATION_ROOT = Path("/data/user_data/ziyinx/3dfa-vla-data-processing")
DEFAULT_SPATRACKER_ROOT = Path("/data/user_data/ziyinx/SpaTrackerV2")
DEFAULT_SPATRACKER_CHECKPOINT = DEFAULT_SPATRACKER_ROOT / "checkpoints/SpatialTrackerV2-Offline"
DEFAULT_SPATRACKER_CONFIG = DEFAULT_SPATRACKER_ROOT / "config/magic_infer_offline.yaml"
DEFAULT_CONFIG = REPO_ROOT / "motion_expert/configs/calvin_florence2.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt"
)
DEFAULT_PACKAGE = REPO_ROOT / "train_logs/motion_expert/calvin_pointworld_viz/sample_rgbd_tracks.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "train_logs/motion_expert/calvin_pointworld_viz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one CALVIN RGB-D + 20x20x15 motion-track package, run a trained "
            "MotionExpert on selected RGB frames, and render PointWorld-style "
            "point-cloud trajectory videos."
        )
    )
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rerun-tracking", action="store_true")
    parser.add_argument(
        "--calvin-root",
        type=Path,
        default=None,
        help=(
            "CALVIN split directory containing lang_annotations/auto_lang_ann.npy "
            "and episode_*.npz. Required with --rerun-tracking."
        ),
    )
    parser.add_argument("--ann-id", type=int, default=None)
    parser.add_argument("--unit-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--future-len", type=int, default=15)
    parser.add_argument("--query-grid-size", type=int, default=20)
    parser.add_argument("--motion-annotation-root", type=Path, default=DEFAULT_MOTION_ANNOTATION_ROOT)
    parser.add_argument("--spatracker-root", type=Path, default=DEFAULT_SPATRACKER_ROOT)
    parser.add_argument("--spatracker-checkpoint", type=str, default=str(DEFAULT_SPATRACKER_CHECKPOINT))
    parser.add_argument("--spatracker-config", type=Path, default=DEFAULT_SPATRACKER_CONFIG)
    parser.add_argument("--spatracker-track-num", type=int, default=756)
    parser.add_argument("--spatracker-iters-track", type=int, default=2)
    parser.add_argument("--track-visibility-threshold", type=float, default=0.05)
    parser.add_argument("--track-confidence-threshold", type=float, default=0.05)
    parser.add_argument("--track-jump-max-meters", type=float, default=0.12)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default="0",
        help="Comma-separated package frame indices to predict from, e.g. 0,4,8.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--video-frames", type=int, default=96)
    parser.add_argument("--pcd-stride", type=int, default=2)
    parser.add_argument("--pcd-max-points", type=int, default=45000)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--motion-threshold", type=float, default=0.002)
    parser.add_argument("--max-moving-points", type=int, default=240)
    parser.add_argument("--anchor-size", type=float, default=34.0)
    parser.add_argument("--line-width", type=float, default=1.6)
    parser.add_argument("--scene-alpha", type=float, default=0.38)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def _parse_frame_indices(raw: str, num_frames: int) -> List[int]:
    out: List[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx += num_frames
        if idx < 0 or idx >= num_frames:
            raise IndexError(f"Frame index {idx} is outside package range [0, {num_frames}).")
        out.append(idx)
    if not out:
        out.append(0)
    return out


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if str(path).endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle)


def _npz_scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _build_grid_points(height: int, width: int, grid_size: int) -> np.ndarray:
    ys = np.linspace(0.0, float(height - 1), grid_size, endpoint=False, dtype=np.float32)
    xs = np.linspace(0.0, float(width - 1), grid_size, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)


def _bilinear_depth_sample(depth: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u = uv[:, 0]
    v = uv[:, 1]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    in_bounds = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full((uv.shape[0],), np.nan, dtype=np.float32)
    idx = np.where(in_bounds)[0]
    if idx.size == 0:
        return out, np.zeros_like(in_bounds, dtype=bool)
    x0v, y0v, x1v, y1v = x0[idx], y0[idx], x1[idx], y1[idx]
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
        du2, dv2 = du[valid], dv[valid]
        out[idx2] = (
            (1.0 - du2) * (1.0 - dv2) * z00[valid]
            + du2 * (1.0 - dv2) * z10[valid]
            + (1.0 - du2) * dv2 * z01[valid]
            + du2 * dv2 * z11[valid]
        )
    valid_out = np.isfinite(out) & (out > 0.0)
    return out, valid_out


def _unproject_uv_depth_to_world(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
) -> np.ndarray:
    n = uv.shape[0]
    ones = np.ones((n, 1), dtype=np.float32)
    uv1 = np.concatenate([uv.astype(np.float32), ones], axis=1)
    xyz_cam = (np.linalg.inv(intrinsic).astype(np.float32) @ uv1.T).T * depth[:, None]
    xyz_cam_h = np.concatenate([xyz_cam, ones], axis=1)
    c2w = np.linalg.inv(extrinsic_w2c).astype(np.float32)
    xyz_world_h = (c2w @ xyz_cam_h.T).T
    return xyz_world_h[:, :3].astype(np.float32)


def _rgbd_to_world_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    *,
    stride: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    uv = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    z = depth[yy, xx].reshape(-1).astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0)
    uv = uv[valid]
    z = z[valid]
    points = _unproject_uv_depth_to_world(uv, z, intrinsic, extrinsic_w2c)
    colors = rgb[yy, xx].reshape(-1, 3)[valid].astype(np.float32) / 255.0
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points.astype(np.float32), colors.astype(np.float32)


def _query_world_xyz_from_rgbd(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    query_uv = _build_grid_points(h, w, grid_size)
    query_depth, valid = _bilinear_depth_sample(depth, query_uv)
    safe_depth = np.where(valid, query_depth, 0.0).astype(np.float32)
    query_xyz = _unproject_uv_depth_to_world(query_uv, safe_depth, intrinsic, extrinsic_w2c)
    query_xyz[~valid] = 0.0
    return query_uv, query_xyz.astype(np.float32), valid


def _future_abs_from_delta(anchor_xyz: np.ndarray, delta_xyz: np.ndarray) -> np.ndarray:
    return anchor_xyz[:, None, :] + np.cumsum(delta_xyz, axis=1)


def rerun_tracking_and_save_package(args: argparse.Namespace) -> Path:
    if args.calvin_root is None:
        raise ValueError("--calvin-root is required when --rerun-tracking is set.")
    ma_root = args.motion_annotation_root.resolve()
    if str(ma_root) not in sys.path:
        sys.path.insert(0, str(ma_root))

    import motion_annotation as ma  # type: ignore

    training_root = ma.resolve_single_calvin_root(str(args.calvin_root))
    manifest_path = args.output_dir / "calvin_manifest.jsonl"
    if manifest_path.exists():
        rows = ma.load_calvin_manifest(manifest_path)
    else:
        rows = ma.build_calvin_manifest_rows(training_root)
        ma.save_calvin_manifest(manifest_path, rows)
    if args.ann_id is not None:
        matches = [row for row in rows if int(row.ann_id) == int(args.ann_id)]
        if not matches:
            raise ValueError(f"No CALVIN manifest row found for ann_id={args.ann_id}.")
        row = matches[0]
    else:
        if args.unit_index < 0 or args.unit_index >= len(rows):
            raise IndexError(f"--unit-index must be in [0, {len(rows)}).")
        row = rows[int(args.unit_index)]

    ep_data = ma.load_calvin_segment_data(training_root, row)
    query_index = int(args.query_index)
    if query_index < 0 or query_index >= ep_data.rgb.shape[0] - 1:
        raise IndexError(
            f"--query-index must be in [0, {ep_data.rgb.shape[0] - 1}) for this segment."
        )

    device = str(_device_from_arg(args.device))
    runner = ma.SpaTrackerOfflineRunner(
        spatracker_root=args.spatracker_root,
        checkpoint=args.spatracker_checkpoint,
        device=device,
        config_path=args.spatracker_config,
        track_num=max(int(args.spatracker_track_num), int(args.query_grid_size) ** 2),
        iters_track=int(args.spatracker_iters_track),
    )
    datapoint, metrics = ma.build_datapoint(
        runner=runner,
        unit_rgb=ep_data.rgb,
        unit_depth=ep_data.depth,
        unit_intrinsics=ep_data.intrinsics,
        unit_extrinsics_w2c=ep_data.extrinsics_w2c,
        query_index=query_index,
        future_len=int(args.future_len),
        query_grid_size=int(args.query_grid_size),
        track_visibility_threshold=float(args.track_visibility_threshold),
        track_confidence_threshold=float(args.track_confidence_threshold),
        track_jump_max_meters=float(args.track_jump_max_meters),
    )
    if datapoint is None:
        raise RuntimeError(f"Motion tracking dropped this query frame: {metrics}")

    end = min(query_index + int(args.future_len), ep_data.rgb.shape[0] - 1)
    frame_slice = slice(query_index, end + 1)
    package_path = args.package
    package_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        package_path,
        rgb=ep_data.rgb[frame_slice].astype(np.uint8),
        depth=ep_data.depth[frame_slice].astype(np.float32),
        intrinsics=ep_data.intrinsics[frame_slice].astype(np.float32),
        extrinsics_w2c=ep_data.extrinsics_w2c[frame_slice].astype(np.float32),
        raw_frame_indices=ep_data.raw_frame_indices[frame_slice].astype(np.int32),
        instruction=np.array(str(row.instruction)),
        ann_id=np.array(row.ann_id, dtype=np.int32),
        rollout_index=np.array(row.rollout_index, dtype=np.int32),
        source_query_index=np.array(query_index, dtype=np.int32),
        package_query_frame_index=np.array(0, dtype=np.int32),
        query_uv=datapoint["query_uv"].astype(np.float32),
        query_world_xyz=datapoint["query_world_xyz"].astype(np.float32),
        gt_delta_xyz=datapoint["traj_3d_xyz"].astype(np.float32),
        gt_abs_xyz=_future_abs_from_delta(
            datapoint["query_world_xyz"].astype(np.float32),
            np.nan_to_num(datapoint["traj_3d_xyz"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        ).astype(np.float32),
        traj_mask=datapoint["valid_mask"].astype(bool),
        metrics_json=np.array(json.dumps(metrics)),
    )
    print(f"[calvin-pointworld] saved RGB-D/GT package: {package_path}")
    return package_path


def _load_package(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Package does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    package = {key: data[key] for key in data.files}
    required = ("rgb", "depth", "intrinsics", "extrinsics_w2c", "instruction")
    missing = [key for key in required if key not in package]
    if missing:
        raise KeyError(f"Package {path} is missing required key(s): {missing}")
    return package


def _build_motion_expert(config_path: Path, checkpoint_path: Path, device: torch.device):
    from motion_expert.compute_calvin_normalization_stats import (
        extract_calvin_normalization_tensors,
        load_calvin_normalization_stats,
    )
    from motion_expert.config import build_motion_expert_config
    from motion_expert.pipeline import MotionExpertPipeline

    raw_cfg = _load_yaml_or_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = build_motion_expert_config(checkpoint.get("config", raw_cfg["model"]))

    if model_cfg.qwen.model_name_or_path is not None:
        os.environ["QWEN3VL_MODEL_PATH"] = model_cfg.qwen.model_name_or_path
    if model_cfg.florence.model_name_or_path is not None:
        os.environ["FLORENCE2_MODEL_PATH"] = model_cfg.florence.model_name_or_path

    pipeline = MotionExpertPipeline.from_backbone(model_cfg)
    train_cfg = raw_cfg.get("train", {})
    if bool(train_cfg.get("use_calvin_normalization", False)):
        stats_path = train_cfg.get("normalization_stats_path")
        if stats_path:
            stats = load_calvin_normalization_stats(Path(stats_path))
            query_center, query_scale, traj_center, traj_scale = extract_calvin_normalization_tensors(stats)
            pipeline.set_calvin_normalization_statistics(
                torch.from_numpy(query_center),
                torch.from_numpy(query_scale),
                torch.from_numpy(traj_center),
                torch.from_numpy(traj_scale),
            )
    pipeline.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pipeline = pipeline.to(device).eval()
    return raw_cfg, model_cfg, pipeline


def _make_motion_batch(
    *,
    model_cfg,
    tokenizer,
    rgb: np.ndarray,
    instruction: str,
    query_uv_source: np.ndarray,
    query_world_xyz: np.ndarray,
    gt_delta_xyz: Optional[np.ndarray] = None,
    traj_mask: Optional[np.ndarray] = None,
):
    from motion_expert.data import MotionExpertBatch
    from motion_expert.geometry import aspect_resize_and_pad_rgb_and_query_uv, center_query_world_xyz

    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    rgb_model, query_uv_model = aspect_resize_and_pad_rgb_and_query_uv(
        rgb=rgb_tensor,
        query_uv=torch.from_numpy(query_uv_source).float(),
        target_hw=(int(model_cfg.input_height), int(model_cfg.input_width)),
        query_uv_format="source_pixels",
        pad_value=float(model_cfg.image_pad_value),
    )
    tokenized = tokenizer([instruction])
    if gt_delta_xyz is not None:
        gt_delta_xyz = np.nan_to_num(gt_delta_xyz, nan=0.0, posinf=0.0, neginf=0.0)
    future = None if gt_delta_xyz is None else torch.from_numpy(gt_delta_xyz).float().unsqueeze(0)
    mask = None if traj_mask is None else torch.from_numpy(traj_mask).bool().unsqueeze(0)
    return MotionExpertBatch(
        rgb=rgb_model.unsqueeze(0),
        instruction_text=[instruction],
        text_token_ids=tokenized.input_ids,
        text_attention_mask=tokenized.attention_mask,
        text_valid_lens=tokenized.attention_mask.sum(dim=1),
        query_uv=query_uv_model.unsqueeze(0),
        query_world_xyz=center_query_world_xyz(torch.from_numpy(query_world_xyz).float()).unsqueeze(0),
        future_delta_xyz=future,
        traj_mask=mask,
        image_hw=torch.tensor([[int(model_cfg.input_height), int(model_cfg.input_width)]], dtype=torch.long),
    )


@torch.no_grad()
def predict_tracks(
    package: Dict[str, Any],
    args: argparse.Namespace,
    frame_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    from motion_expert.zarr_dataset import MotionExpertQwenTokenizer

    device = _device_from_arg(args.device)
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
    instruction = _npz_scalar_to_str(package["instruction"])
    predictions: List[Dict[str, Any]] = []
    rgb_seq = np.asarray(package["rgb"], dtype=np.uint8)
    depth_seq = np.asarray(package["depth"], dtype=np.float32)
    intr_seq = np.asarray(package["intrinsics"], dtype=np.float32)
    ext_seq = np.asarray(package["extrinsics_w2c"], dtype=np.float32)

    for frame_idx in frame_indices:
        query_uv, query_xyz, query_valid = _query_world_xyz_from_rgbd(
            depth_seq[frame_idx],
            intr_seq[frame_idx],
            ext_seq[frame_idx],
            int(model_cfg.grid_size),
        )
        gt_delta = None
        gt_mask = None
        package_query_frame = int(np.asarray(package.get("package_query_frame_index", 0)).item())
        if frame_idx == package_query_frame and "gt_delta_xyz" in package:
            gt_delta = np.asarray(package["gt_delta_xyz"], dtype=np.float32)
            gt_mask = np.asarray(package["traj_mask"], dtype=bool)

        batch = _make_motion_batch(
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            rgb=rgb_seq[frame_idx],
            instruction=instruction,
            query_uv_source=query_uv,
            query_world_xyz=query_xyz,
            gt_delta_xyz=gt_delta,
            traj_mask=gt_mask,
        ).to(device)
        outputs = pipeline.predict(batch, num_inference_steps=inference_steps)
        pred_delta = outputs["pred_delta_xyz"].detach().cpu().float().numpy()[0]
        predictions.append(
            {
                "frame_idx": int(frame_idx),
                "query_uv": query_uv,
                "query_world_xyz": query_xyz,
                "query_valid": query_valid,
                "pred_delta_xyz": pred_delta,
                "instruction": instruction,
                "metrics": {
                    key: float(value.detach().cpu().item())
                    for key, value in outputs.items()
                    if torch.is_tensor(value) and value.ndim == 0
                },
            }
        )
        print(
            "[calvin-pointworld] predicted "
            f"frame={frame_idx} valid_query_points={int(query_valid.sum())}/{query_valid.size}"
        )
    return predictions


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
    if idx.size > max_points:
        order = np.argsort(disp[idx])[::-1][:max_points]
        idx = idx[order]
    return idx.astype(np.int64)


def _axis_limits_for_focus(points: np.ndarray, padding: float) -> Tuple[np.ndarray, float]:
    finite = np.isfinite(points).all(axis=1)
    pts = points[finite]
    if pts.size == 0:
        center = np.zeros(3, dtype=np.float32)
        return center, 0.5
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) + float(padding)
    return center.astype(np.float32), max(radius, 0.05)


def _set_3d_limits(ax, center: np.ndarray, radius: float, zoom: float) -> None:
    r = max(float(radius) / max(float(zoom), 1e-6), 1e-3)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def _render_prediction_video(
    *,
    package: Dict[str, Any],
    prediction: Dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    import matplotlib.pyplot as plt

    frame_idx = int(prediction["frame_idx"])
    rgb = np.asarray(package["rgb"], dtype=np.uint8)[frame_idx]
    depth = np.asarray(package["depth"], dtype=np.float32)[frame_idx]
    intr = np.asarray(package["intrinsics"], dtype=np.float32)[frame_idx]
    ext = np.asarray(package["extrinsics_w2c"], dtype=np.float32)[frame_idx]
    scene_points, scene_colors = _rgbd_to_world_cloud(
        rgb,
        depth,
        intr,
        ext,
        stride=max(1, int(args.pcd_stride)),
        max_points=max(1, int(args.pcd_max_points)),
    )

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
    traj_points = np.concatenate(
        [query_xyz[keep_idx, None, :], pred_abs[keep_idx]],
        axis=1,
    )
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="calvin_pointworld_"))
    fig = plt.figure(figsize=(7.4, 7.4), dpi=int(args.dpi))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(keep_idx), 1)))
    try:
        total_frames = max(2, int(args.video_frames))
        future_steps = int(pred_delta.shape[1])
        for video_i in range(total_frames):
            phase = video_i / float(total_frames - 1)
            hist_count = int(round(phase * future_steps))
            azim = -68.0 + 52.0 * phase
            elev = 24.0 + 5.0 * np.sin(phase * np.pi)
            zoom = 1.0 + 0.55 * phase
            ax.cla()
            if scene_points_plot.size:
                ax.scatter(
                    scene_points_plot[:, 0],
                    scene_points_plot[:, 1],
                    scene_points_plot[:, 2],
                    c=scene_colors_plot,
                    s=1.0,
                    alpha=float(args.scene_alpha),
                    linewidths=0,
                    depthshade=False,
                )
            for local_i, point_idx in enumerate(keep_idx.tolist()):
                anchor = query_xyz[point_idx]
                color = colors[local_i]
                ax.scatter(
                    anchor[0:1],
                    anchor[1:2],
                    anchor[2:3],
                    color=color[None, :],
                    s=float(args.anchor_size),
                    edgecolors="black",
                    linewidths=0.35,
                    depthshade=False,
                )
                if hist_count <= 0:
                    continue
                poly = np.concatenate([anchor[None], pred_abs[point_idx, :hist_count]], axis=0)
                ax.plot(
                    poly[:, 0],
                    poly[:, 1],
                    poly[:, 2],
                    color=color,
                    linewidth=float(args.line_width),
                    alpha=0.92,
                )

            _set_3d_limits(ax, center, radius, zoom)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            ax.set_box_aspect((1, 1, 1))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            frame_path = tmp_dir / f"frame_{video_i:05d}.png"
            fig.savefig(frame_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found in PATH; cannot write MP4.")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(max(1, int(args.fps))),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing {output_path}:\n{proc.stderr}")
    finally:
        plt.close(fig)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(
        "[calvin-pointworld] rendered "
        f"frame={frame_idx} moving_points={len(keep_idx)} video={output_path}"
    )
    return output_path


def _save_predictions(path: Path, predictions: Sequence[Dict[str, Any]]) -> None:
    payload: Dict[str, Any] = {"num_predictions": np.array(len(predictions), dtype=np.int32)}
    for i, pred in enumerate(predictions):
        prefix = f"pred_{i:03d}"
        payload[f"{prefix}_frame_idx"] = np.array(pred["frame_idx"], dtype=np.int32)
        payload[f"{prefix}_query_uv"] = pred["query_uv"].astype(np.float32)
        payload[f"{prefix}_query_world_xyz"] = pred["query_world_xyz"].astype(np.float32)
        payload[f"{prefix}_query_valid"] = pred["query_valid"].astype(bool)
        payload[f"{prefix}_delta_xyz"] = pred["pred_delta_xyz"].astype(np.float32)
        payload[f"{prefix}_metrics_json"] = np.array(json.dumps(pred.get("metrics", {})))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"[calvin-pointworld] saved predictions: {path}")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.rerun_tracking:
        rerun_tracking_and_save_package(args)
    package = _load_package(args.package)
    frame_indices = _parse_frame_indices(args.frame_indices, int(np.asarray(package["rgb"]).shape[0]))

    predictions: List[Dict[str, Any]] = []
    if not args.skip_predict:
        predictions = predict_tracks(package, args, frame_indices)
        if args.save_predictions:
            _save_predictions(args.output_dir / "predicted_point_tracks.npz", predictions)

    if not args.skip_render:
        if not predictions:
            raise ValueError("Rendering needs predictions; omit --skip-predict or load predictions support first.")
        for pred in predictions:
            output_path = args.output_dir / f"pointworld_pred_frame_{int(pred['frame_idx']):03d}.mp4"
            _render_prediction_video(
                package=package,
                prediction=pred,
                output_path=output_path,
                args=args,
            )


if __name__ == "__main__":
    main()
