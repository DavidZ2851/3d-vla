from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from motion_expert.visualize_calvin_pointworld import (
    DEFAULT_MOTION_ANNOTATION_ROOT,
    DEFAULT_SPATRACKER_CHECKPOINT,
    DEFAULT_SPATRACKER_CONFIG,
    DEFAULT_SPATRACKER_ROOT,
    _device_from_arg,
    _ensure_flow_vis_available,
    _ensure_utils3d_available,
    _future_abs_from_delta,
    _load_package,
    _parse_frame_indices,
    _render_episode_video,
    _save_predictions,
    _write_rgb_video,
    predict_tracks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBERO_ROOT = Path("/data/group_data/katefgroup/datasets/ziyinx/libero")
DEFAULT_CONFIG = REPO_ROOT / "motion_expert/configs/libero_florence2.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "train_logs/motion_expert/libero_florence2/B32_18k.pt"
DEFAULT_OUTPUT_DIR = Path("/data/user_data/ziyinx/vis_outputs/libero_pointworld")
DEFAULT_PACKAGE = DEFAULT_OUTPUT_DIR / "sample_rgbd_tracks.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one LIBERO RGB-D + 20x20x15 motion-track package, run a trained "
            "MotionExpert checkpoint, and render PointWorld-style point-cloud trajectory videos."
        )
    )
    parser.add_argument("--stage", choices=("track", "predict", "all"), default="all")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Global index in sorted LIBERO demos under --libero-root.",
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=None,
        help="Optional exact LIBERO task HDF5. If set, --demo-index selects the demo in this file.",
    )
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument(
        "--camera-name",
        choices=("agentview", "eye_in_hand"),
        default="agentview",
    )
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--future-len", type=int, default=15)
    parser.add_argument("--track-step", type=int, default=3)
    parser.add_argument("--query-stride-raw", type=int, default=45)
    parser.add_argument("--max-query-chunks", type=int, default=-1)
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
    parser.add_argument("--frame-indices", type=str, default="0")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--raw-rgb-video", type=Path, default=None)
    parser.add_argument("--skip-raw-rgb-video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=16)
    parser.add_argument("--pcd-stride", type=int, default=2)
    parser.add_argument("--pcd-max-points", type=int, default=45000)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--motion-threshold", type=float, default=0.002)
    parser.add_argument("--max-moving-points", type=int, default=240)
    parser.add_argument("--anchor-size", type=float, default=5.0)
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument("--gt-line-width", type=float, default=2.4)
    parser.add_argument("--scene-alpha", type=float, default=1.0)
    parser.add_argument("--scene-point-size", type=float, default=1.25)
    parser.add_argument("--scene-color-gain", type=float, default=1.8)
    parser.add_argument("--camera-zoom", type=float, default=1.35)
    parser.add_argument("--query-color", type=str, default="#7dd3fc")
    parser.add_argument("--pred-color", type=str, default="#00d084")
    parser.add_argument("--gt-color", type=str, default="#ff4f9a")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def _load_motion_annotation(args: argparse.Namespace):
    injected_utils3d = _ensure_utils3d_available()
    if injected_utils3d is not None:
        print(f"[libero-pointworld] using local utils3d from {injected_utils3d}")
    injected_flow_vis = _ensure_flow_vis_available()
    if injected_flow_vis is not None:
        print(f"[libero-pointworld] using local flow_vis from {injected_flow_vis}")
    ma_root = args.motion_annotation_root.resolve()
    if str(ma_root) not in sys.path:
        sys.path.insert(0, str(ma_root))
    import motion_annotation as ma  # type: ignore

    return ma


def _select_libero_episode(ma: Any, args: argparse.Namespace) -> Any:
    if args.hdf5_path is not None:
        hdf5_path = args.hdf5_path.resolve()
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Missing LIBERO HDF5: {hdf5_path}")
        try:
            import h5py  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError("h5py is required to inspect LIBERO HDF5 demos.") from exc
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(
                list(f["data"].keys()),
                key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else n,
            )
        if args.demo_index < 0 or args.demo_index >= len(demo_keys):
            raise IndexError(f"--demo-index must be in [0, {len(demo_keys)}).")
        task_name = hdf5_path.stem
        if task_name.endswith("_demo"):
            task_name = task_name[: -len("_demo")]
        return ma.LiberoEpisodeRef(
            suite_name=hdf5_path.parent.name,
            task_name=task_name,
            file_index=-1,
            demo_index=int(args.demo_index),
            demo_key=demo_keys[int(args.demo_index)],
            hdf5_path=hdf5_path,
        )

    episodes = ma.discover_libero_episodes(args.libero_root)
    if not episodes:
        raise ValueError(f"No LIBERO demos found under {args.libero_root}")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"--episode-index must be in [0, {len(episodes)}).")
    return episodes[int(args.episode_index)]


def rerun_tracking_and_save_package(args: argparse.Namespace) -> Path:
    ma = _load_motion_annotation(args)
    ep = _select_libero_episode(ma, args)
    ep_data = ma.load_libero_episode_data(ep, args.camera_name)
    if ep_data is None:
        raise RuntimeError(f"Failed to load LIBERO episode {ep.hdf5_path}:{ep.demo_key}")

    query_index = int(args.query_index)
    if query_index < 0 or query_index >= ep_data.rgb.shape[0] - 1:
        raise IndexError(
            f"--query-index must be in [0, {ep_data.rgb.shape[0] - 1}) for this demo."
        )
    future_len = int(args.future_len)
    track_step = int(args.track_step)
    query_stride_raw = int(args.query_stride_raw)
    if future_len <= 0:
        raise ValueError("--future-len must be positive.")
    if track_step <= 0:
        raise ValueError("--track-step must be positive.")
    if query_stride_raw <= 0:
        raise ValueError("--query-stride-raw must be positive.")

    full_horizon = future_len * track_step
    max_query = ep_data.rgb.shape[0] - 1 - full_horizon
    if max_query >= query_index:
        query_indices = list(range(query_index, max_query + 1, query_stride_raw))
        tail_query = max(0, ep_data.rgb.shape[0] - 1 - full_horizon)
        if tail_query not in query_indices:
            query_indices.append(tail_query)
    else:
        query_indices = [query_index]
    query_indices = sorted(set(int(q) for q in query_indices))
    if int(args.max_query_chunks) >= 0:
        query_indices = query_indices[: int(args.max_query_chunks)]
    if not query_indices:
        raise ValueError("No query chunks fit in this demo.")

    device = str(_device_from_arg(args.device))
    runner = ma.SpaTrackerOfflineRunner(
        spatracker_root=args.spatracker_root,
        checkpoint=args.spatracker_checkpoint,
        device=device,
        config_path=args.spatracker_config,
        track_num=max(int(args.spatracker_track_num), int(args.query_grid_size) ** 2),
        iters_track=int(args.spatracker_iters_track),
    )

    query_uv_chunks: List[np.ndarray] = []
    query_world_xyz_chunks: List[np.ndarray] = []
    gt_delta_chunks: List[np.ndarray] = []
    gt_abs_chunks: List[np.ndarray] = []
    traj_mask_chunks: List[np.ndarray] = []
    chunk_frame_indices: List[np.ndarray] = []
    chunk_raw_frame_indices: List[np.ndarray] = []
    metrics_json: List[str] = []
    kept_query_indices: List[int] = []

    for chunk_i, q in enumerate(query_indices):
        raw_window = q + np.arange(future_len + 1, dtype=np.int32) * track_step
        raw_window = np.clip(raw_window, 0, ep_data.rgb.shape[0] - 1).astype(np.int32)
        actual_steps = int(np.count_nonzero(raw_window[1:] > raw_window[:-1]))
        if actual_steps <= 0:
            print(f"[libero-pointworld] dropped query chunk q={q}: no future raw frames")
            continue
        datapoint, metrics = ma.build_datapoint(
            runner=runner,
            unit_rgb=ep_data.rgb[raw_window],
            unit_depth=ep_data.depth[raw_window],
            unit_intrinsics=ep_data.intrinsics[raw_window],
            unit_extrinsics_w2c=ep_data.extrinsics_w2c[raw_window],
            query_index=0,
            future_len=actual_steps,
            query_grid_size=int(args.query_grid_size),
            track_visibility_threshold=float(args.track_visibility_threshold),
            track_confidence_threshold=float(args.track_confidence_threshold),
            track_jump_max_meters=float(args.track_jump_max_meters),
        )
        if datapoint is None:
            print(f"[libero-pointworld] dropped query chunk q={q}: {metrics}")
            continue

        query_uv_chunks.append(datapoint["query_uv"].astype(np.float32))
        query_world_xyz = datapoint["query_world_xyz"].astype(np.float32)
        gt_delta_actual = datapoint["traj_3d_xyz"].astype(np.float32)
        gt_mask_actual = datapoint["valid_mask"].astype(bool)
        gt_delta = np.full((gt_delta_actual.shape[0], future_len, 3), -np.inf, dtype=np.float32)
        gt_mask = np.zeros((gt_mask_actual.shape[0], future_len), dtype=bool)
        gt_delta[:, :actual_steps] = gt_delta_actual[:, :actual_steps]
        gt_mask[:, :actual_steps] = gt_mask_actual[:, :actual_steps]
        query_world_xyz_chunks.append(query_world_xyz)
        gt_delta_chunks.append(gt_delta)
        gt_abs_chunks.append(
            _future_abs_from_delta(
                query_world_xyz,
                np.nan_to_num(gt_delta, nan=0.0, posinf=0.0, neginf=0.0),
            ).astype(np.float32)
        )
        traj_mask_chunks.append(gt_mask)
        chunk_frame_indices.append(raw_window.astype(np.int32))
        chunk_raw_frame_indices.append(ep_data.raw_frame_indices[raw_window].astype(np.int32))
        metrics_json.append(json.dumps(metrics))
        kept_query_indices.append(int(q))
        print(
            "[libero-pointworld] tracked chunk "
            f"{chunk_i + 1}/{len(query_indices)} q={q} raw={int(ep_data.raw_frame_indices[q])}"
        )

    if not kept_query_indices:
        raise RuntimeError("All motion tracking chunks were dropped.")

    instruction = ep.task_name.replace("_", " ")
    package_path = args.package
    package_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        package_path,
        rgb=ep_data.rgb.astype(np.uint8),
        depth=ep_data.depth.astype(np.float32),
        intrinsics=ep_data.intrinsics.astype(np.float32),
        extrinsics_w2c=ep_data.extrinsics_w2c.astype(np.float32),
        raw_frame_indices=ep_data.raw_frame_indices.astype(np.int32),
        instruction=np.array(instruction),
        dataset=np.array("libero"),
        suite_name=np.array(ep.suite_name),
        task_name=np.array(ep.task_name),
        hdf5_path=np.array(str(ep.hdf5_path)),
        demo_key=np.array(ep.demo_key),
        demo_index=np.array(ep.demo_index, dtype=np.int32),
        camera_name=np.array(str(args.camera_name)),
        source_query_index=np.array(query_index, dtype=np.int32),
        track_step=np.array(track_step, dtype=np.int32),
        query_stride_raw=np.array(query_stride_raw, dtype=np.int32),
        query_frame_indices=np.asarray(kept_query_indices, dtype=np.int32),
        raw_query_frame_indices=ep_data.raw_frame_indices[np.asarray(kept_query_indices)].astype(np.int32),
        chunk_frame_indices=np.stack(chunk_frame_indices).astype(np.int32),
        chunk_raw_frame_indices=np.stack(chunk_raw_frame_indices).astype(np.int32),
        query_uv=np.stack(query_uv_chunks).astype(np.float32),
        query_world_xyz=np.stack(query_world_xyz_chunks).astype(np.float32),
        gt_delta_xyz=np.stack(gt_delta_chunks).astype(np.float32),
        gt_abs_xyz=np.stack(gt_abs_chunks).astype(np.float32),
        traj_mask=np.stack(traj_mask_chunks).astype(bool),
        metrics_json=np.asarray(metrics_json),
    )
    print(
        "[libero-pointworld] saved RGB-D/GT package: "
        f"{package_path} chunks={len(kept_query_indices)} demo={ep.hdf5_path}:{ep.demo_key}"
    )
    if not bool(args.skip_raw_rgb_video):
        raw_rgb_video_path = (
            args.raw_rgb_video
            if args.raw_rgb_video is not None
            else package_path.with_name(f"{package_path.stem}_raw_rgb.mp4")
        )
        _write_rgb_video(ep_data.rgb, raw_rgb_video_path, int(args.fps))
    return package_path


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("track", "all"):
        rerun_tracking_and_save_package(args)
    if args.stage == "track":
        print(f"[libero-pointworld] tracking stage complete package={args.package}")
        return

    package = _load_package(args.package)
    frame_indices = _parse_frame_indices(args.frame_indices, int(np.asarray(package["rgb"]).shape[0]))

    predictions: List[Dict[str, Any]] = []
    if not args.skip_predict:
        predictions = predict_tracks(package, args, frame_indices)
        if args.save_predictions:
            _save_predictions(args.output_dir / "predicted_point_tracks.npz", predictions)

    if not args.skip_render:
        if not predictions:
            raise ValueError("Rendering needs predictions; omit --skip-predict.")
        _render_episode_video(
            package=package,
            predictions=predictions,
            output_path=args.output_dir / "pointworld_pred_only.mp4",
            args=args,
            include_gt=False,
        )
        _render_episode_video(
            package=package,
            predictions=predictions,
            output_path=args.output_dir / "pointworld_pred_gt.mp4",
            args=args,
            include_gt=True,
        )


if __name__ == "__main__":
    main()
