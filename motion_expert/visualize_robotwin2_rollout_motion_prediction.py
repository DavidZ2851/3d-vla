#!/usr/bin/env python3
"""Predict and render MotionExpert trajectories from a saved RoboTwin2 rollout.

RoboTwin2 analog of ``visualize_libero_rollout_motion_prediction.py``. The head
camera matrices are written into the rollout summary by the opt-in recorder in
``online_evaluation_robotwin/rollout_recorder.py`` during an eval rollout
(enable with ROBOTWIN_ROLLOUT_DUMP_DIR).

Unlike LIBERO (which stores camera-to-world), RoboTwin stores ``extrinsic_cv``
as world->camera OpenCV already, so ``head_extrinsics_w2c`` is used verbatim --
no inversion. Everything else (depth load, prediction, render, npz dump) is
reused from the CALVIN rollout module.

Example:
  python -m motion_expert.visualize_robotwin2_rollout_motion_prediction \
      --rollout-dir vis_outputs/robotwin2_task_rollouts/beat_block_hammer \
      --frame-index 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import motion_expert.visualize_calvin_rollout_motion_prediction as crm

DEFAULT_ROLLOUT_DIR = REPO_ROOT / "vis_outputs/robotwin2_task_rollouts/beat_block_hammer"
DEFAULT_CONFIG = REPO_ROOT / "motion_expert/configs/robotwin2_florence2_viz.yml"
DEFAULT_CHECKPOINT = REPO_ROOT / "train_logs/motion_expert/robotwin2_florence2/checkpoint_step_20000.pt"


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
    p.add_argument("--pcd-stride", type=int, default=3)
    p.add_argument("--pcd-max-points", type=int, default=25000)
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


def _head_matrices(summary: dict, depth_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (intrinsics scaled to depth size, world->camera extrinsics).

    RoboTwin's extrinsic_cv is already world->camera, so it is used as-is.
    """
    if "head_intrinsics" not in summary or "head_extrinsics_w2c" not in summary:
        raise KeyError(
            "RoboTwin2 rollout summary is missing camera matrices "
            "('head_intrinsics' / 'head_extrinsics_w2c'). Re-run the rollout with "
            "ROBOTWIN_ROLLOUT_DUMP_DIR set to regenerate it."
        )
    intr = np.asarray(summary["head_intrinsics"], dtype=np.float32)
    ext_w2c = np.asarray(summary["head_extrinsics_w2c"], dtype=np.float32)

    src_h = int(summary.get("img_h", depth_hw[0]))
    src_w = int(summary.get("img_w", depth_hw[1]))
    h, w = int(depth_hw[0]), int(depth_hw[1])
    if (h, w) != (src_h, src_w):
        intr = intr.copy()
        intr[0, :] *= float(w) / float(src_w)
        intr[1, :] *= float(h) / float(src_h)
    return intr.astype(np.float32), ext_w2c.astype(np.float32)


@torch.no_grad()
def _predict_robotwin2(*, rgb, depth, instruction, intr, ext_w2c, args, device):
    from motion_expert.visualize_calvin_pointworld import (
        _make_motion_batch,
        _query_world_xyz_from_rgbd,
    )

    query_uv, query_xyz, query_valid = _query_world_xyz_from_rgbd(
        depth, intr, ext_w2c, int(args.grid_size)
    )
    model_cfg, pipeline, tokenizer, inference_steps = crm._load_motion_predictor(args, device)
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
        "extrinsics_w2c": ext_w2c,
        "query_uv": query_uv,
        "query_world_xyz": query_xyz,
        "query_valid": query_valid,
        "pred_delta_xyz": pred_delta,
        "inference_steps": inference_steps,
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    summary_path, rgb_video, depth_video, summary = crm._resolve_rollout_files(args)
    output_dir = args.output_dir or (args.rollout_dir / "motion_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_seq = crm._read_rgb_video(rgb_video)
    depth_seq = crm._load_depth(summary, depth_video, args.rollout_dir)
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
    intr, ext_w2c = _head_matrices(summary, depth.shape)
    print(
        "[robotwin2-rollout-motion-viz] "
        f"folder={args.rollout_dir} frame={frame_idx}/{total - 1} "
        f"instruction={instruction!r}"
    )
    print(f"[robotwin2-rollout-motion-viz] rgb={rgb_video}")
    if args.dry_run:
        print("[robotwin2-rollout-motion-viz] dry run: loaded videos, summary, and camera matrices only.")
        print(f"  intrinsics=\n{intr}\n  extrinsics_w2c=\n{ext_w2c}")
        return

    device = crm._device_from_arg(args.device)
    prediction = _predict_robotwin2(
        rgb=rgb,
        depth=depth,
        instruction=instruction,
        intr=intr,
        ext_w2c=ext_w2c,
        args=args,
        device=device,
    )
    stem = crm._stem_from_summary(summary_path, frame_idx)
    output_video = output_dir / f"{stem}_pred_traj.mp4"
    output_spaghetti = output_dir / f"{stem}_spaghetti.png"
    output_npz = output_dir / f"{stem}_prediction.npz"
    output_video, output_spaghetti, moving_points = crm._render_prediction_video(
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
    crm._save_prediction_npz(output_npz, prediction, metadata)
    print(f"[robotwin2-rollout-motion-viz] saved video: {output_video}")
    print(f"[robotwin2-rollout-motion-viz] saved final spaghetti: {output_spaghetti}")
    print(f"[robotwin2-rollout-motion-viz] saved query/prediction npz: {output_npz}")


if __name__ == "__main__":
    main()
