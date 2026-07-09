"""Roll out one LIBERO task with a trained policy and save videos + raw depth.

Mirrors online_evaluation_calvin/rollout_task_video.py but for LIBERO. It runs a
single environment for one task / init-state, records the static `agentview`
camera RGB and metric depth every step, and writes:

  task{ID}_{name}_agentview_rgb.mp4      RGB video (motion-viz reads this)
  task{ID}_{name}_agentview_depth.mp4    grayscale depth video (human viewing)
  task{ID}_{name}_agentview_depth_raw.npz   metric depth (H,W) per frame
  task{ID}_{name}_agentview_summary.json    instruction + camera intrinsics/extrinsics

The summary stores the agentview intrinsics (3x3) and camera-to-world extrinsics
(4x4) so a later step can pick a frame and unproject the RGBD for the motion
expert spaghetti prediction (same role the hardcoded CALVIN matrices play in
motion_expert/visualize_calvin_rollout_motion_prediction.py).

Example:
  python -m online_evaluation_libero.rollout_task_video \
      --benchmark libero_spatial --task_id 0 \
      --checkpoint train_logs/libero/3dfavla_motion/75k.pth \
      --model_type flower_actor_3d_motion_latent --backbone florence2 \
      --motion_expert_config motion_expert/configs/libero_florence2.yaml \
      --motion_expert_checkpoint train_logs/motion_expert/libero_florence2/B32_18k.pt \
      --out_dir vis_outputs/libero_task_rollouts/libero_spatial
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_evaluation_libero.evaluate_policy import (
    _prepare_libero_imports,
    _validate_control_setup,
    _validate_image_setup,
    parse_arguments,
)
from utils.common_utils import str2bool


def str2bool_arg(value):
    if isinstance(value, bool):
        return value
    return str2bool(value)


def parse_rollout_args():
    """Parse rollout-only args and strip them so parse_arguments() sees the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Match a task by name/language (case-insensitive substring). Overrides --task_id.")
    parser.add_argument("--init_state_index", type=int, default=0)
    parser.add_argument("--camera", choices=["agentview"], default="agentview",
                        help="Camera recorded for the static motion-viz (only agentview supported).")
    parser.add_argument("--out_dir", type=Path, default=Path("vis_outputs/libero_task_rollouts"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stop_on_success", type=str2bool_arg, default=True)
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--save_depth_npz", type=str2bool_arg, default=True)
    parser.add_argument("--depth_min", type=float, default=None)
    parser.add_argument("--depth_max", type=float, default=None)
    parser.add_argument("--depth_percentile_low", type=float, default=1.0)
    parser.add_argument("--depth_percentile_high", type=float, default=99.0)
    parser.add_argument("--invert_depth", type=str2bool_arg, default=False)
    rollout_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return rollout_args


def slugify(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_") or "task"


def resolve_task_id(benchmark, rollout_args, args):
    n = benchmark.n_tasks
    if rollout_args.task_name is not None:
        needle = rollout_args.task_name.strip().lower()
        matches = []
        for tid in range(n):
            task = benchmark.get_task(tid)
            if needle in str(task.name).lower() or needle in str(task.language).lower():
                matches.append(tid)
        if not matches:
            avail = "\n".join(
                f"  [{tid}] {benchmark.get_task(tid).name} :: {benchmark.get_task(tid).language}"
                for tid in range(n)
            )
            raise ValueError(f"No LIBERO task matches --task_name {rollout_args.task_name!r}.\n{avail}")
        if len(matches) > 1:
            opts = ", ".join(str(m) for m in matches)
            print(f"[rollout] --task_name {rollout_args.task_name!r} matched task_ids {opts}; using {matches[0]}.")
        return matches[0]
    if args.task_id is None or args.task_id < 0:
        raise ValueError("Provide --task_id (>=0) or --task_name to select a LIBERO task.")
    if args.task_id >= n:
        raise ValueError(f"--task_id {args.task_id} out of range [0, {n}).")
    return args.task_id


def depth_to_video_frames(depth_frames, rollout_args):
    depth = np.stack([np.squeeze(f) for f in depth_frames]).astype(np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("Depth frames contain no finite values.")
    lo = rollout_args.depth_min
    hi = rollout_args.depth_max
    if lo is None:
        lo = float(np.percentile(finite, rollout_args.depth_percentile_low))
    if hi is None:
        hi = float(np.percentile(finite, rollout_args.depth_percentile_high))
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    if rollout_args.invert_depth:
        normalized = 1.0 - normalized
    gray = (normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1), {"depth_min": lo, "depth_max": hi}


def write_video(path, frames, fps):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(1, fps)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame)[:, :, ::-1])  # RGB -> BGR
    finally:
        writer.release()
    return path


def record_agentview(obs_list, env, cam_idx=0):
    """Return (rgb HxWx3 uint8, metric depth HxW float32) for env 0, agentview camera.

    Uses the same orientation convention (vertical flip) as training/eval
    `extract_rgb_depth`, so the saved intrinsics/extrinsics match the images.
    """
    from online_evaluation_libero.utils_with_libero import (
        convert_depth_batch_to_meters,
        extract_rgb_depth,
    )

    rgb_batch, depth_batch = extract_rgb_depth([obs_list[0]])      # (1,ncam,3,H,W), (1,ncam,H,W)
    depth_batch = convert_depth_batch_to_meters(env, depth_batch)  # metric meters
    rgb = np.ascontiguousarray(rgb_batch[0, cam_idx].transpose(1, 2, 0).astype(np.uint8))
    depth = np.ascontiguousarray(depth_batch[0, cam_idx].astype(np.float32))
    return rgb, depth


@torch.no_grad()
def main():
    rollout_args = parse_rollout_args()
    args = parse_arguments()

    _prepare_libero_imports(args)
    _validate_image_setup(args)
    _validate_control_setup(args)

    from online_evaluation_libero.utils_with_libero import (
        build_benchmark,
        create_vectorized_env,
        get_static_camera_matrices,
        load_init_states,
    )

    benchmark = build_benchmark(args.benchmark, args.task_order_index)

    if rollout_args.list_tasks:
        for tid in range(benchmark.n_tasks):
            task = benchmark.get_task(tid)
            print(f"[{tid}] {task.name} :: {task.language}")
        return

    if (not args.use_circle_policy) and args.checkpoint is None:
        raise ValueError("--checkpoint is required to roll out a policy.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_id = resolve_task_id(benchmark, rollout_args, args)
    task = benchmark.get_task(task_id)
    instruction = task.language
    print(f"[rollout] benchmark={args.benchmark} task_id={task_id} name={task.name}")
    print(f"[rollout] instruction={instruction!r}")

    from online_evaluation_libero.model_wrapper import create_model

    model = create_model(args)

    controller_updates = None
    if args.libero_controller == "TCP_TARGET_POSE":
        controller_updates = {
            "target_mode": args.libero_tcp_target_mode,
            "target_pos_tol": args.libero_target_pos_tol,
            "target_ori_tol": args.libero_target_ori_tol,
            "target_max_steps": args.libero_target_max_steps,
            "absolute_gripper_openness": False,
        }
    env = create_vectorized_env(
        task, args.img_h, args.img_w, 1,
        controller=args.libero_controller,
        controller_config_updates=controller_updates,
    )
    env.seed(args.seed)

    try:
        init_states = load_init_states(task)
        idx = int(rollout_args.init_state_index)
        if idx < 0 or idx >= init_states.shape[0]:
            raise ValueError(
                f"--init_state_index {idx} out of range [0, {init_states.shape[0]}) for this task."
            )
        env.reset()
        obs = env.set_init_state(init_states[[idx]])

        action_dim = env.workers[0].env.env.action_dim
        for _ in range(int(rollout_args.settle_steps)):
            obs, _, _, _ = env.step(np.zeros((1, action_dim), dtype=np.float32))

        intrinsics_cache, agent_extrinsics_cache = get_static_camera_matrices(
            env, args.img_h, args.img_w
        )

        if hasattr(model, "reset"):
            model.reset()

        rgb_frames, depth_frames = [], []
        rgb, depth = record_agentview(obs, env)
        rgb_frames.append(rgb)
        depth_frames.append(depth)

        success = False
        env_steps = 0
        max_queries = int(math.ceil(float(args.max_steps) / float(args.prediction_len)))
        for _ in range(max_queries):
            model_input = model.preprocess(obs, env, intrinsics_cache, agent_extrinsics_cache)
            action_chunk = model.step(model_input, instruction).astype(np.float32)  # (1, T, D)
            for act_ind in range(action_chunk.shape[1]):
                if env_steps >= args.max_steps:
                    break
                obs, _, done, _ = env.step(action_chunk[:, act_ind, :])
                env_steps += 1
                rgb, depth = record_agentview(obs, env)
                rgb_frames.append(rgb)
                depth_frames.append(depth)
                if bool(done[0]):
                    success = True
                    break
            if success and rollout_args.stop_on_success:
                break
            if env_steps >= args.max_steps:
                break

        if not rgb_frames:
            raise RuntimeError("No frames were recorded.")

        out_dir = rollout_args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"task{task_id:02d}_{slugify(task.name)}_{rollout_args.camera}"
        rgb_path = out_dir / f"{stem}_rgb.mp4"
        depth_path = out_dir / f"{stem}_depth.mp4"
        summary_path = out_dir / f"{stem}_summary.json"

        write_video(rgb_path, rgb_frames, rollout_args.fps)
        depth_video_frames, depth_scale = depth_to_video_frames(depth_frames, rollout_args)
        write_video(depth_path, depth_video_frames, rollout_args.fps)

        depth_npz_path = None
        if rollout_args.save_depth_npz:
            depth_npz_path = out_dir / f"{stem}_depth_raw.npz"
            np.savez_compressed(
                depth_npz_path,
                depth=np.stack([np.squeeze(f) for f in depth_frames]).astype(np.float32),
            )

        # agentview is camera index 0 in utils_with_libero.CAMERA_NAMES
        agent_intrinsics = np.asarray(intrinsics_cache[0, 0], dtype=np.float32)
        agent_c2w = np.asarray(agent_extrinsics_cache[0], dtype=np.float32)
        summary = {
            "benchmark": args.benchmark,
            "task_order_index": args.task_order_index,
            "task_id": task_id,
            "task_name": task.name,
            "instruction": instruction,
            "init_state_index": idx,
            "success": bool(success),
            "num_steps": env_steps,
            "camera": rollout_args.camera,
            "image_vertically_flipped": True,
            "num_rgb_frames": len(rgb_frames),
            "num_depth_frames": len(depth_frames),
            "fps": rollout_args.fps,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "rgb_video": str(rgb_path),
            "depth_video": str(depth_path),
            "raw_depth_npz": str(depth_npz_path) if depth_npz_path is not None else None,
            "agentview_intrinsics": agent_intrinsics.tolist(),
            "agentview_extrinsics_c2w": agent_c2w.tolist(),
            **depth_scale,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[rollout] success={success} steps={env_steps}")
        print(f"[rollout] saved rgb video: {rgb_path}")
        print(f"[rollout] saved depth video: {depth_path}")
        if depth_npz_path is not None:
            print(f"[rollout] saved raw depth npz: {depth_npz_path}")
        print(f"[rollout] saved summary: {summary_path}")
    finally:
        env.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
