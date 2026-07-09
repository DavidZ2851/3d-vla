"""Roll out one CALVIN task with flower_actor_3d_motion_latent and save videos.

Example:
  python -m online_evaluation_calvin.rollout_task_video \
      --task open_drawer \
      --checkpoint train_logs/calvin/.../last.pth \
      --model_type flower_actor_3d_motion_latent \
      --backbone florence2 \
      --custom_img_size 224 \
      --motion_expert_config motion_expert/configs/calvin_florence2.yaml \
      --motion_expert_checkpoint train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from online_evaluation_calvin.evaluate_policy import NUM_SEQUENCES, parse_arguments
from online_evaluation_calvin.model_wrapper import create_model
from online_evaluation_calvin.multistep_sequences import get_sequences
from online_evaluation_calvin.utils_with_calvin import (
    get_env,
    get_env_state_for_initial_condition,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}.")


def parse_rollout_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default="open_drawer")
    parser.add_argument("--list_tasks", action="store_true")
    parser.add_argument("--sequence_index", type=int, default=None)
    parser.add_argument("--subtask_index", type=int, default=0)
    parser.add_argument("--task_occurrence", type=int, default=0)
    parser.add_argument("--num_sequences", type=int, default=NUM_SEQUENCES)
    parser.add_argument("--sequence_workers", type=int, default=1)
    parser.add_argument("--run_prefix", type=str2bool, default=True)
    parser.add_argument("--record_prefix", type=str2bool, default=False)
    parser.add_argument("--require_prefix_success", type=str2bool, default=True)
    parser.add_argument("--stop_on_success", type=str2bool, default=True)
    parser.add_argument("--max_replans", type=int, default=None)
    parser.add_argument("--camera", choices=["static", "gripper"], default="static")
    parser.add_argument("--out_dir", type=Path, default=Path("vis_outputs/calvin_task_rollouts"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--depth_min", type=float, default=None)
    parser.add_argument("--depth_max", type=float, default=None)
    parser.add_argument("--depth_percentile_low", type=float, default=1.0)
    parser.add_argument("--depth_percentile_high", type=float, default=99.0)
    parser.add_argument("--invert_depth", type=str2bool, default=False)
    parser.add_argument("--save_depth_npz", type=str2bool, default=False)
    rollout_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return rollout_args


def slugify(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_") or "rollout"


def get_camera_obs(raw_obs, camera):
    suffix = "static" if camera == "static" else "gripper"
    rgb = raw_obs["rgb_obs"][f"rgb_{suffix}"]
    depth = raw_obs["depth_obs"][f"depth_{suffix}"]
    return np.asarray(rgb), np.asarray(depth)


def append_frames(raw_obs, camera, rgb_frames, depth_frames):
    rgb, depth = get_camera_obs(raw_obs, camera)
    rgb_frames.append(np.asarray(rgb, dtype=np.uint8).copy())
    depth_frames.append(np.asarray(depth, dtype=np.float32).copy())


def execute_action(env, model, action):
    exec_mode = getattr(model, "calvin_action_exec", "cartesian_abs")
    if exec_mode == "relative_array":
        step_action = np.asarray(action, dtype=np.float64).reshape(7)
        step_action[6] = 1 if step_action[6] > 0 else -1
        raw_obs, _, _, current_info = env.step(step_action)
        return raw_obs, current_info
    if exec_mode not in ("cartesian_abs", "cartesian_rel"):
        raise ValueError(
            f"Unknown calvin_action_exec={exec_mode!r}; expected "
            "'cartesian_abs', 'cartesian_rel', or 'relative_array'."
        )
    curr_action = {
        "action": [
            action[:3],
            action[3:6],
            action[[6]],
        ],
        "type": exec_mode,
    }
    raw_obs, _, _, current_info = env.step(curr_action)
    return raw_obs, current_info


@torch.no_grad()
def rollout_once(
    env,
    model,
    task_oracle,
    subtask,
    instruction,
    max_replans,
    camera,
    record_video,
    stop_on_success,
):
    model.reset()
    start_info = env.get_info()
    raw_obs = env.get_obs()
    rgb_frames, depth_frames = [], []
    if record_video:
        append_frames(raw_obs, camera, rgb_frames, depth_frames)

    for replan in range(max_replans):
        obs = model.preprocess(raw_obs, env)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory = model.step(obs, instruction)

        for act_ind in range(trajectory.shape[1]):
            raw_obs, current_info = execute_action(env, model, trajectory[0, act_ind])
            if record_video:
                append_frames(raw_obs, camera, rgb_frames, depth_frames)

            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )
            if stop_on_success and len(current_task_info) > 0:
                return True, rgb_frames, depth_frames, replan + 1

    final_info = env.get_info()
    current_task_info = task_oracle.get_task_info_for_set(
        start_info, final_info, {subtask}
    )
    return bool(current_task_info), rgb_frames, depth_frames, max_replans


def find_task_sequence(eval_sequences, task, occurrence, prefer_first=True):
    matches = []
    if prefer_first:
        for seq_ind, (_, eval_sequence) in enumerate(eval_sequences):
            if len(eval_sequence) > 0 and eval_sequence[0] == task:
                matches.append((seq_ind, 0))
    for seq_ind, (_, eval_sequence) in enumerate(eval_sequences):
        for subtask_ind, subtask in enumerate(eval_sequence):
            if subtask == task and (seq_ind, subtask_ind) not in matches:
                matches.append((seq_ind, subtask_ind))
    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"Task {task!r} appears {len(matches)} times in the first "
            f"{len(eval_sequences)} generated CALVIN sequences; requested "
            f"--task_occurrence {occurrence}."
        )
    return matches[occurrence]


def choose_rollout(eval_sequences, rollout_args, instr_dict):
    if rollout_args.sequence_index is not None:
        seq_ind = rollout_args.sequence_index
        if seq_ind < 0 or seq_ind >= len(eval_sequences):
            raise ValueError(
                f"--sequence_index must be in [0, {len(eval_sequences) - 1}], "
                f"got {seq_ind}."
            )
        _, eval_sequence = eval_sequences[seq_ind]
        subtask_ind = rollout_args.subtask_index
        if subtask_ind < 0 or subtask_ind >= len(eval_sequence):
            raise ValueError(
                f"--subtask_index must be in [0, {len(eval_sequence) - 1}], "
                f"got {subtask_ind}."
            )
        return seq_ind, subtask_ind

    if rollout_args.task not in instr_dict:
        available = ", ".join(sorted(instr_dict.keys()))
        raise ValueError(f"Unknown CALVIN task {rollout_args.task!r}. Available: {available}")
    return find_task_sequence(
        eval_sequences,
        rollout_args.task,
        rollout_args.task_occurrence,
    )


def depth_to_video_frames(depth_frames, args):
    depth = np.stack([np.squeeze(frame) for frame in depth_frames]).astype(np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("Depth frames contain no finite values.")

    lo = args.depth_min
    hi = args.depth_max
    if lo is None:
        lo = float(np.percentile(finite, args.depth_percentile_low))
    if hi is None:
        hi = float(np.percentile(finite, args.depth_percentile_high))
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    if args.invert_depth:
        normalized = 1.0 - normalized
    gray = (normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1), {"depth_min": lo, "depth_max": hi}


def write_video(path, frames, fps):
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip([np.ascontiguousarray(frame) for frame in frames], fps=fps)
    try:
        clip.write_videofile(str(path), codec="libx264", audio=False)
    finally:
        close = getattr(clip, "close", None)
        if close is not None:
            close()
    return path


def set_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This rollout script expects CUDA; Model.step moves tokens with .cuda().")
    args.local_rank = 0
    args.device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    rollout_args = parse_rollout_args()
    args = parse_arguments()

    instr_dict = OmegaConf.load(args.ann_config_file)
    if rollout_args.list_tasks:
        for task in sorted(instr_dict.keys()):
            print(task)
        return

    set_runtime(args)

    eval_sequences = get_sequences(
        rollout_args.num_sequences,
        num_workers=rollout_args.sequence_workers,
    )
    seq_ind, subtask_ind = choose_rollout(eval_sequences, rollout_args, instr_dict)
    initial_state, eval_sequence = eval_sequences[seq_ind]
    target_task = eval_sequence[subtask_ind]
    target_instruction = instr_dict[target_task][0]
    prefix_tasks = list(eval_sequence[:subtask_ind])
    max_replans = (
        rollout_args.max_replans
        if rollout_args.max_replans is not None
        else max(1, 360 // int(args.pred_len))
    )

    print(f"[rollout] sequence_index={seq_ind} subtask_index={subtask_ind}")
    print(f"[rollout] target={target_task!r} instruction={target_instruction!r}")
    if prefix_tasks:
        print(f"[rollout] prefix tasks: {prefix_tasks}")

    model = create_model(args)
    env = get_env(args.merged_config_file, show_gui=False)
    task_oracle = hydra.utils.instantiate(OmegaConf.load(args.task_config_file))

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    all_rgb_frames, all_depth_frames = [], []
    prefix_results = []
    try:
        if prefix_tasks and rollout_args.run_prefix:
            for prefix_task in prefix_tasks:
                prefix_instruction = instr_dict[prefix_task][0]
                print(f"[rollout] running prefix task {prefix_task!r}")
                success, rgb, depth, replans = rollout_once(
                    env,
                    model,
                    task_oracle,
                    prefix_task,
                    prefix_instruction,
                    max_replans,
                    rollout_args.camera,
                    rollout_args.record_prefix,
                    rollout_args.stop_on_success,
                )
                prefix_results.append({
                    "task": prefix_task,
                    "instruction": prefix_instruction,
                    "success": bool(success),
                    "replans": int(replans),
                })
                if rollout_args.record_prefix:
                    all_rgb_frames.extend(rgb)
                    all_depth_frames.extend(depth)
                print(f"[rollout] prefix {prefix_task!r} success={success}")
                if not success and rollout_args.require_prefix_success:
                    raise RuntimeError(
                        f"Prefix task {prefix_task!r} failed; target task "
                        f"{target_task!r} was not rolled out."
                    )

        success, rgb_frames, depth_frames, replans = rollout_once(
            env,
            model,
            task_oracle,
            target_task,
            target_instruction,
            max_replans,
            rollout_args.camera,
            True,
            rollout_args.stop_on_success,
        )
        all_rgb_frames.extend(rgb_frames)
        all_depth_frames.extend(depth_frames)

        if not all_rgb_frames or not all_depth_frames:
            raise RuntimeError("No frames were recorded.")

        rollout_args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"seq{seq_ind:04d}_sub{subtask_ind}_{slugify(target_task)}_"
            f"{rollout_args.camera}"
        )
        rgb_path = rollout_args.out_dir / f"{stem}_rgb.mp4"
        depth_path = rollout_args.out_dir / f"{stem}_depth.mp4"
        summary_path = rollout_args.out_dir / f"{stem}_summary.json"

        write_video(rgb_path, all_rgb_frames, rollout_args.fps)
        depth_video_frames, depth_scale = depth_to_video_frames(all_depth_frames, rollout_args)
        write_video(depth_path, depth_video_frames, rollout_args.fps)
        depth_npz_path = None
        if rollout_args.save_depth_npz:
            depth_npz_path = rollout_args.out_dir / f"{stem}_depth_raw.npz"
            np.savez_compressed(
                depth_npz_path,
                depth=np.stack([np.squeeze(frame) for frame in all_depth_frames]).astype(np.float32),
            )

        summary = {
            "sequence_index": seq_ind,
            "subtask_index": subtask_ind,
            "target_task": target_task,
            "instruction": target_instruction,
            "success": bool(success),
            "target_replans": int(replans),
            "prefix_results": prefix_results,
            "camera": rollout_args.camera,
            "num_rgb_frames": len(all_rgb_frames),
            "num_depth_frames": len(all_depth_frames),
            "fps": rollout_args.fps,
            "rgb_video": str(rgb_path),
            "depth_video": str(depth_path),
            "raw_depth_npz": str(depth_npz_path) if depth_npz_path is not None else None,
            **depth_scale,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[rollout] target success={success}")
        print(f"[rollout] saved rgb video: {rgb_path}")
        print(f"[rollout] saved depth video: {depth_path}")
        print(f"[rollout] saved summary: {summary_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
