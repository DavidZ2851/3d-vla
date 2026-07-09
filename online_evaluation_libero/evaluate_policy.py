"""Online evaluation script for LIBERO."""
import argparse, math, random, sys, numpy as np, torch
from pathlib import Path

# Ensure local package imports resolve to this repository.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.common_utils import round_floats, str2bool, str_none


LIBERO_RGBD_SIZE = 224


def parse_arguments():
    parser = argparse.ArgumentParser("Evaluate policy on LIBERO simulator")
    arguments = [
        # Setup arguments
        ("libero_root", str, "/data/user_data/ziyinx/LIBERO"),
        ("libero_variant", str, "standard"),
        ("seed", int, 0),
        # Task arguments
        ("benchmark", str, "libero_spatial"),
        ("task_order_index", int, 0),
        ("task_id", int, -1),
        ("all_tasks", str2bool, False),
        ("task_start", int, 0),
        ("task_end", int, -1),
        ("num_eval", int, 20),
        ("init_state_index_offset", int, 0),
        ("max_steps", int, 600),
        ("auto_max_steps", str2bool, False),
        ("aggregate_plus_categories", str2bool, False),
        ("img_h", int, LIBERO_RGBD_SIZE),
        ("img_w", int, LIBERO_RGBD_SIZE),
        ("use_mp", str2bool, False),
        # Logging arguments
        ("output_file", Path, Path("online_evaluation_libero/eval.json")),
        ("save_video", str2bool, False),
        ("video_dir", str, "eval_logs/libero_videos"),
        ("video_fps", int, 20),
        ("video_env_index", int, 0),
        # Model checkpoint
        ("checkpoint", str_none, None),
        # Fake trajectory sanity-check mode
        ("use_circle_policy", str2bool, False),
        ("circle_radius", float, 0.03),
        ("circle_omega", float, 0.20),
        ("circle_z_bias", float, 0.0),
        ("circle_keep_openness", str2bool, True),
        ("circle_fixed_openness", float, 0.02),
        # Model arguments
        ("model_type", str, "qwen_flower_actor_3d"),
        ("prediction_len", int, 10),
        ("num_history", int, 1),
        ("pre_tokenize", str2bool, True),
        ("custom_img_size", int, LIBERO_RGBD_SIZE),
        ("backbone", str, "qwen3vl"),
        ("num_finetune_vision_layers", int, 0),
        ("vision_lora_rank", int, 0),
        ("vision_lora_alpha", int, 32),
        ("vision_lora_dropout", float, 0.05),
        ("vision_lora_num_layers", int, 0),
        ("vision_lora_target_modules", str, "attn.qkv,attn.proj"),
        ("llm_lora_rank", int, 0),
        ("llm_lora_alpha", int, 32),
        ("llm_lora_dropout", float, 0.05),
        ("llm_lora_num_layers", int, 0),
        ("llm_lora_target_modules", str, "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"),
        ("fps_subsampling_factor", int, 1),
        ("embedding_dim", int, 1024),
        ("num_attn_heads", int, 16),
        ("num_vis_instr_attn_layers", int, 3),
        ("num_shared_attn_layers", int, 16),
        ("use_multiscale", str2bool, False),
        ("gradient_checkpointing", str2bool, False),
        ("relative_action", str2bool, True),
        ("rotation_format", str, "axisangle"),
        ("libero_controller", str, "OSC_POSE"),
        ("libero_tcp_target_mode", str, "absolute"),
        ("libero_target_pos_tol", float, 0.001),
        ("libero_target_ori_tol", float, 0.01),
        ("libero_target_max_steps", int, 250),
        ("denoise_timesteps", int, 5),
        ("denoise_model", str, "rectified_flow"),
        # Model arguments: motion latent (flower_actor_3d_motion_latent)
        ("motion_expert_config", str_none, None),
        ("motion_expert_checkpoint", str_none, None),
        ("motion_solver_steps", int, 10),
        ("motion_steps_to_run", int, 1),
        ("motion_finetune", str2bool, False),
        ("motion_scene_fusion", str2bool, True),
        ("motion_action_fusion", str2bool, False),
    ]
    for name, typ, default in arguments:
        parser.add_argument(f"--{name}", type=typ, default=default)
    return parser.parse_args()


class CircleTrajectoryPolicy:
    """Fixed controller that moves TCP in a circle around rollout-start pose."""

    gripper_open_threshold = 0.02

    def __init__(self, args):
        self.radius = float(args.circle_radius)
        self.omega = float(args.circle_omega)
        self.z_bias = float(args.circle_z_bias)
        self.keep_openness = bool(args.circle_keep_openness)
        self.fixed_openness = float(args.circle_fixed_openness)
        self.step_idx = 0
        self.anchor_pos = None
        self.anchor_quat = None
        self.anchor_open = None

    def reset(self):
        self.step_idx = 0
        self.anchor_pos = None
        self.anchor_quat = None
        self.anchor_open = None

    def _init_anchor(self, obs_list):
        poses = []
        quats = []
        opens = []
        for obs in obs_list:
            poses.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            quats.append(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
            opens.append(abs(float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[0])))
        self.anchor_pos = np.stack(poses, axis=0)
        self.anchor_quat = np.stack(quats, axis=0)
        self.anchor_open = np.asarray(opens, dtype=np.float32)

    def step(self, obs_list):
        if self.anchor_pos is None:
            self._init_anchor(obs_list)
        theta = self.step_idx * self.omega
        dx = self.radius * np.cos(theta)
        dy = self.radius * np.sin(theta)
        dz = self.z_bias

        target_pos = self.anchor_pos.copy()
        target_pos[:, 0] += dx
        target_pos[:, 1] += dy
        target_pos[:, 2] += dz

        if self.keep_openness:
            gripper_command = np.where(
                self.anchor_open[:, None] > self.gripper_open_threshold,
                -1.0,
                1.0,
            ).astype(np.float32)
        else:
            gripper_command = np.full(
                (self.anchor_pos.shape[0], 1),
                -1.0 if self.fixed_openness > self.gripper_open_threshold else 1.0,
                dtype=np.float32,
            )

        self.step_idx += 1
        return np.concatenate([target_pos, self.anchor_quat, gripper_command], axis=1).astype(np.float32)


def _validate_image_setup(args):
    expected_hw = (LIBERO_RGBD_SIZE, LIBERO_RGBD_SIZE)
    actual_hw = (int(args.img_h), int(args.img_w))
    if actual_hw != expected_hw:
        raise ValueError(
            "LIBERO online evaluation expects 224x224 RGBD for agentview and wrist, "
            f"got img_h/img_w={actual_hw}."
        )
    if int(args.custom_img_size) != LIBERO_RGBD_SIZE:
        raise ValueError(
            "LIBERO online evaluation expects custom_img_size=224 to match the "
            f"224x224 RGBD inputs, got {args.custom_img_size}."
        )


def _validate_control_setup(args):
    if args.relative_action and args.rotation_format != "axisangle":
        raise ValueError(
            "LIBERO relative_action=True expects rotation_format=axisangle "
            f"for the unnormalized OSC delta rel_action, got {args.rotation_format!r}."
        )


def _record_video_if_needed(frames, args, task_id, task_name):
    if not args.save_video or len(frames) == 0:
        return None
    try:
        import cv2
    except ImportError as exc:
        print(f"[warn] save_video=True but OpenCV not available: {exc}")
        return None

    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(task_name))
    out_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"task{task_id}_{safe_name}.mp4"

    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.video_fps),
        (w, h),
    )
    if not writer.isOpened():
        print(f"[warn] Failed to open video writer: {out_path}")
        return None
    for frame in frames:
        writer.write(frame[:, :, ::-1])  # RGB -> BGR
    writer.release()
    return str(out_path)


def _prepare_libero_imports(args):
    from online_evaluation_libero.libero_runtime import infer_max_steps, prepare_libero_runtime

    if args.libero_root is None:
        args.libero_root = str(Path(__file__).resolve().parents[2] / "LIBERO")
    prepare_libero_runtime(args.libero_root)
    if args.auto_max_steps:
        args.max_steps = infer_max_steps(args.benchmark, default=args.max_steps)


def _select_init_state_indices(num_eval, num_init_states, offset=0):
    if num_init_states <= 0:
        raise ValueError("LIBERO task has no saved init states to select from.")
    start = int(offset)
    stop = start + int(num_eval)
    if start < 0 or stop > int(num_init_states):
        raise ValueError(
            "Requested LIBERO init-state range "
            f"[{start}, {stop}) is outside available range [0, {num_init_states})."
        )
    return np.arange(start, stop, dtype=np.int64)


@torch.no_grad()
def evaluate_one_task(args, model, task, task_id, benchmark):
    from online_evaluation_libero.utils_with_libero import (
        create_vectorized_env,
        get_static_camera_matrices,
        load_init_states,
    )

    env_num = args.num_eval
    controller_updates = None
    if args.libero_controller == "TCP_TARGET_POSE":
        controller_updates = {
            "target_mode": args.libero_tcp_target_mode,
            "target_pos_tol": args.libero_target_pos_tol,
            "target_ori_tol": args.libero_target_ori_tol,
            "target_max_steps": args.libero_target_max_steps,
            "absolute_gripper_openness": False,
        }
    init_states = load_init_states(task, benchmark=benchmark, task_id=task_id)
    if init_states.ndim == 1:
        init_states = init_states.unsqueeze(0)
    available_inits = int(init_states.shape[0])
    if env_num > available_inits:
        print(
            f"[warn] task_id={task_id} requested num_eval={env_num} but only "
            f"{available_inits} init states are available; using {available_inits}."
        )
        env_num = available_inits
    env = create_vectorized_env(
        task,
        args.img_h,
        args.img_w,
        env_num,
        controller=args.libero_controller,
        controller_config_updates=controller_updates,
    )
    env.seed(args.seed)

    indices = _select_init_state_indices(
        env_num,
        available_inits,
        offset=args.init_state_index_offset,
    )
    init_states_batch = init_states[indices]

    env.reset()
    obs = env.set_init_state(init_states_batch)

    # Physics settling warmup.
    action_dim = env.workers[0].env.env.action_dim
    for _ in range(30):
        obs, _, _, _ = env.step(np.zeros((env_num, action_dim), dtype=np.float32))

    # Cache static matrices once per rollout.
    intrinsics_cache, agent_extrinsics_cache = get_static_camera_matrices(
        env, args.img_h, args.img_w
    )

    dones = [False] * env_num
    instruction = task.language
    circle_policy = CircleTrajectoryPolicy(args) if args.use_circle_policy else None
    if circle_policy is not None:
        if args.libero_controller != "TCP_TARGET_POSE":
            raise ValueError("Circle policy requires --libero_controller TCP_TARGET_POSE")
        circle_policy.reset()
    video_frames = []
    video_env_idx = max(0, min(int(args.video_env_index), env_num - 1))
    if args.save_video:
        frame = np.asarray(obs[video_env_idx]["agentview_image"])[::-1]
        wrist = np.asarray(obs[video_env_idx]["robot0_eye_in_hand_image"])[::-1]
        video_frames.append(np.concatenate([frame, wrist], axis=1))

    max_policy_queries = int(math.ceil(float(args.max_steps) / float(args.prediction_len)))
    env_steps_taken = 0
    for _ in range(max_policy_queries):
        if circle_policy is not None:
            action_chunk = circle_policy.step(obs)[:, None, :]
        else:
            model_input = model.preprocess(
                obs, env, intrinsics_cache, agent_extrinsics_cache
            )
            action_chunk = model.step(model_input, instruction).astype(np.float32)

        for act_ind in range(action_chunk.shape[1]):
            if env_steps_taken >= args.max_steps:
                break
            obs, _, done, _ = env.step(action_chunk[:, act_ind, :])
            env_steps_taken += 1

            for k in range(env_num):
                dones[k] = dones[k] or bool(done[k])
            if all(dones):
                break

            if args.save_video:
                frame = np.asarray(obs[video_env_idx]["agentview_image"])[::-1]
                wrist = np.asarray(obs[video_env_idx]["robot0_eye_in_hand_image"])[::-1]
                video_frames.append(np.concatenate([frame, wrist], axis=1))

        if env_steps_taken >= args.max_steps or all(dones):
            break

    success_rate = float(np.mean(np.array(dones, dtype=np.float32)))
    successful_init_state_indices = [
        int(indices[k])
        for k, done in enumerate(dones)
        if done
    ]
    video_path = _record_video_if_needed(video_frames, args, task_id=task_id, task_name=task.name)
    env.close()
    return success_rate, video_path, successful_init_state_indices


@torch.no_grad()
def main(args):
    _prepare_libero_imports(args)
    _validate_image_setup(args)
    _validate_control_setup(args)

    from online_evaluation_libero.utils_with_libero import (
        aggregate_plus_categories,
        build_benchmark,
        save_results,
        select_task_ids,
    )

    if (not args.use_circle_policy) and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --use_circle_policy true")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not args.use_circle_policy:
        from online_evaluation_libero.model_wrapper import create_model
        model = create_model(args)

    benchmark = build_benchmark(args.benchmark, args.task_order_index)
    task_ids = select_task_ids(
        benchmark,
        all_tasks=args.all_tasks,
        task_id=args.task_id,
        task_start=args.task_start,
        task_end=args.task_end,
    )

    results = {
        "benchmark": args.benchmark,
        "libero_root": args.libero_root,
        "libero_variant": args.libero_variant,
        "task_order_index": args.task_order_index,
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "num_eval": args.num_eval,
        "init_state_index_offset": args.init_state_index_offset,
        "max_steps": args.max_steps,
        "task_start": args.task_start,
        "task_end": args.task_end if args.task_end >= 0 else benchmark.n_tasks,
        "results": {},
    }

    for task_id in task_ids:
        task = benchmark.get_task(task_id)
        print(f"[eval] task_id={task_id} name={task.name} lang='{task.language}'")
        success_rate, video_path, successful_init_state_indices = evaluate_one_task(
            args,
            model,
            task,
            task_id=task_id,
            benchmark=benchmark,
        )
        results["results"][str(task_id)] = {
            "task_name": task.name,
            "language": task.language,
            "success_rate": success_rate,
            "successful_init_state_indices": successful_init_state_indices,
        }
        if video_path is not None:
            results["results"][str(task_id)]["video_path"] = video_path
            print(f"[eval] task_id={task_id} video={video_path}")
        print(f"[eval] task_id={task_id} success_rate={success_rate:.4f}")

    if results["results"]:
        results["mean_success_rate"] = float(
            np.mean([x["success_rate"] for x in results["results"].values()])
        )
    else:
        results["mean_success_rate"] = 0.0

    if args.aggregate_plus_categories:
        category_rates = aggregate_plus_categories(
            args.benchmark,
            results["results"],
            args.libero_root,
        )
        if category_rates:
            results["category_success_rates"] = category_rates
            results["category_mean_success_rate"] = float(
                np.mean(list(category_rates.values()))
            )

    save_results(str(args.output_file), round_floats(results))
    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main(parse_arguments())
