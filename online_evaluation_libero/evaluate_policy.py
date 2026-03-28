"""Online evaluation script for LIBERO."""
import argparse, random, sys, numpy as np, torch
from pathlib import Path

# Ensure local package imports resolve to this repository.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.common_utils import round_floats, str2bool, str_none


def parse_arguments():
    parser = argparse.ArgumentParser("Evaluate policy on LIBERO simulator")
    arguments = [
        # Setup arguments
        ("libero_root", str, "/data/user_data/ziyinx/LIBERO"),
        ("seed", int, 0),
        # Task arguments
        ("benchmark", str, "libero_10"),
        ("task_order_index", int, 0),
        ("task_id", int, -1),
        ("all_tasks", str2bool, False),
        ("num_eval", int, 20),
        ("max_steps", int, 600),
        ("img_h", int, 128),
        ("img_w", int, 128),
        ("use_mp", str2bool, False),
        # Logging arguments
        ("output_file", Path, Path("online_evaluation_libero/eval.json")),
        ("save_video", str2bool, False),  # placeholder for compatibility
        # Model checkpoint
        ("checkpoint", str_none, None),
        # Model arguments
        ("model_type", str, "qwen_flower_actor_3d"),
        ("prediction_len", int, 10),
        ("num_history", int, 1),
        ("pre_tokenize", str2bool, True),
        ("custom_img_size", int, 128),
        ("backbone", str, "qwen3vl"),
        ("fps_subsampling_factor", int, 1),
        ("embedding_dim", int, 1024),
        ("num_attn_heads", int, 16),
        ("num_vis_instr_attn_layers", int, 3),
        ("num_shared_attn_layers", int, 16),
        ("relative_action", str2bool, True),
        ("rotation_format", str, "axisangle"),
        ("denoise_timesteps", int, 5),
        ("denoise_model", str, "rectified_flow"),
    ]
    for name, typ, default in arguments:
        parser.add_argument(f"--{name}", type=typ, default=default)
    return parser.parse_args()


def _prepare_libero_imports(args):
    if args.libero_root is None:
        args.libero_root = str(Path(__file__).resolve().parents[2] / "LIBERO")
    if args.libero_root not in sys.path:
        sys.path.insert(0, args.libero_root)


@torch.no_grad()
def evaluate_one_task(args, model, task):
    from online_evaluation_libero.utils_with_libero import (
        create_vectorized_env,
        get_static_camera_matrices,
        load_init_states,
    )

    env_num = args.num_eval
    env = create_vectorized_env(task, args.img_h, args.img_w, env_num)
    env.seed(args.seed)

    init_states = load_init_states(task)
    indices = np.arange(env_num) % init_states.shape[0]
    init_states_batch = init_states[indices]

    env.reset()
    obs = env.set_init_state(init_states_batch)

    # Physics settling warmup.
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((env_num, 7), dtype=np.float32))

    # Cache static matrices once per rollout.
    intrinsics_cache, agent_extrinsics_cache = get_static_camera_matrices(
        env, args.img_h, args.img_w
    )

    dones = [False] * env_num
    instruction = task.language

    for _ in range(args.max_steps):
        model_input = model.preprocess(
            obs, env, intrinsics_cache, agent_extrinsics_cache
        )
        actions = model.step(model_input, instruction).astype(np.float32)
        obs, _, done, _ = env.step(actions)

        for k in range(env_num):
            dones[k] = dones[k] or bool(done[k])
        if all(dones):
            break

    success_rate = float(np.mean(np.array(dones, dtype=np.float32)))
    env.close()
    return success_rate


@torch.no_grad()
def main(args):
    _prepare_libero_imports(args)

    from online_evaluation_libero.model_wrapper import create_model
    from online_evaluation_libero.utils_with_libero import build_benchmark, save_results

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args)

    benchmark = build_benchmark(args.benchmark, args.task_order_index)
    task_ids = (
        list(range(benchmark.n_tasks))
        if args.all_tasks or args.task_id < 0
        else [args.task_id]
    )

    results = {
        "benchmark": args.benchmark,
        "task_order_index": args.task_order_index,
        "checkpoint": args.checkpoint,
        "num_eval": args.num_eval,
        "max_steps": args.max_steps,
        "results": {},
    }

    for task_id in task_ids:
        task = benchmark.get_task(task_id)
        print(f"[eval] task_id={task_id} name={task.name} lang='{task.language}'")
        success_rate = evaluate_one_task(args, model, task)
        results["results"][str(task_id)] = {
            "task_name": task.name,
            "language": task.language,
            "success_rate": success_rate,
        }
        print(f"[eval] task_id={task_id} success_rate={success_rate:.4f}")

    if results["results"]:
        results["mean_success_rate"] = float(
            np.mean([x["success_rate"] for x in results["results"].values()])
        )
    else:
        results["mean_success_rate"] = 0.0

    save_results(str(args.output_file), round_floats(results))
    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main(parse_arguments())
