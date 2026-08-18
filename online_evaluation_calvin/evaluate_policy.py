import argparse
import importlib
import os
import gc
import random
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from utils.common_utils import str2bool, str_none
from online_evaluation_calvin.model_wrapper import create_model
from online_evaluation_calvin.utils_with_calvin import (
    get_env,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from online_evaluation_calvin.multistep_sequences import get_sequences


logger = logging.getLogger(__name__)


EP_LEN = 36
NUM_SEQUENCES = 1000


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Online environment
        ('merged_config_file', Path, "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml"),
        ('task_config_file', Path, "online_evaluation_calvin/configs/new_playtable_tasks.yaml"),
        ('ann_config_file', Path, "online_evaluation_calvin/configs/new_playtable_validation.yaml"),
        # Testing arguments
        ('checkpoint', str_none, None),
        ('seed', int, 0),
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "eval_logs" / "calvin"),
        ('save_video', str2bool, False),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('pred_len', int, 10),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('num_finetune_vision_layers', int, 0),
        ('vision_lora_rank', int, 0),
        ('vision_lora_alpha', int, 32),
        ('vision_lora_dropout', float, 0.05),
        ('vision_lora_num_layers', int, 0),
        ('vision_lora_target_modules', str, "attn.qkv,attn.proj"),
        ('llm_lora_rank', int, 0),
        ('llm_lora_alpha', int, 32),
        ('llm_lora_dropout', float, 0.05),
        ('llm_lora_num_layers', int, 0),
        ('llm_lora_target_modules', str, "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"),
        ('fps_subsampling_factor', int, 5),
        ('use_multiscale', str2bool, False),
        # Model arguments: encoder and head
        ('embedding_dim', int, 144),
        ('num_attn_heads', int, 9),
        ('num_vis_instr_attn_layers', int, 2),
        # Model arguments: head
        ('num_shared_attn_layers', int, 16),
        ('relative_action', str2bool, True),
        ('rotation_format', str, 'euler'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        # Model arguments: motion latent (flower_actor_3d_motion_latent)
        ('motion_expert_config', str_none, None),
        ('motion_expert_checkpoint', str_none, None),
        ('motion_solver_steps', int, 10),
        ('motion_steps_to_run', int, 1),
        ('motion_finetune', str2bool, False),
        ('motion_scene_fusion', str2bool, True),
        ('motion_action_fusion', str2bool, False),
        # 'world' (default) or 'camera': the coordinate frame the policy was
        # trained in. See utils/calvin_camera_frame.
        # 'world' | 'camera' | 'yaw45' -- fixed frame the policy was trained in.
        ('coord_frame', str, 'world'),
        ('motion_expert_frame', str, 'world'),
        # Model arguments: joint-angle decoder (flower_actor_3d_motion_latent_joint)
        ('joint_action_dim', int, 7),
        ('joint_proprio_dim', int, 8),
        ('joint_num_layers', int, 4),
        ('joint_embedding_dim', int, 0),
        ('joint_num_attn_heads', int, 0),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def evaluate_policy(model, env, task_config_file, ann_config_file,
                    eval_log_dir=None, save_video=False, sequence_indices=[],
                    eval_sequences=None):
    """Run this function to evaluate a model on the CALVIN challenge.

    If ``eval_sequences`` is provided (list of ``(initial_state, subtask_list)``),
    it is used instead of ``get_sequences(NUM_SEQUENCES)``. Callers must pass
    ``sequence_indices`` consistent with ``len(eval_sequences)``.
    """
    # Load configs for tasks and instructions
    task_cfg = OmegaConf.load(task_config_file)
    task_oracle = hydra.utils.instantiate(task_cfg)
    instr_dict = OmegaConf.load(ann_config_file)

    # Create a directory to store results
    eval_log_dir = get_log_dir(eval_log_dir)

    # Sample sequences of 5 instructions (or use a caller-provided benchmark list)
    if eval_sequences is None:
        eval_sequences = get_sequences(NUM_SEQUENCES)

    # Load cached results (in case evaluation gets killed)
    results, tested_sequence_indices = collect_results(eval_log_dir)

    # Loop over all sampled test sequences
    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue  # trick to split evaluation across gpus
        if seq_ind in tested_sequence_indices:
            continue  # omit tested indices
        # Run the model on the sequence, querying one instruction at a time
        result, videos = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, instr_dict, save_video
        )
        # Store results on the logging file
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        # Print up-to-current results
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        # Optionally store videos
        if save_video:
            image_sequence_clip = importlib.import_module("moviepy.video.io.ImageSequenceClip")
            clip = []
            for video in videos:
                clip.extend(video)
            clip = image_sequence_clip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      instr_dict, save_video, first_only=False, verbose=True):
    """
    Evaluates a sequence of language instructions.

    Returns:
        success_counter: the (int) number of tasks completed
        video_aggregator: list of lists of images

    """
    # Reset environment
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    # Loop over the subtasks of a sequence of language goals
    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # Get lang annotation for subtask
        lang = instr_dict[subtask][0]

        # Run the policy
        success, video = rollout(
            env, model, task_checker, subtask, lang, verbose
        )
        if save_video:
            video_aggregators.append(video)

        # Only move to the next subgoal if this one was successful
        if success:
            success_counter += 1
            if first_only:
                return success_counter, video_aggregators
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask, lang, verbose=True):
    """
    Run the actual rollout on one subtask.

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = []  # show video for debugging
    obs = env.get_obs()

    model.reset()
    start_info = env.get_info()

    if verbose:
        print('------------------------------')
        print(f'task: {lang}')
    video.append(obs["rgb_obs"]["rgb_static"])

    # Loop at most EP_LEN times
    if verbose:
        pbar = tqdm(range(EP_LEN))
    else:
        pbar = range(EP_LEN)
    for step in pbar:
        # Prepare input
        obs = model.preprocess(obs, env)
        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory = model.step(obs, lang)
        # Execute
        for act_ind in range(trajectory.shape[1]):
            # How the env interprets the 7D command (see calvin_env robot.apply_action):
            # - cartesian_abs / cartesian_rel: dict with pos, euler, gripper chunks
            # - relative_array: bare length-7 ndarray -> relative_to_absolute (NOT absolute pose)
            exec_mode = getattr(model, "calvin_action_exec", "cartesian_abs")
            if exec_mode == "joint_abs":
                # Joint-space command: 7 absolute arm-joint targets (radians) +
                # discrete gripper. calvin_env robot.apply_action sends these
                # straight to POSITION_CONTROL (no IK). trajectory is (1, T, 8).
                step_action = np.asarray(trajectory[0, act_ind], dtype=np.float64).reshape(8)
                step_action[7] = 1 if step_action[7] > 0 else -1
                if verbose:
                    pbar.set_description(f"step: {step}")
                obs, _, _, current_info = env.step(
                    {"action": step_action, "type": "joint_abs"}
                )
            elif exec_mode == "relative_array":
                step_action = np.asarray(trajectory[0, act_ind], dtype=np.float64).reshape(7)
                step_action[6] = 1 if step_action[6] > 0 else -1
                if verbose:
                    pbar.set_description(f"step: {step}")
                obs, _, _, current_info = env.step(step_action)
            else:
                if exec_mode not in ("cartesian_abs", "cartesian_rel"):
                    raise ValueError(
                        f"Unknown calvin_action_exec={exec_mode!r}; use "
                        "'cartesian_abs', 'cartesian_rel', 'relative_array', or 'joint_abs'"
                    )
                curr_action = {
                    "action": [
                        trajectory[0, act_ind, :3],  # [x, y, z]
                        trajectory[0, act_ind, 3:6],  # [euler_x, euler_y, euler_z]
                        trajectory[0, act_ind, [6]],  # [open in {-1, 1}]
                    ],
                    "type": exec_mode,
                }
                if verbose:
                    pbar.set_description(f"step: {step}")
                obs, _, _, current_info = env.step(curr_action)

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )

            video.append(obs["rgb_obs"]["rgb_static"])

            if len(current_task_info) > 0:
                return True, video

    return False, video


@torch.no_grad()
def main(args):

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load a custom model wrapper
    model = create_model(args)

    # Optional per-action-chunk inference timing during live rollouts (BENCH_TIMING=1).
    from online_evaluation_calvin._bench_timing import maybe_install
    bench_timer = maybe_install(model)

    # Split sequence indices for multi-processing
    sequence_indices = list(
        range(args.local_rank, NUM_SEQUENCES, int(os.environ["WORLD_SIZE"]))
    )
    # Cap the number of sequences when benchmarking a handful of rollouts.
    bench_max_seq = os.environ.get("BENCH_MAX_SEQ")
    if bench_max_seq is not None:
        sequence_indices = sequence_indices[: int(bench_max_seq)]

    # Make environment
    env = get_env(args.merged_config_file, show_gui=False)

    # Run evaluation on all episodes
    evaluate_policy(model, env,
                    task_config_file=args.task_config_file,
                    ann_config_file=args.ann_config_file,
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video)

    if bench_timer is not None:
        bench_timer.restore()
        bench_timer.report(
            tag=f"{args.model_type}",
            pred_len=args.pred_len,
            json_out=os.environ.get("BENCH_JSON_OUT"),
        )

    # Gather statistics from the whole dataset
    results, _ = collect_results(args.base_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f'Load {len(results)}/1000 episodes...')
    print(str_results + "\n")

    env.close()
    del env
    gc.collect()


if __name__ == "__main__":
    args = parse_arguments()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.device = torch.device('cuda')
    EP_LEN = 360 // args.pred_len

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
