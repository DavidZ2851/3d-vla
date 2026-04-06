import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from utils.common_utils import str2bool
from online_evaluation_calvin import evaluate_policy as calvin_eval
from online_evaluation_calvin.utils_with_calvin import collect_results, count_success, get_env, get_log_dir
from online_evaluation_calvin.xiaomi_model_wrapper import XiaomiModel


def parse_arguments():
    parser = argparse.ArgumentParser("Evaluate Xiaomi MiBoT on CALVIN (shared evaluate_policy path)")
    arguments = [
        ("merged_config_file", Path, "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml"),
        ("task_config_file", Path, "online_evaluation_calvin/configs/new_playtable_tasks.yaml"),
        ("ann_config_file", Path, "online_evaluation_calvin/configs/new_playtable_validation.yaml"),
        (
            "eval_sequences_file",
            str,
            "",
        ),  # empty: multistep_sequences.get_sequences (CALVIN standard eval)
        ("model_path", str, "/path/to/xiaomi_calvin"),
        ("split", str, "abc"),
        ("crop_ratio", float, 0.95),
        ("pred_len", int, 10),
        ("num_sequences", int, calvin_eval.NUM_SEQUENCES),
        ("seed", int, 0),
        ("base_log_dir", Path, Path(__file__).parent / "eval_logs" / "xiaomi_calvin"),
        ("save_video", str2bool, False),
        (
            "action_exec",
            str,
            "relative_array",
        ),  # MiBoT-style raw 7D -> env.relative_to_absolute; alternatives: cartesian_abs | cartesian_rel
    ]
    for name, value_type, default in arguments:
        parser.add_argument(f"--{name}", type=value_type, default=default)
    return parser.parse_args()


def load_eval_sequences_json(path: Path, num_sequences: int):
    """Optional external list of (initial_state, subtask_list); shuffled like Xiaomi tool for comparability."""
    original_random_state = random.getstate()
    random.seed(42)
    with open(path) as f:
        sequences = json.load(f)
    random.shuffle(sequences)
    sequences = sequences[:num_sequences]
    random.setstate(original_random_state)
    return sequences


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = XiaomiModel(
        model_path=args.model_path,
        split=args.split,
        crop_ratio=args.crop_ratio,
        action_chunk=args.pred_len,
        device=args.device,
    )
    model.calvin_action_exec = args.action_exec

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    n_seq = args.num_sequences
    eval_sequences = None
    if str(args.eval_sequences_file).strip():
        p = Path(args.eval_sequences_file)
        if not p.is_file():
            raise FileNotFoundError(f"eval_sequences_file not found: {p}")
        eval_sequences = load_eval_sequences_json(p, args.num_sequences)
        n_seq = len(eval_sequences)

    sequence_indices = list(range(args.local_rank, n_seq, world_size))

    env = get_env(args.merged_config_file, show_gui=False)
    eval_log_dir = get_log_dir(args.base_log_dir)

    calvin_eval.evaluate_policy(
        model,
        env,
        task_config_file=args.task_config_file,
        ann_config_file=args.ann_config_file,
        eval_log_dir=eval_log_dir,
        sequence_indices=sequence_indices,
        save_video=args.save_video,
        eval_sequences=eval_sequences,
    )

    results, _ = collect_results(eval_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f"Load {len(results)}/{n_seq} episodes...")
    print(str_results + "\n")

    env.close()
    del env
    gc.collect()


if __name__ == "__main__":
    args = parse_arguments()
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calvin_eval.EP_LEN = 360 // args.pred_len

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.device.type == "cuda":
        torch.cuda.set_device(args.local_rank)
        if world_size > 1:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    main(args)
