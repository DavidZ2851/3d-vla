"""Main script for training and testing."""

import argparse
import os
from pathlib import Path
import random
import sys
import datetime

import numpy as np

os.environ["MOTION_EXPERT_USE_FLASH_ATTN_VARLEN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from utils.common_utils import str2bool, str_none
from utils.trainers import fetch_train_tester


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, ''),
        ('eval_data_dir', Path, ''),
        ('train_instructions', Path, ''),
        ('val_instructions', Path, ''),
        ('dataset', str, "Peract"),
        # Master RNG seed. Seeds model init, flow-matching noise and dropout
        # (main.py, offset per rank) and the data order (DistributedSampler +
        # DataLoader generator in utils/trainers/base.py). 0 = the historical
        # data order, so existing runs are unchanged.
        ('seed', int, 0),
        ('num_workers', int, 4),
        ('batch_size', int, 64),
        ('batch_size_val', int, 64),
        ('chunk_size', int, 1),
        ('memory_limit', float, 8),  # cache limit in GB
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, "run"),
        # Training and testing arguments
        ('checkpoint', str_none, None),
        ('val_freq', int, 4000),
        ('interm_ckpt_freq', int, 1000000),
        ('eval_only', str2bool, False),
        ('lr', float, 1e-4),
        ('backbone_lr', float, 1e-4),
        ('num_finetune_vision_layers', int, 0),
        ('vision_lr_scale', float, 0.05),
        ('projector_lr_scale', float, 0.1),
        ('lr_scheduler', str, "constant"),
        ('wd', float, 5e-3),
        ('train_iters', int, 600000),
        ('use_compile', str2bool, False),
        ('find_unused_parameters', str2bool, True),
        ('use_ema', str2bool, False),
        ('lv2_batch_size', int, 1),
        # Model arguments: general policy type
        ('model_type', str, 'denoise3d'),
        ('bimanual', str2bool, False),
        ('keypose_only', str2bool, True),
        ('pre_tokenize', str2bool, True),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, 0.04),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('finetune_backbone', str2bool, False),
        ('finetune_text_encoder', str2bool, False),
        ('fps_subsampling_factor', int, 5),
        ('use_multiscale', str2bool, False),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),  # divisible by num_attn_heads
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 3),
        ('num_history', int, 1),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 10),
        ('denoise_model', str, "rectified_flow"),
        ('finetune', str2bool, False),  # NOTE: for finetune
        # LoRA arguments
        ('vision_lora_rank', int, 0),
        ('vision_lora_alpha', int, 16),
        ('vision_lora_dropout', float, 0.0),
        ('vision_lora_num_layers', int, 0),
        ('vision_lora_target_modules', str, 'attn.qkv,attn.proj'),
        ('llm_lora_rank', int, 0),
        ('llm_lora_alpha', int, 16),
        ('llm_lora_dropout', float, 0.0),
        ('llm_lora_num_layers', int, 0),
        ('llm_lora_target_modules', str, 'self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj'),
        ('vision_lora_lr_scale', float, 0.05),
        ('merger_lr_scale', float, 0.1),
        ('gradient_checkpointing', str2bool, False),
        ('grounding_loss_weight', float, 10.0),
        ('grounding_loss_weight_hold_iters', int, 0),  # 0 = fixed weight; else hold --grounding_loss_weight for this many steps
        ('grounding_loss_weight_final', float, 0.1),
        # Training-only conditioning dropout for grounding-conditioned policies.
        # Samples dropped here are trained exactly like inference: no grounding token.
        ('grounding_dropout_prob', float, 0.0),
        # Optional RoboTwin2 rel_action stats JSON for action normalization.
        ('robotwin2_action_stats', str_none, None),
        ('robotwin2_relative_rotation', str2bool, False),
        # Motion Expert latent conditioning.
        ('motion_expert_config', str_none, None),
        ('motion_expert_checkpoint', str_none, None),
        ('motion_solver_steps', int, 10),
        ('motion_steps_to_run', int, 1),
        ('motion_finetune', str2bool, True),
        ('motion_scene_fusion', str2bool, True),
        ('motion_action_fusion', str2bool, True),
        # Geometry frame handed to the motion expert by a fixed-frame policy
        # (flower_actor_3d_motion_latent_cam / _yaw45). 'world' reuses a
        # world-trained expert checkpoint as-is by converting the query back to
        # world; 'camera' feeds the policy's own frame straight through and
        # therefore needs an expert trained in THAT frame (the value is a
        # frame-agnostic "use my frame", not literally the camera).
        ('motion_expert_frame', str, 'world'),
        # Joint-action decoder (flower_actor_3d_motion_latent_joint / flower_actor_2d_joint).
        ('joint_action_dim', int, 7),
        ('joint_proprio_dim', int, 8),
        ('gripper_dim', int, 1),
        ('joint_num_layers', int, 4),
        # 0 -> share the backbone embedding_dim / num_attn_heads.
        ('joint_embedding_dim', int, 0),
        ('joint_num_attn_heads', int, 0),
        ('joint_loss_weight', float, 1.0),
        ('joint_action_weight', float, 30.0),
        # Number of 3D scene cameras the 2D ablation (flower_actor_2d*) tags with
        # camera-id embeddings. CALVIN/LIBERO: 2; RoboTwin2 head/left/right: 3.
        ('num_camera_ids', int, 2),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main(log_dir=None):
    if int(os.environ.get("RANK", 0)) != 0:
        if os.environ.get("FLOWERVLA_KEEP_NON_MAIN_OUTPUT", "0") == "1":
            return
        if log_dir is not None and os.environ.get("FLOWERVLA_RANK_LOGS", "1") != "0":
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            rank_log_dir = Path(log_dir) / "rank_logs"
            rank_log_dir.mkdir(exist_ok=True, parents=True)
            sys.stdout = open(
                rank_log_dir / f"rank{rank}_local{local_rank}.out",
                "a",
                buffering=1,
            )
            sys.stderr = open(
                rank_log_dir / f"rank{rank}_local{local_rank}.err",
                "a",
                buffering=1,
            )
        else:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available():
        torch.set_flush_denormal(True)
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])
    suppress_output_on_non_main(log_dir)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(hours=2))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed the global RNGs: model init, flow-matching noise, dropout. The rank
    # offset keeps per-rank noise decorrelated (DDP broadcasts parameters from
    # rank 0 at construction, so init stays consistent across ranks regardless).
    # Data order is seeded separately from args.seed in utils/trainers/base.py.
    rank_seed = args.seed * 1000 + int(os.environ.get("RANK", 0))
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)

    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_class = fetch_model_class(args.model_type)

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_class)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
