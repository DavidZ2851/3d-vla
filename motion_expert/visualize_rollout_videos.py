from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from motion_expert.pipeline import MotionExpertPipeline
    from motion_expert.zarr_dataset import MotionExpertZarrDataset


DEFAULT_CHECKPOINT = (
    "/data/user_data/ziyinx/analogical_manipulation_clean/"
    "train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt"
)
DEFAULT_CONFIG = (
    "/data/user_data/ziyinx/analogical_manipulation_clean/"
    "motion_expert/configs/calvin_florence2_viz.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MotionExpert inference from a checkpoint and save one spaghetti "
            "rollout video per sample using query points only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint_step_*.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="YAML/JSON config used to locate the dataset and tokenizer settings.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val"),
        default="val",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--train-zarr",
        type=str,
        default=None,
        help="Optional override for data.train_zarr from the config.",
    )
    parser.add_argument(
        "--val-zarr",
        type=str,
        default="/data/group_data/katefgroup/VLA/zarr_datasets/calvin_motion_zarr_rechunk/calvin_motion_zarr_rechunk/val_rechunked8.zarr",
        help="Optional override for data.val_zarr from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/user_data/ziyinx/vis_outputs/",
        help="Directory for saved rollout videos. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--rgb-output-dir",
        type=str,
        default="/data/user_data/ziyinx/vis_outputs_rgb/",
        help="Directory for saved per-rollout RGB input images. Defaults to <output-dir>_rgb.",
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=128,
        help="Optional cap on the number of rollout videos to export.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional raw sample batch size. Defaults to config train.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Inference steps. Defaults to train.val_num_inference_steps from config.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for saved MP4s.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.005,
        help="Meters required for a trajectory to count as moving.",
    )
    parser.add_argument(
        "--max-points-per-sample",
        type=int,
        default=400,
        help="Maximum moved query points to render per rollout.",
    )
    return parser.parse_args()


def _load_raw_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        if str(path).endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle)


def _chunk_loader_batch_size(raw_batch_size: int, chunk_size: int) -> int:
    return max(1, int(raw_batch_size) // max(1, int(chunk_size)))


def _device_from_arg(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataset(
    *,
    raw_cfg: Dict[str, Any],
    model_cfg,
    split: str,
    max_rollouts: Optional[int],
) -> "MotionExpertZarrDataset":
    from motion_expert.zarr_dataset import MotionExpertZarrDataset

    data_cfg = raw_cfg["data"]
    instructions_path = data_cfg.get(
        f"{split}_instructions_path",
        data_cfg.get("instructions_path"),
    )
    zarr_override = {
        "train": raw_cfg.get("_cli_train_zarr"),
        "val": raw_cfg.get("_cli_val_zarr"),
    }
    zarr_path = zarr_override.get(split) or data_cfg[f"{split}_zarr"]
    if not Path(zarr_path).exists():
        raise FileNotFoundError(
            f"Configured {split} zarr path does not exist: {zarr_path}\n"
            f"Config file: {raw_cfg.get('_config_path', '<unknown>')}\n"
            "Pass --train-zarr/--val-zarr to override, or update the config."
        )
    return MotionExpertZarrDataset(
        root=zarr_path,
        instructions_path=instructions_path,
        image_size=(model_cfg.input_height, model_cfg.input_width),
        instruction_id_key=data_cfg.get("instruction_id_key"),
        query_uv_format=model_cfg.query_uv_format,
        image_pad_value=model_cfg.image_pad_value,
        mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
        traj_3d_xyz_is_delta=data_cfg.get("traj_3d_xyz_is_delta"),
        max_samples=max_rollouts,
        chunk_size=int(data_cfg.get("chunk_size", 1)),
    )


def _maybe_apply_calvin_normalization(
    pipeline: "MotionExpertPipeline",
    raw_cfg: Dict[str, Any],
) -> None:
    from motion_expert.compute_calvin_normalization_stats import (
        extract_calvin_normalization_tensors,
        load_calvin_normalization_stats,
    )

    train_cfg = raw_cfg.get("train", {})
    if not bool(train_cfg.get("use_calvin_normalization", False)):
        return
    stats_path = train_cfg.get("normalization_stats_path")
    if stats_path is None:
        return
    stats = load_calvin_normalization_stats(Path(stats_path))
    query_center, query_scale, traj_center, traj_scale = extract_calvin_normalization_tensors(stats)
    pipeline.set_calvin_normalization_statistics(
        torch.from_numpy(query_center),
        torch.from_numpy(query_scale),
        torch.from_numpy(traj_center),
        torch.from_numpy(traj_scale),
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    from motion_expert.config import build_motion_expert_config
    from motion_expert.pipeline import MotionExpertPipeline
    from motion_expert.visualization import save_motion_expert_spaghetti_rollout_video
    from motion_expert.zarr_dataset import (
        MotionExpertCollator,
        MotionExpertQwenTokenizer,
    )
    raw_cfg = _load_raw_config(args.config)
    raw_cfg["_config_path"] = args.config
    raw_cfg["_cli_train_zarr"] = args.train_zarr
    raw_cfg["_cli_val_zarr"] = args.val_zarr

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = build_motion_expert_config(checkpoint.get("config", raw_cfg["model"]))

    if model_cfg.qwen.model_name_or_path is not None:
        os.environ["QWEN3VL_MODEL_PATH"] = model_cfg.qwen.model_name_or_path
    if model_cfg.florence.model_name_or_path is not None:
        os.environ["FLORENCE2_MODEL_PATH"] = model_cfg.florence.model_name_or_path

    dataset = _build_dataset(
        raw_cfg=raw_cfg,
        model_cfg=model_cfg,
        split=args.split,
        max_rollouts=args.max_rollouts,
    )

    backbone_type = str(model_cfg.backbone_type).lower()
    tokenizer_model_path = (
        model_cfg.qwen.model_name_or_path
        if backbone_type == "qwen"
        else model_cfg.florence.model_name_or_path
    )
    tokenizer = MotionExpertQwenTokenizer(
        backbone_type=backbone_type,
        model_name_or_path=tokenizer_model_path,
        max_length=model_cfg.text_max_length,
    )
    collator = MotionExpertCollator(tokenizer)

    chunk_size = int(raw_cfg["data"].get("chunk_size", 1))
    raw_batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(raw_cfg.get("train", {}).get("batch_size", 8))
    )
    loader = DataLoader(
        dataset,
        batch_size=_chunk_loader_batch_size(raw_batch_size, chunk_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=collator,
        persistent_workers=bool(args.num_workers > 0),
    )

    device = _device_from_arg(args.device)
    pipeline = MotionExpertPipeline.from_backbone(model_cfg)
    _maybe_apply_calvin_normalization(pipeline, raw_cfg)
    pipeline.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pipeline = pipeline.to(device)
    pipeline.eval()

    inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(raw_cfg.get("train", {}).get("val_num_inference_steps", 10))
    )

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (
        checkpoint_path.parent / f"{args.split}_query_spaghetti_videos_step_{checkpoint.get('step', 'unk')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_output_dir = (
        Path(args.rgb_output_dir)
        if args.rgb_output_dir
        else output_dir.parent / f"{output_dir.name}_rgb"
    )
    rgb_output_dir.mkdir(parents=True, exist_ok=True)

    rollout_index = 0
    total_available = int(getattr(dataset, "_effective_raw_length", len(dataset)))
    total_target = (
        total_available
        if args.max_rollouts is None
        else min(total_available, int(args.max_rollouts))
    )

    print(
        f"[motion_expert][viz] split={args.split} total_rollouts={total_target} "
        f"device={device} output_dir={output_dir} rgb_output_dir={rgb_output_dir}"
    )

    with torch.no_grad():
        for batch in loader:
            batch_device = batch.to(device)
            outputs = pipeline.predict(
                batch_device,
                num_inference_steps=inference_steps,
            )
            pred_delta_xyz = outputs["pred_delta_xyz"].detach().cpu()

            batch_rgb = batch.rgb.detach().cpu()
            batch_query_world_xyz = batch.query_world_xyz.detach().cpu()
            batch_future_delta_xyz = batch.future_delta_xyz.detach().cpu()
            batch_traj_mask = batch.traj_mask.detach().cpu()

            batch_size = int(batch_query_world_xyz.shape[0])
            for local_idx in range(batch_size):
                if rollout_index >= total_target:
                    break
                stem = f"rollout_{rollout_index:05d}"
                output_path = output_dir / f"{stem}.gif"
                rgb_path = rgb_output_dir / f"{stem}.png"
                save_motion_expert_spaghetti_rollout_video(
                    query_world_xyz=batch_query_world_xyz[local_idx],
                    gt_delta_xyz=batch_future_delta_xyz[local_idx],
                    pred_delta_xyz=pred_delta_xyz[local_idx],
                    traj_mask=batch_traj_mask[local_idx],
                    output_path=output_path,
                    fps=int(args.fps),
                    motion_threshold=float(args.motion_threshold),
                    max_points_per_sample=int(args.max_points_per_sample),
                    instruction_text=batch.instruction_text[local_idx],
                    sample_index=rollout_index,
                    show_anchor_points=False,
                )
                rgb_img = batch_rgb[local_idx].float()
                if rgb_img.dim() == 3 and rgb_img.shape[0] in (1, 3):
                    rgb_img = rgb_img.permute(1, 2, 0).contiguous()
                rgb_img = rgb_img.clamp(0.0, 1.0).numpy()
                plt.imsave(rgb_path, rgb_img)
                print(
                    f"[motion_expert][viz] saved rollout={rollout_index} "
                    f"path={output_path} rgb_path={rgb_path}"
                )
                rollout_index += 1
            if rollout_index >= total_target:
                break

    print(
        f"[motion_expert][viz] complete saved_rollouts={rollout_index} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
