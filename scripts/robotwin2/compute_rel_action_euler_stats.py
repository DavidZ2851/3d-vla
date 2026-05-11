#!/usr/bin/env python3
"""Compute RobotWin2 rel_action quat_wxyz -> XYZ Euler statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import zarr

from datasets.utils import quat_wxyz_to_euler_xyz


DEFAULT_TRAIN_ZARR = Path(
    "/data/user_data/ziyinx/datasets/zarr_datasets/robotwin2_zarr/train.zarr"
)
DEFAULT_VAL_ZARR = Path(
    "/data/group_data/katefgroup/datasets/ziyinx/"
    "robotwin2_zarr_rechunked_stride5/val_rechunked8.zarr"
)
HAND_NAMES = ("left", "right")
EULER_NAMES = ("x", "y", "z")


def update_stats(stats: dict[str, torch.Tensor], euler: torch.Tensor) -> None:
    """Update per-hand Euler min/max/sum/sumsq/count.

    Args:
        stats: running statistics.
        euler: tensor with shape (N, T, 2, 3), radians.
    """
    flat = euler.reshape(-1, euler.shape[-2], 3)
    stats["min"] = torch.minimum(stats["min"], flat.amin(dim=0))
    stats["max"] = torch.maximum(stats["max"], flat.amax(dim=0))
    stats["sum"] += flat.sum(dim=0, dtype=torch.float64)
    stats["sumsq"] += (flat.to(torch.float64) ** 2).sum(dim=0)
    stats["count"] += flat.shape[0]


def compute_split_stats(zarr_path: Path, chunk_size: int, max_samples: int | None):
    root = zarr.open_group(str(zarr_path), mode="r")
    if "rel_action" not in root:
        raise KeyError(f"{zarr_path} does not contain rel_action")

    rel_action = root["rel_action"]
    if len(rel_action.shape) != 4 or rel_action.shape[-2:] != (2, 8):
        raise ValueError(
            f"Expected rel_action shape (N, T, 2, 8), got {rel_action.shape}"
        )

    total = rel_action.shape[0]
    if max_samples is not None:
        total = min(total, max_samples)

    stats = {
        "min": torch.full((2, 3), float("inf"), dtype=torch.float64),
        "max": torch.full((2, 3), float("-inf"), dtype=torch.float64),
        "sum": torch.zeros((2, 3), dtype=torch.float64),
        "sumsq": torch.zeros((2, 3), dtype=torch.float64),
        "count": 0,
    }

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        action = torch.as_tensor(rel_action[start:end])
        quat_wxyz = action[..., 3:7].to(torch.float32)
        euler = quat_wxyz_to_euler_xyz(quat_wxyz).to(torch.float64)
        update_stats(stats, euler)

    if stats["count"] == 0:
        raise ValueError(f"No rel_action rows read from {zarr_path}")

    mean = stats["sum"] / stats["count"]
    var = stats["sumsq"] / stats["count"] - mean.square()
    stats["mean"] = mean
    stats["std"] = var.clamp_min(0).sqrt()
    stats["rows"] = total
    stats["tokens_per_hand"] = stats["count"]
    return stats


def print_split(name: str, zarr_path: Path, stats: dict[str, torch.Tensor]) -> None:
    print(f"\n{name}: {zarr_path}")
    print(f"rows={stats['rows']} tokens_per_hand={stats['tokens_per_hand']}")
    for hand_idx, hand_name in enumerate(HAND_NAMES):
        print(f"  {hand_name}_hand euler_xyz radians:")
        for axis_idx, axis_name in enumerate(EULER_NAMES):
            print(
                f"    {axis_name}: "
                f"min={stats['min'][hand_idx, axis_idx].item(): .9f} "
                f"max={stats['max'][hand_idx, axis_idx].item(): .9f} "
                f"mean={stats['mean'][hand_idx, axis_idx].item(): .9f} "
                f"std={stats['std'][hand_idx, axis_idx].item(): .9f}"
            )


def stats_to_jsonable(zarr_path: Path, stats: dict[str, torch.Tensor]) -> dict:
    hands = {}
    for hand_idx, hand_name in enumerate(HAND_NAMES):
        axes = {}
        for axis_idx, axis_name in enumerate(EULER_NAMES):
            axes[axis_name] = {
                "min": stats["min"][hand_idx, axis_idx].item(),
                "max": stats["max"][hand_idx, axis_idx].item(),
                "mean": stats["mean"][hand_idx, axis_idx].item(),
                "std": stats["std"][hand_idx, axis_idx].item(),
            }
        hands[hand_name] = axes

    return {
        "zarr_path": str(zarr_path),
        "rows": int(stats["rows"]),
        "tokens_per_hand": int(stats["tokens_per_hand"]),
        "rotation_source": "rel_action[..., 3:7]",
        "quaternion_format": "wxyz",
        "euler_convention": "xyz",
        "units": "radians",
        "hands": hands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-hand statistics for RobotWin2 rel_action quaternion "
            "labels converted from quat_wxyz to XYZ Euler radians."
        )
    )
    parser.add_argument("--train-zarr", type=Path, default=DEFAULT_TRAIN_ZARR)
    parser.add_argument("--val-zarr", type=Path, default=DEFAULT_VAL_ZARR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Number of zarr rows to read per chunk.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for quick smoke tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("robotwin2_rel_action_euler_stats.json"),
        help="Path to write train/val statistics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output = {}
    for split_name, zarr_path in (
        ("train", args.train_zarr),
        ("val", args.val_zarr),
    ):
        stats = compute_split_stats(zarr_path, args.chunk_size, args.max_samples)
        print_split(split_name, zarr_path, stats)
        output[split_name] = stats_to_jsonable(zarr_path, stats)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nWrote JSON stats to {args.output_json}")


if __name__ == "__main__":
    main()
