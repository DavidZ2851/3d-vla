#!/usr/bin/env python3
"""Compute per-dimension statistics for LIBERO rel_action with quat->euler conversion.

Conversion for quat_xyzw exactly matches LIBERO training dataset code:
quat_xyzw -> quat_wxyz -> quaternion_to_matrix -> matrix_to_euler_xyz.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show per-dimension statistics for rel_action in a LIBERO zarr dataset, "
            "after converting quat_xyzw to euler_xyz using LIBERO's training conversion."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/val_rechunked8.zarr"),
        help="Path to the LIBERO .zarr store.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="rel_action",
        help="Key in the zarr store (default: rel_action).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of episodes to process per batch.",
    )
    return parser.parse_args()


def normalize_shape(batch: np.ndarray) -> np.ndarray:
    # [B, T, 1, D] -> [B, T, D], keep [B, T, D] as-is.
    if batch.ndim == 4:
        if batch.shape[2] != 1:
            raise ValueError(f"Expected axis-2 size 1 for 4D input, got shape {batch.shape}")
        batch = batch[:, :, 0, :]
    elif batch.ndim != 3:
        raise ValueError(f"Expected 3D or 4D array, got shape {batch.shape}")
    return batch


def get_libero_quat_to_euler_fn():
    repo_root = "/data/user_data/ziyinx/analogical_manipulation_clean"
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Use the exact conversion function used by LIBERO dataset in training.
    from datasets.libero import LIBERODataset  # pylint: disable=import-error

    return LIBERODataset._quat_xyzw_to_euler_xyz


def compute_stats(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p0.5": float(np.percentile(values, 0.5)),
        "p99.5": float(np.percentile(values, 99.5)),
        "p0.1": float(np.percentile(values, 0.1)),
        "p99.9": float(np.percentile(values, 99.9)),
        "p0.05": float(np.percentile(values, 0.05)),
        "p99.95": float(np.percentile(values, 99.95)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_std": float(np.std(np.abs(values))),
    }


def main() -> None:
    args = parse_args()
    root = zarr.open(str(args.zarr_path), mode="r")

    if args.key not in root:
        raise KeyError(f"Key '{args.key}' not found in store: {args.zarr_path}")

    arr = root[args.key]
    n = arr.shape[0]

    quat_to_euler = get_libero_quat_to_euler_fn()

    converted_chunks = []
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = np.asarray(arr[start:end], dtype=np.float32)
        batch = normalize_shape(batch)  # [B, T, 8]

        batch_torch = torch.from_numpy(batch)
        converted = quat_to_euler(batch_torch).cpu().numpy()  # [B, T, 7]
        converted_chunks.append(converted.reshape(-1, converted.shape[-1]))

    all_values = np.concatenate(converted_chunks, axis=0)  # [N*T, 7]

    dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    if all_values.shape[1] != len(dim_names):
        dim_names = [f"dim_{i}" for i in range(all_values.shape[1])]

    print(f"zarr_path: {args.zarr_path}")
    print(f"key: {args.key}")
    print(f"raw_shape: {arr.shape}")
    print(f"converted_shape_flat: {all_values.shape}  # [N*T, D]")
    print("conversion: LIBERODataset._quat_xyzw_to_euler_xyz")
    print()

    header = (
        "dim,min,max,p0.5,p99.5,p0.1,p99.9,p0.05,p99.95,"
        "mean,std,abs_mean,abs_std"
    )
    print(header)

    for i, name in enumerate(dim_names):
        s = compute_stats(all_values[:, i])
        print(
            f"{name},"
            f"{s['min']:.10f},{s['max']:.10f},"
            f"{s['p0.5']:.10f},{s['p99.5']:.10f},"
            f"{s['p0.1']:.10f},{s['p99.9']:.10f},"
            f"{s['p0.05']:.10f},{s['p99.95']:.10f},"
            f"{s['mean']:.10f},{s['std']:.10f},"
            f"{s['abs_mean']:.10f},{s['abs_std']:.10f}"
        )


if __name__ == "__main__":
    main()
