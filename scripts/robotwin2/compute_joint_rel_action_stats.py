#!/usr/bin/env python3
"""Compute RoboTwin2 ``rel_joint_action`` statistics for the joint-action models.

The RoboTwin2 joint zarr (``robotwin2_zarr_joint``) stores ``rel_joint_action``
with shape ``(N, H, nhand=2, 7)``: per arm, dims ``0..5`` are the anchor-relative
arm-joint deltas (6 joints) and dim ``6`` is the (binary) gripper command. The
joint flow head normalizes the 6 arm joints to ~[-1, 1] via a symmetric per-joint
``[min, max]`` range; this script derives that range from the data.

Unlike CALVIN (single Panda arm, 7 joints), RoboTwin2 is bimanual with 6 arm
joints per arm. The two arms share one normalizer, so stats are pooled across
``nhand`` (left+right) per joint index.

Outputs per-joint min / max / mean / abs-max / high percentiles, prints a
ready-to-paste symmetric ``hi`` vector for ``RoboTwin2JointTrainer.get_joint_normalizer``
(the normalizer is ``[-hi, hi]``), and writes a JSON next to the repo datasets.
The gripper dim is reported but excluded from the normalizer (binary, BCE head).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


DEFAULT_ROOT = Path(
    "/data/user_data/ziyinx/datasets/zarr_datasets/robotwin2_zarr_joint"
)
DEFAULT_TRAIN_ZARR = DEFAULT_ROOT / "train_rechunked8.zarr"
DEFAULT_VAL_ZARR = DEFAULT_ROOT / "val_rechunked8.zarr"

NUM_ARM_JOINTS = 6
DIM_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
)

# Histogram support for streaming percentile estimation of |value| per dim.
HIST_MAX_ABS = 4.0
HIST_BINS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zarr", type=Path, default=DEFAULT_TRAIN_ZARR)
    parser.add_argument("--val-zarr", type=Path, default=DEFAULT_VAL_ZARR)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[99.0, 99.9, 99.99, 100.0],
        help="Percentiles of |value| to report per joint (for robust ranges).",
    )
    parser.add_argument(
        "--normalizer-percentile",
        type=float,
        default=99.9,
        help="Percentile of |value| used for the suggested symmetric range.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.15,
        help="Multiplicative margin applied to the suggested range so most "
        "targets stay inside [-1, 1] (the model also clamps the rare tail).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "datasets"
        / "robotwin2_joint_rel_action_stats.json",
    )
    return parser.parse_args()


def percentile_from_hist(
    counts: np.ndarray, edges: np.ndarray, total: int, pct: float, abs_max: float
) -> float:
    """Estimate a percentile of |value| from a cumulative histogram.

    Returns the right edge of the bin where the cumulative count crosses the
    target rank. The 100th percentile returns the exact abs-max.
    """
    if total == 0:
        return float("nan")
    if pct >= 100.0:
        return float(abs_max)
    target = pct / 100.0 * total
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(edges) - 2)
    return float(edges[idx + 1])


def compute_split(zarr_path: Path, chunk_size: int, percentiles, norm_pct) -> dict:
    root = zarr.open_group(str(zarr_path), mode="r")
    if "rel_joint_action" not in root:
        raise KeyError(f"{zarr_path} does not contain rel_joint_action")
    arr = root["rel_joint_action"]
    if arr.ndim != 4 or arr.shape[-1] != len(DIM_NAMES):
        raise ValueError(
            f"Expected rel_joint_action shape (N, H, nhand, {len(DIM_NAMES)}), "
            f"got {arr.shape}"
        )
    ndim = arr.shape[-1]

    min_v = np.full((ndim,), np.inf, dtype=np.float64)
    max_v = np.full((ndim,), -np.inf, dtype=np.float64)
    sum_v = np.zeros((ndim,), dtype=np.float64)
    sum_abs = np.zeros((ndim,), dtype=np.float64)
    sum_abs_sq = np.zeros((ndim,), dtype=np.float64)
    abs_max = np.zeros((ndim,), dtype=np.float64)
    count = 0

    edges = np.linspace(0.0, HIST_MAX_ABS, HIST_BINS + 1)
    hist = np.zeros((ndim, HIST_BINS), dtype=np.int64)

    for start in range(0, arr.shape[0], chunk_size):
        stop = min(start + chunk_size, arr.shape[0])
        block = np.asarray(arr[start:stop])  # (n, H, nhand, ndim)
        # Pool both arms (nhand) per joint index: one normalizer for both arms.
        flat = block.reshape(-1, ndim).astype(np.float64)
        if flat.shape[0] == 0:
            continue
        min_v = np.minimum(min_v, flat.min(axis=0))
        max_v = np.maximum(max_v, flat.max(axis=0))
        sum_v += flat.sum(axis=0)
        abs_flat = np.abs(flat)
        sum_abs += abs_flat.sum(axis=0)
        sum_abs_sq += (abs_flat ** 2).sum(axis=0)
        abs_max = np.maximum(abs_max, abs_flat.max(axis=0))
        count += flat.shape[0]
        for d in range(ndim):
            hist[d] += np.histogram(abs_flat[:, d], bins=edges)[0]

    if count == 0:
        raise ValueError(f"No rows read from {zarr_path}")

    mean = sum_v / count
    mean_abs = sum_abs / count
    var_abs = sum_abs_sq / count - mean_abs ** 2
    std_abs = np.sqrt(np.maximum(var_abs, 0.0))

    pct_table = {
        f"p{p:g}": [
            percentile_from_hist(hist[d], edges, count, p, abs_max[d])
            for d in range(ndim)
        ]
        for p in percentiles
    }
    norm_abs = np.array(
        [
            percentile_from_hist(hist[d], edges, count, norm_pct, abs_max[d])
            for d in range(ndim)
        ]
    )

    return {
        "zarr_path": str(zarr_path),
        "shape": tuple(int(x) for x in arr.shape),
        "key": "rel_joint_action",
        "action_interpretation": (
            "anchor-relative future arm-joint deltas (6) + binary gripper command, "
            "pooled across nhand (left+right)"
        ),
        "num_arm_joints": NUM_ARM_JOINTS,
        "count": int(count),
        "dims": {
            name: {
                "min": float(min_v[i]),
                "max": float(max_v[i]),
                "abs_max": float(abs_max[i]),
                "mean": float(mean[i]),
                "mean_abs": float(mean_abs[i]),
                "std_abs": float(std_abs[i]),
                **{k: float(v[i]) for k, v in pct_table.items()},
            }
            for i, name in enumerate(DIM_NAMES)
        },
        "_norm_abs_arm": norm_abs[:NUM_ARM_JOINTS].tolist(),
    }


def suggest_normalizer(stats: dict, margin: float) -> list:
    arm = np.array(stats["_norm_abs_arm"])
    hi = np.round(arm * margin, 3)
    return hi.tolist()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    splits = []
    if args.split in ("all", "train"):
        splits.append(("train", args.train_zarr))
    if args.split in ("all", "val"):
        splits.append(("val", args.val_zarr))

    output = {}
    for name, path in splits:
        stats = compute_split(
            path, args.chunk_size, args.percentiles, args.normalizer_percentile
        )
        output[name] = stats
        print(f"\n{name}: {stats['zarr_path']} shape={stats['shape']} count={stats['count']}")
        for dim_name, ds in stats["dims"].items():
            print(
                f"  {dim_name}: min={ds['min']:+.5f} max={ds['max']:+.5f} "
                f"abs_max={ds['abs_max']:.5f} mean={ds['mean']:+.5f} "
                f"mean_abs={ds['mean_abs']:.5f} std_abs={ds['std_abs']:.5f} "
                f"p99.9={ds.get('p99.9', float('nan')):.5f}"
            )

    base_name = "train" if "train" in output else next(iter(output))
    base = output[base_name]
    hi = suggest_normalizer(base, args.margin)
    print(
        f"\nSuggested symmetric arm-joint range (p{args.normalizer_percentile:g} of |value| "
        f"x{args.margin:g} margin), based on '{base_name}':"
    )
    print(f"  hi = torch.tensor({[round(v, 3) for v in hi]})")
    print("  # joint_normalizer = torch.stack([-hi, hi])  # shape (2, 6)")
    output["suggested_normalizer"] = {
        "percentile": args.normalizer_percentile,
        "margin": args.margin,
        "hi": [round(v, 3) for v in hi],
        "note": "symmetric range; joint_normalizer = stack([-hi, hi]); gripper excluded",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"\nWrote JSON stats to {args.output_json}")


if __name__ == "__main__":
    main()
