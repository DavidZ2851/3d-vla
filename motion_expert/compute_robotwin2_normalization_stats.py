#!/usr/bin/env python3
"""Compute RobotWin2 Motion Expert percentile normalization statistics.

The output schema matches datasets/calvin_motion_normalization_stats.json:
- query_centered_xyz: centered finite query points.
- traj_delta_xyz: finite trajectory deltas selected by valid_mask.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import zarr


DEFAULT_ROBOTWIN2_SOURCE_ROOT = Path(
    "/data/user_data/ziyinx/datasets/zarr_datasets/robotwin2_motion_zarr/train_rechunked8.zarr"
)
DEFAULT_OUTPUT_JSON = Path(
    "/data/user_data/ziyinx/analogical_manipulation_clean/datasets/robotwin2_motion_normalization_stats.json"
)
DEFAULT_PERCENTILES = (0.5, 99.5)
DEFAULT_NUM_BINS = 131072
DEFAULT_CHUNK_SIZE = 2048


@dataclass
class AxisSummary:
    count: int
    min: List[float]
    max: List[float]
    p005: List[float]
    p995: List[float]
    center: List[float]
    scale: List[float]


def iter_robotwin2_zarr_stores(source_root: Path) -> Iterator[Path]:
    if source_root.is_dir() and source_root.suffix == ".zarr":
        yield source_root
        return

    stores = sorted(p for p in source_root.glob("*.zarr") if p.is_dir())
    if stores:
        for path in stores:
            yield path
        return

    raise RuntimeError(f"No RobotWin2 motion zarr stores found under {source_root}.")


def _finite_query_centered_xyz(query_world_xyz: np.ndarray) -> np.ndarray:
    query_world_xyz = np.asarray(query_world_xyz, dtype=np.float32)
    if query_world_xyz.ndim != 2 or query_world_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected query_world_xyz to have shape [N, 3], got {query_world_xyz.shape}."
        )
    finite = np.isfinite(query_world_xyz).all(axis=1)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float32)
    query_valid = query_world_xyz[finite]
    origin = query_valid.mean(axis=0, keepdims=True)
    return query_valid - origin


def _valid_traj_delta_xyz(traj_3d_xyz: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    traj_3d_xyz = np.asarray(traj_3d_xyz, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if traj_3d_xyz.ndim != 3 or traj_3d_xyz.shape[-1] != 3:
        raise ValueError(
            f"Expected traj_3d_xyz to have shape [N, T, 3], got {traj_3d_xyz.shape}."
        )
    if valid_mask.shape != traj_3d_xyz.shape[:2]:
        raise ValueError(
            f"Expected valid_mask shape {traj_3d_xyz.shape[:2]}, got {valid_mask.shape}."
        )
    values = traj_3d_xyz[valid_mask]
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float32)
    return values[finite]


def _update_min_max_count(
    values: np.ndarray,
    current_min: np.ndarray,
    current_max: np.ndarray,
) -> int:
    if values.size == 0:
        return 0
    current_min[:] = np.minimum(current_min, values.min(axis=0))
    current_max[:] = np.maximum(current_max, values.max(axis=0))
    return int(values.shape[0])


def _accumulate_histogram(
    values: np.ndarray,
    hist: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
) -> None:
    if values.size == 0:
        return
    num_bins = hist.shape[1]
    for axis in range(3):
        axis_min = float(min_vals[axis])
        axis_max = float(max_vals[axis])
        axis_values = values[:, axis]
        if axis_values.size == 0:
            continue
        if axis_max <= axis_min:
            hist[axis, 0] += axis_values.shape[0]
            continue
        bin_width = (axis_max - axis_min) / float(num_bins)
        indices = np.floor((axis_values - axis_min) / bin_width).astype(np.int64)
        indices = np.clip(indices, 0, num_bins - 1)
        hist[axis] += np.bincount(indices, minlength=num_bins)


def _percentile_from_histogram(
    hist: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    counts: np.ndarray,
    percentile: float,
) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    num_bins = hist.shape[1]
    for axis in range(3):
        axis_count = int(counts[axis])
        if axis_count <= 0:
            raise RuntimeError("Cannot compute percentiles without any values.")
        axis_min = float(min_vals[axis])
        axis_max = float(max_vals[axis])
        if axis_max <= axis_min:
            out[axis] = axis_min
            continue
        cdf = np.cumsum(hist[axis], dtype=np.int64)
        target_rank = percentile / 100.0 * float(axis_count - 1)
        target_index = int(np.floor(target_rank)) + 1
        bin_idx = int(np.searchsorted(cdf, target_index, side="left"))
        bin_idx = min(max(bin_idx, 0), num_bins - 1)
        prev_cum = int(cdf[bin_idx - 1]) if bin_idx > 0 else 0
        bin_count = int(hist[axis, bin_idx])
        bin_width = (axis_max - axis_min) / float(num_bins)
        left = axis_min + bin_idx * bin_width
        right = axis_max if bin_idx == num_bins - 1 else left + bin_width
        if bin_count <= 0:
            out[axis] = left
            continue
        frac = (target_rank - prev_cum) / float(bin_count)
        frac = min(max(frac, 0.0), 1.0)
        out[axis] = left + frac * (right - left)
    return out


def _finalize_axis_summary(
    hist: np.ndarray,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    total_count: int,
    percentiles: Tuple[float, float],
) -> AxisSummary:
    counts = np.full(3, total_count, dtype=np.int64)
    p005 = _percentile_from_histogram(hist, min_vals, max_vals, counts, percentiles[0])
    p995 = _percentile_from_histogram(hist, min_vals, max_vals, counts, percentiles[1])
    center = (p005 + p995) / 2.0
    scale = np.maximum((p995 - p005) / 2.0, 1e-6)
    return AxisSummary(
        count=int(total_count),
        min=min_vals.astype(np.float64).tolist(),
        max=max_vals.astype(np.float64).tolist(),
        p005=p005.astype(np.float64).tolist(),
        p995=p995.astype(np.float64).tolist(),
        center=center.astype(np.float64).tolist(),
        scale=scale.astype(np.float64).tolist(),
    )


def _iter_store_samples(
    query_arr: zarr.Array,
    traj_arr: zarr.Array,
    valid_arr: zarr.Array,
    chunk_size: int,
    remaining_limit: Optional[int],
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    num_samples = int(query_arr.shape[0])
    stop = num_samples if remaining_limit is None else min(num_samples, int(remaining_limit))
    for start in range(0, stop, chunk_size):
        end = min(start + chunk_size, stop)
        query = np.asarray(query_arr[start:end], dtype=np.float32)
        traj = np.asarray(traj_arr[start:end], dtype=np.float32)
        valid = np.asarray(valid_arr[start:end], dtype=bool)
        yield query, traj, valid


def _check_store_shapes(
    store_path: Path,
    root: zarr.Group,
) -> Tuple[zarr.Array, zarr.Array, zarr.Array]:
    for key in ("query_world_xyz", "traj_3d_xyz", "valid_mask"):
        if key not in root:
            raise KeyError(f"Missing key '{key}' in {store_path}.")
    query_arr = root["query_world_xyz"]
    traj_arr = root["traj_3d_xyz"]
    valid_arr = root["valid_mask"]
    n = int(query_arr.shape[0])
    if int(traj_arr.shape[0]) != n or int(valid_arr.shape[0]) != n:
        raise ValueError(f"Leading dimensions mismatch in {store_path}.")
    if query_arr.shape[-1] != 3 or traj_arr.shape[-1] != 3:
        raise ValueError(f"Expected XYZ arrays to end with 3 in {store_path}.")
    if valid_arr.shape != traj_arr.shape[:-1]:
        raise ValueError(
            f"Expected valid_mask shape {traj_arr.shape[:-1]}, got {valid_arr.shape} "
            f"in {store_path}."
        )
    return query_arr, traj_arr, valid_arr


def compute_robotwin2_normalization_stats(
    source_root: Path,
    percentiles: Tuple[float, float] = DEFAULT_PERCENTILES,
    num_bins: int = DEFAULT_NUM_BINS,
    max_samples: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, object]:
    if not source_root.exists():
        raise FileNotFoundError(f"RobotWin2 source root does not exist: {source_root}")
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")

    stores = list(iter_robotwin2_zarr_stores(source_root))
    query_min = np.full(3, np.inf, dtype=np.float64)
    query_max = np.full(3, -np.inf, dtype=np.float64)
    traj_min = np.full(3, np.inf, dtype=np.float64)
    traj_max = np.full(3, -np.inf, dtype=np.float64)
    query_count = 0
    traj_count = 0
    num_samples = 0
    remaining = None if max_samples is None else int(max_samples)
    processed_stores: List[str] = []

    for store_path in stores:
        if remaining is not None and remaining <= 0:
            break
        root = zarr.open(str(store_path), mode="r")
        query_arr, traj_arr, valid_arr = _check_store_shapes(store_path, root)
        store_take = int(query_arr.shape[0]) if remaining is None else min(int(query_arr.shape[0]), remaining)
        num_samples += store_take
        processed_stores.append(str(store_path))
        for query_chunk, traj_chunk, valid_chunk in _iter_store_samples(
            query_arr, traj_arr, valid_arr, chunk_size=chunk_size, remaining_limit=remaining
        ):
            for i in range(query_chunk.shape[0]):
                query_centered = _finite_query_centered_xyz(query_chunk[i])
                traj_delta = _valid_traj_delta_xyz(traj_chunk[i], valid_chunk[i])
                query_count += _update_min_max_count(query_centered, query_min, query_max)
                traj_count += _update_min_max_count(traj_delta, traj_min, traj_max)
        if remaining is not None:
            remaining -= store_take

    if not processed_stores:
        raise RuntimeError(f"No readable RobotWin2 zarr stores were processed under {source_root}.")
    if query_count <= 0 or traj_count <= 0:
        raise RuntimeError("Could not collect any valid RobotWin2 normalization values.")

    query_hist = np.zeros((3, num_bins), dtype=np.int64)
    traj_hist = np.zeros((3, num_bins), dtype=np.int64)
    remaining = None if max_samples is None else int(max_samples)

    for store_path_str in processed_stores:
        if remaining is not None and remaining <= 0:
            break
        store_path = Path(store_path_str)
        root = zarr.open(str(store_path), mode="r")
        query_arr, traj_arr, valid_arr = _check_store_shapes(store_path, root)
        store_take = int(query_arr.shape[0]) if remaining is None else min(int(query_arr.shape[0]), remaining)
        for query_chunk, traj_chunk, valid_chunk in _iter_store_samples(
            query_arr, traj_arr, valid_arr, chunk_size=chunk_size, remaining_limit=remaining
        ):
            for i in range(query_chunk.shape[0]):
                query_centered = _finite_query_centered_xyz(query_chunk[i])
                traj_delta = _valid_traj_delta_xyz(traj_chunk[i], valid_chunk[i])
                _accumulate_histogram(query_centered, query_hist, query_min, query_max)
                _accumulate_histogram(traj_delta, traj_hist, traj_min, traj_max)
        if remaining is not None:
            remaining -= store_take

    return {
        "source_root": str(source_root),
        "source_zarr_paths": processed_stores,
        "num_files": int(num_samples),
        "num_stores": int(len(processed_stores)),
        "percentile_range": [float(percentiles[0]), float(percentiles[1])],
        "estimation_method": {
            "type": "two_pass_histogram",
            "num_bins": int(num_bins),
            "chunk_size": int(chunk_size),
            "note": "query_world_xyz is centered per sample before accumulating statistics.",
        },
        "query_centered_xyz": _finalize_axis_summary(
            query_hist,
            query_min,
            query_max,
            total_count=query_count,
            percentiles=percentiles,
        ).__dict__,
        "traj_delta_xyz": _finalize_axis_summary(
            traj_hist,
            traj_min,
            traj_max,
            total_count=traj_count,
            percentiles=percentiles,
        ).__dict__,
    }


def save_robotwin2_normalization_stats(stats: Dict[str, object], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RobotWin2 Motion Expert percentile normalization statistics."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_ROBOTWIN2_SOURCE_ROOT,
        help=(
            "Root containing RobotWin2 motion zarr stores (*.zarr), or one .zarr store. "
            "Defaults to train_rechunked8.zarr for train-only statistics."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to save the computed normalization JSON.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Histogram bin count used for the streaming percentile estimate.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of samples to read from zarr per chunk.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests or quick iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_robotwin2_normalization_stats(
        source_root=args.source_root,
        num_bins=int(args.num_bins),
        max_samples=args.max_samples,
        chunk_size=int(args.chunk_size),
    )
    save_robotwin2_normalization_stats(stats, args.output_json)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
