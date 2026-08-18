#!/usr/bin/env python3
"""Compute Calvin percentile normalization statistics from motion-annotation NPZs.

This script is intentionally streaming and disk-safe. The Calvin corpus contains
hundreds of thousands of NPZs, so exact percentile computation by materializing
all values in memory is not practical. We instead use a two-pass histogram
estimate with configurable bin count.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


DEFAULT_CALVIN_SOURCE_ROOT = Path("/data/user_data/ziyinx/datasets/motion_annotations/calvin")
DEFAULT_PERCENTILES = (0.5, 99.5)
DEFAULT_NUM_BINS = 131072


@dataclass
class AxisSummary:
    count: int
    min: List[float]
    max: List[float]
    p005: List[float]
    p995: List[float]
    center: List[float]
    scale: List[float]


def iter_calvin_annotation_files(
    source_root: Path,
    max_files: Optional[int] = None,
) -> Iterator[Path]:
    count = 0
    units_root = source_root / "units"
    if units_root.exists():
        for unit_dir in sorted(p for p in units_root.iterdir() if p.is_dir()):
            annotations_dir = unit_dir / "annotations"
            if not annotations_dir.exists():
                continue
            for path in sorted(annotations_dir.glob("*.npz")):
                yield path
                count += 1
                if max_files is not None and count >= max_files:
                    return
    if count > 0:
        return
    # Fallback to rglob in case the tree shape changes slightly.
    for path in sorted(source_root.rglob("annotations/*.npz")):
        yield path
        count += 1
        if max_files is not None and count >= max_files:
            return


def _finite_query_centered_xyz(query_world_xyz: np.ndarray) -> np.ndarray:
    query_world_xyz = np.asarray(query_world_xyz, dtype=np.float32)
    if query_world_xyz.ndim != 2 or query_world_xyz.shape[-1] != 3:
        raise ValueError(f"Expected query_world_xyz to have shape [N, 3], got {query_world_xyz.shape}.")
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
        raise ValueError(f"Expected traj_3d_xyz to have shape [N, T, 3], got {traj_3d_xyz.shape}.")
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
    name: str,
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


def compute_calvin_normalization_stats(
    source_root: Path,
    percentiles: Tuple[float, float] = DEFAULT_PERCENTILES,
    num_bins: int = DEFAULT_NUM_BINS,
    max_files: Optional[int] = None,
) -> Dict[str, object]:
    if not source_root.exists():
        raise FileNotFoundError(f"Calvin source root does not exist: {source_root}")
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1.")

    query_min = np.full(3, np.inf, dtype=np.float64)
    query_max = np.full(3, -np.inf, dtype=np.float64)
    traj_min = np.full(3, np.inf, dtype=np.float64)
    traj_max = np.full(3, -np.inf, dtype=np.float64)
    query_count = 0
    traj_count = 0
    num_files = 0

    found_file = False
    for npz_path in iter_calvin_annotation_files(source_root, max_files=max_files):
        found_file = True
        num_files += 1
        with np.load(npz_path, allow_pickle=False) as data:
            query_centered = _finite_query_centered_xyz(data["query_world_xyz"])
            traj_delta = _valid_traj_delta_xyz(data["traj_3d_xyz"], data["valid_mask"])
        query_count += _update_min_max_count(query_centered, query_min, query_max)
        traj_count += _update_min_max_count(traj_delta, traj_min, traj_max)

    if not found_file:
        raise RuntimeError(f"No Calvin annotation NPZ files were found under {source_root}.")
    if query_count <= 0 or traj_count <= 0:
        raise RuntimeError("Could not collect any valid Calvin normalization values.")

    query_hist = np.zeros((3, num_bins), dtype=np.int64)
    traj_hist = np.zeros((3, num_bins), dtype=np.int64)

    for npz_path in iter_calvin_annotation_files(source_root, max_files=max_files):
        with np.load(npz_path, allow_pickle=False) as data:
            query_centered = _finite_query_centered_xyz(data["query_world_xyz"])
            traj_delta = _valid_traj_delta_xyz(data["traj_3d_xyz"], data["valid_mask"])
        _accumulate_histogram(query_centered, query_hist, query_min, query_max)
        _accumulate_histogram(traj_delta, traj_hist, traj_min, traj_max)

    result = {
        "source_root": str(source_root),
        "num_files": int(num_files),
        "percentile_range": [float(percentiles[0]), float(percentiles[1])],
        "estimation_method": {
            "type": "two_pass_histogram",
            "num_bins": int(num_bins),
        },
        "query_centered_xyz": _finalize_axis_summary(
            "query_centered_xyz",
            query_hist,
            query_min,
            query_max,
            total_count=query_count,
            percentiles=percentiles,
        ).__dict__,
        "traj_delta_xyz": _finalize_axis_summary(
            "traj_delta_xyz",
            traj_hist,
            traj_min,
            traj_max,
            total_count=traj_count,
            percentiles=percentiles,
        ).__dict__,
    }
    return result


def _iter_zarr_frame_values(
    zarr_paths,
    *,
    camera_frame,
    traj_is_delta,
    mem_limit_gb,
    chunk_size,
    max_samples,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(query_centered, traj_delta)`` value blocks from motion zarrs.

    Applies the same world -> camera re-anchoring the dataset applies at load
    time, in the same order (transform, then per-sample centering), so the
    resulting statistics describe exactly what the model will see.
    """
    from datasets.utils import read_zarr_with_cache

    from .zarr_dataset import apply_camera_frame_np, resolve_camera_frame_matrix

    world2cam = resolve_camera_frame_matrix(camera_frame)
    remaining = None if max_samples is None else int(max_samples)

    for zarr_path in zarr_paths:
        if remaining is not None and remaining <= 0:
            break
        annos = read_zarr_with_cache(str(zarr_path), mem_gb=mem_limit_gb)
        query_arr = annos["query_world_xyz"]
        traj_arr = annos["traj_3d_xyz"]
        mask_arr = annos["traj_mask" if "traj_mask" in annos else "valid_mask"]

        store_len = len(query_arr)
        if remaining is not None:
            store_len = min(store_len, remaining)
        for start in range(0, store_len, chunk_size):
            end = min(start + chunk_size, store_len)

            q = np.asarray(query_arr[start:end], dtype=np.float32)
            q = q.reshape(q.shape[0], -1, 3)
            q = apply_camera_frame_np(q, world2cam, is_delta=False)
            centered = []
            for sample in q:
                finite = np.isfinite(sample).all(axis=-1)
                if not finite.any():
                    continue
                valid = sample[finite]
                centered.append(valid - valid.mean(axis=0, keepdims=True))
            query_values = (
                np.concatenate(centered, axis=0)
                if centered
                else np.empty((0, 3), dtype=np.float32)
            )

            t = np.asarray(traj_arr[start:end], dtype=np.float32)
            t = t.reshape(t.shape[0], -1, t.shape[-2], 3)
            t = apply_camera_frame_np(t, world2cam, is_delta=traj_is_delta)
            m = np.asarray(mask_arr[start:end])
            if m.shape == t.shape:
                m = m.all(axis=-1)
            m = m.reshape(t.shape[0], -1, t.shape[-2]).astype(bool)
            traj_values = t[m & np.isfinite(t).all(axis=-1)]

            yield query_values, traj_values
        if remaining is not None:
            remaining -= store_len


def compute_calvin_zarr_percentile_stats(
    zarr_paths,
    percentiles: Tuple[float, float] = DEFAULT_PERCENTILES,
    num_bins: int = DEFAULT_NUM_BINS,
    camera_frame: Optional[str] = None,
    traj_is_delta: bool = True,
    mem_limit_gb: float = 16.0,
    chunk_size: int = 512,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Percentile normalization statistics estimated from motion zarrs.

    Same two-pass histogram estimator as ``compute_calvin_normalization_stats``
    (which reads the raw NPZs), so world- and camera-frame runs can share one
    definition of center/scale and stay comparable. The NPZ path cannot express
    a frame transform, which is why this zarr-sourced twin exists.

    ``max_samples`` caps how many rows are read. The rechunked stores are row
    shuffled, so a prefix is an unbiased sample -- 0.5/99.5 percentiles are
    stable well below the full corpus and a capped pass is far cheaper than
    streaming ~100 GB twice.
    """
    paths = [str(p) for p in zarr_paths]
    kwargs = dict(
        camera_frame=camera_frame,
        traj_is_delta=traj_is_delta,
        mem_limit_gb=mem_limit_gb,
        chunk_size=chunk_size,
        max_samples=max_samples,
    )

    # Pass 1: extent + counts (histogram bin edges need the true min/max).
    query_min = np.full(3, np.inf, dtype=np.float64)
    query_max = np.full(3, -np.inf, dtype=np.float64)
    traj_min = np.full(3, np.inf, dtype=np.float64)
    traj_max = np.full(3, -np.inf, dtype=np.float64)
    query_count = 0
    traj_count = 0
    for query_values, traj_values in _iter_zarr_frame_values(paths, **kwargs):
        query_count += _update_min_max_count(query_values, query_min, query_max)
        traj_count += _update_min_max_count(traj_values, traj_min, traj_max)

    if query_count <= 0 or traj_count <= 0:
        raise RuntimeError(
            "Could not compute Calvin zarr percentile statistics: no valid values."
        )

    # Pass 2: histograms over the now-known extent.
    query_hist = np.zeros((3, num_bins), dtype=np.int64)
    traj_hist = np.zeros((3, num_bins), dtype=np.int64)
    for query_values, traj_values in _iter_zarr_frame_values(paths, **kwargs):
        _accumulate_histogram(query_values, query_hist, query_min, query_max)
        _accumulate_histogram(traj_values, traj_hist, traj_min, traj_max)

    return {
        "source_zarr_paths": paths,
        "camera_frame": "world" if camera_frame in (None, "", "none", "world") else str(camera_frame),
        "traj_is_delta": bool(traj_is_delta),
        "max_samples": None if max_samples is None else int(max_samples),
        "percentile_range": [float(percentiles[0]), float(percentiles[1])],
        "estimation_method": {
            "type": "two_pass_histogram",
            "num_bins": int(num_bins),
            "source": "zarr",
        },
        "query_centered_xyz": _finalize_axis_summary(
            "query_centered_xyz",
            query_hist,
            query_min,
            query_max,
            total_count=query_count,
            percentiles=percentiles,
        ).__dict__,
        "traj_delta_xyz": _finalize_axis_summary(
            "traj_delta_xyz",
            traj_hist,
            traj_min,
            traj_max,
            total_count=traj_count,
            percentiles=percentiles,
        ).__dict__,
    }


def save_calvin_normalization_stats(stats: Dict[str, object], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2) + "\n")


def load_calvin_normalization_stats(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def extract_calvin_normalization_tensors(
    stats: Dict[str, object],
    zero_centered_geometry: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = stats["query_centered_xyz"]
    traj = stats["traj_delta_xyz"]
    query_center = np.asarray(query["center"], dtype=np.float32)
    traj_center = np.asarray(traj["center"], dtype=np.float32)
    if zero_centered_geometry:
        # Query XYZ is already centered per sample, and future scene-flow
        # deltas are naturally zero-centered, so we only keep dataset scale.
        query_center = np.zeros_like(query_center)
        traj_center = np.zeros_like(traj_center)
    return (
        query_center,
        np.asarray(query["scale"], dtype=np.float32),
        traj_center,
        np.asarray(traj["scale"], dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Calvin percentile normalization statistics.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_CALVIN_SOURCE_ROOT,
        help="Calvin motion-annotation root containing units/*/annotations/*.npz.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default="/data/user_data/ziyinx/analogical_manipulation_clean/datasets/calvin_motion_normalization_stats.json",
        help="Where to save the computed normalization JSON.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Histogram bin count used for the streaming percentile estimate.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for smoke tests or quick iteration.",
    )
    parser.add_argument(
        "--zarr",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Motion zarr store(s) to estimate from instead of --source-root. "
            "Required for camera-frame stats: the NPZ path cannot apply a "
            "frame transform."
        ),
    )
    parser.add_argument(
        "--camera-frame",
        type=str,
        default=None,
        help="'world' (default) or 'calvin_static'. Only valid with --zarr.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Cap on rows read from each zarr. Rechunked stores are row "
            "shuffled, so a prefix is an unbiased sample."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Rows read per zarr slice.",
    )
    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=2.0,
        help=(
            "LRU chunk-cache budget. Both passes touch every chunk exactly "
            "once in order, so caching buys nothing here -- the default is "
            "deliberately small. Raising it to the training-time 16 GB makes "
            "the process grow until the host kills it."
        ),
    )
    parser.add_argument(
        "--traj-absolute",
        action="store_true",
        help=(
            "Treat traj_3d_xyz as absolute positions rather than deltas. "
            "Deltas (the default) rotate but must not pick up the translation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.zarr:
        stats = compute_calvin_zarr_percentile_stats(
            args.zarr,
            num_bins=int(args.num_bins),
            camera_frame=args.camera_frame,
            traj_is_delta=not args.traj_absolute,
            mem_limit_gb=float(args.mem_limit_gb),
            chunk_size=int(args.chunk_size),
            max_samples=args.max_samples,
        )
    else:
        if args.camera_frame not in (None, "", "world"):
            parser_error = (
                "--camera-frame requires --zarr; the NPZ annotations are "
                "world-frame and cannot express a transform."
            )
            raise SystemExit(parser_error)
        stats = compute_calvin_normalization_stats(
            source_root=args.source_root,
            num_bins=int(args.num_bins),
            max_files=args.max_files,
        )
    save_calvin_normalization_stats(stats, args.output_json)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
