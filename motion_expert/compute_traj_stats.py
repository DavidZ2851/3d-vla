#!/usr/bin/env python3
"""Dataset statistics utility for trajectory tracking and normalization.

This script computes:
1) Total number of NPZ datapoints in a tracking dataset root.
2) Number of datapoints where mean(traj_3d_xyz) < threshold.
3) Streaming normalization statistics for traj_3d_xyz from MotionExpert data sources.

Supported MotionExpert traj sources:
- One or more .zarr roots (recommended).
- A YAML/JSON train config with data.train_zarr / data.val_zarr.
- NPZ file(s) or directories containing NPZ files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class AxisStats:
    count: int
    mean: List[float]
    std: List[float]
    min: List[float]
    max: List[float]


@dataclass
class Summary:
    total_samples: int
    skipped_samples: int
    traj_stats: AxisStats
    delta_step_abs_max: List[float]


def _iter_npz_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.npz")


def _safe_float_array(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x
    return x.astype(np.float64, copy=False)


def count_tracking_npz(
    root: Path,
    key: str,
    mean_threshold: float,
) -> Dict[str, int]:
    total = 0
    low_abs_mean = 0
    missing_key = 0
    invalid = 0
    inf_mean = 0
    nan_mean = 0

    for npz_path in _iter_npz_files(root):
        total += 1
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if key not in data:
                    missing_key += 1
                    continue
                arr = _safe_float_array(np.asarray(data[key]))
                m = float(np.mean(arr))
                if np.isinf(m):
                    inf_mean += 1
                    continue
                if np.isnan(m):
                    nan_mean += 1
                    continue
                if abs(m) < mean_threshold:
                    low_abs_mean += 1
        except Exception:
            invalid += 1

    return {
        "total_npz_datapoints": total,
        f"num_abs_mean_lt_{mean_threshold:g}": low_abs_mean,
        "excluded_inf_mean": inf_mean,
        "excluded_nan_mean": nan_mean,
        "missing_key": missing_key,
        "invalid_or_failed": invalid,
    }


def _load_config(path: Path) -> dict:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs.")
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format: {path}")


def _collect_motion_expert_sources(
    config_path: Optional[Path],
    zarr_paths: Sequence[str],
    explicit_paths: Sequence[str],
) -> List[Path]:
    sources: List[Path] = []

    if config_path is not None:
        cfg = _load_config(config_path)
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
        for k in ("train_zarr", "val_zarr"):
            p = data_cfg.get(k)
            if p:
                sources.append(Path(str(p)))

    for p in zarr_paths:
        sources.append(Path(p))
    for p in explicit_paths:
        sources.append(Path(p))

    # Keep order, de-duplicate.
    uniq: List[Path] = []
    seen = set()
    for p in sources:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _iter_traj_from_npz(path: Path, key: str) -> Iterator[np.ndarray]:
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key in data:
                yield _safe_float_array(np.asarray(data[key]))
        return

    if path.is_dir():
        for npz_path in _iter_npz_files(path):
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if key in data:
                        yield _safe_float_array(np.asarray(data[key]))
            except Exception:
                continue


def _looks_like_zarr(path: Path) -> bool:
    return path.is_dir() and (
        (path / ".zgroup").exists()
        or (path / "zarr.json").exists()
        or path.suffix == ".zarr"
    )


def _iter_traj_from_zarr(
    zarr_root: Path,
    key: str,
    chunk_size: int,
    mem_limit_gb: float,
) -> Iterator[np.ndarray]:
    try:
        from datasets.utils import read_zarr_with_cache  # local project utility
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "zarr support is unavailable in this environment. Install `zarr` to read .zarr sources."
        ) from exc

    annos = read_zarr_with_cache(str(zarr_root), mem_gb=mem_limit_gb)
    if key not in annos:
        return
    arr = annos[key]
    n = len(arr)
    for start in range(0, n, chunk_size):
        chunk = np.asarray(arr[start : start + chunk_size])
        yield _safe_float_array(chunk)


def _accumulate_axis_stats(
    chunk: np.ndarray,
    sum_xyz: np.ndarray,
    sum_sq_xyz: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
) -> int:
    if chunk.size == 0 or chunk.shape[-1] != 3:
        return 0
    flat = chunk.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    if not finite.any():
        return 0
    flat = flat[finite]
    sum_xyz += flat.sum(axis=0)
    sum_sq_xyz += (flat * flat).sum(axis=0)
    min_xyz[:] = np.minimum(min_xyz, flat.min(axis=0))
    max_xyz[:] = np.maximum(max_xyz, flat.max(axis=0))
    return int(flat.shape[0])


def _accumulate_delta_abs_max(chunk: np.ndarray, delta_abs_max: np.ndarray) -> int:
    # Expected shape includes time on axis -2 and xyz on axis -1.
    if chunk.ndim < 2 or chunk.shape[-1] != 3 or chunk.shape[-2] < 2:
        return 0
    delta = np.diff(chunk, axis=-2)
    flat = delta.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    if not finite.any():
        return 0
    flat = flat[finite]
    delta_abs_max[:] = np.maximum(delta_abs_max, np.abs(flat).max(axis=0))
    return int(flat.shape[0])


def compute_traj_normalization_stats(
    sources: Sequence[Path],
    key: str,
    chunk_size: int,
    mem_limit_gb: float,
) -> Summary:
    sum_xyz = np.zeros(3, dtype=np.float64)
    sum_sq_xyz = np.zeros(3, dtype=np.float64)
    min_xyz = np.full(3, np.inf, dtype=np.float64)
    max_xyz = np.full(3, -np.inf, dtype=np.float64)
    delta_abs_max = np.zeros(3, dtype=np.float64)

    point_count = 0
    total_samples = 0
    skipped_samples = 0

    existing_sources = 0

    for src in sources:
        if not src.exists():
            continue
        existing_sources += 1

        if _looks_like_zarr(src):
            for chunk in _iter_traj_from_zarr(src, key=key, chunk_size=chunk_size, mem_limit_gb=mem_limit_gb):
                if chunk.shape[-1] != 3:
                    skipped_samples += int(chunk.shape[0]) if chunk.ndim > 0 else 1
                    continue
                total_samples += int(chunk.shape[0]) if chunk.ndim > 0 else 1
                point_count += _accumulate_axis_stats(chunk, sum_xyz, sum_sq_xyz, min_xyz, max_xyz)
                _accumulate_delta_abs_max(chunk, delta_abs_max)
            continue

        for arr in _iter_traj_from_npz(src, key=key):
            total_samples += 1
            if arr.shape[-1] != 3:
                skipped_samples += 1
                continue
            point_count += _accumulate_axis_stats(arr, sum_xyz, sum_sq_xyz, min_xyz, max_xyz)
            _accumulate_delta_abs_max(arr, delta_abs_max)

    if existing_sources == 0:
        raise RuntimeError(
            "None of the provided MotionExpert sources exist on disk."
        )

    if total_samples == 0:
        raise RuntimeError(
            f"No samples found for key '{key}'. Check --traj-key and source contents."
        )

    if point_count == 0:
        raise RuntimeError(
            "Samples were found but no valid finite XYZ points could be used."
        )

    mean = sum_xyz / point_count
    var = np.maximum(sum_sq_xyz / point_count - mean * mean, 0.0)
    std = np.sqrt(var)

    traj_stats = AxisStats(
        count=int(point_count),
        mean=mean.tolist(),
        std=std.tolist(),
        min=min_xyz.tolist(),
        max=max_xyz.tolist(),
    )

    return Summary(
        total_samples=int(total_samples),
        skipped_samples=int(skipped_samples),
        traj_stats=traj_stats,
        delta_step_abs_max=delta_abs_max.tolist(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tracking and traj_3d_xyz normalization statistics."
    )

    parser.add_argument(
        "--tracking-root",
        type=Path,
        default=Path("/data/user_data/ziyinx/datasets/motion_annotations"),
        help="Root containing NPZ tracking datapoints.",
    )
    parser.add_argument(
        "--tracking-key",
        type=str,
        default="traj_3d_xyz",
        help="Key used inside NPZ files for tracking count checks.",
    )
    parser.add_argument(
        "--tracking-mean-threshold",
        type=float,
        default=1e-4,
        help="Threshold for counting datapoints where mean(traj_3d_xyz) is lower.",
    )

    parser.add_argument(
        "--motion-expert-config",
        type=Path,
        default=None,
        help="Optional MotionExpert train config (YAML/JSON) to read train_zarr/val_zarr.",
    )
    parser.add_argument(
        "--motion-expert-zarr",
        type=str,
        nargs="*",
        default=[],
        help="Explicit one or more zarr roots for MotionExpert traj stats.",
    )
    parser.add_argument(
        "--motion-expert-path",
        type=str,
        nargs="*",
        default=[],
        help="Additional NPZ file/dir or zarr roots for MotionExpert traj stats.",
    )
    parser.add_argument(
        "--traj-key",
        type=str,
        default="traj_3d_xyz",
        help="Trajectory key used for normalization stats.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size when reading zarr arrays.",
    )
    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=16.0,
        help="LRU cache size used by zarr reader.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tracking = count_tracking_npz(
        root=args.tracking_root,
        key=args.tracking_key,
        mean_threshold=args.tracking_mean_threshold,
    )

    sources = _collect_motion_expert_sources(
        config_path=args.motion_expert_config,
        zarr_paths=args.motion_expert_zarr,
        explicit_paths=args.motion_expert_path,
    )

    if not sources:
        raise RuntimeError(
            "No MotionExpert traj data sources were provided. Use --motion-expert-zarr, "
            "--motion-expert-path, or --motion-expert-config."
        )
    missing_sources = [str(p) for p in sources if not p.exists()]
    if missing_sources:
        raise RuntimeError(
            "The following MotionExpert sources do not exist:\n"
            + "\n".join(missing_sources)
            + "\nProvide real dataset paths (not placeholders like /path/to/train.zarr)."
        )

    summary = compute_traj_normalization_stats(
        sources=sources,
        key=args.traj_key,
        chunk_size=args.chunk_size,
        mem_limit_gb=args.mem_limit_gb,
    )

    result = {
        "tracking": tracking,
        "traj_normalization": {
            "sources": [str(p) for p in sources],
            "total_samples": summary.total_samples,
            "skipped_samples": summary.skipped_samples,
            "traj_3d_xyz": {
                "count": summary.traj_stats.count,
                "mean_xyz": summary.traj_stats.mean,
                "std_xyz": summary.traj_stats.std,
                "min_xyz": summary.traj_stats.min,
                "max_xyz": summary.traj_stats.max,
            },
            # Useful for MotionExpert's delta_range_xyz if normalizing step deltas.
            "delta_step_abs_max_xyz": summary.delta_step_abs_max,
        },
    }

    text = json.dumps(result, indent=2)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
