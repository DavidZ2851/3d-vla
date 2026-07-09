#!/usr/bin/env python3
"""Compute PerAct2 Motion Expert normalization statistics from zarr data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compute_robotwin2_normalization_stats import (
        DEFAULT_NUM_BINS,
        compute_robotwin2_normalization_stats,
        save_robotwin2_normalization_stats,
    )
except ImportError:
    from compute_robotwin2_normalization_stats import (
        DEFAULT_NUM_BINS,
        compute_robotwin2_normalization_stats,
        save_robotwin2_normalization_stats,
    )


DEFAULT_SOURCE_ROOT = Path(
    "/data/user_data/ziyinx/datasets/zarr_datasets/peract2_motion_zarr/train.zarr"
)
DEFAULT_OUTPUT_JSON = Path(
    "/data/user_data/ziyinx/analogical_manipulation_clean/datasets/peract2_motion_normalization_stats.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PerAct2 Motion Expert percentile normalization statistics."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="PerAct2 motion train zarr store, or a directory containing zarr stores.",
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
        default=2048,
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
    stats["dataset"] = "peract2"
    save_robotwin2_normalization_stats(stats, args.output_json)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
