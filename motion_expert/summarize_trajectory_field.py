#!/usr/bin/env python3
"""Summarise eval_trajectory_field runs: all-points vs moved-only, plus the
zero-motion reference (what "predict that nothing moves" would score).

The zero-motion error at step t is just ||cumsum(gt_delta)[t]||, so it comes
straight from the label zarr and needs no model. Reported per run so the model
numbers have an absolute reference rather than only relative ones.

Subsets are kept explicit because the two headline metrics use different ones:
  A ("endpoint")  - points whose track is valid across the whole K-step prefix
  B ("point mean")- points with at least one valid step; displacement measured
                    at that point's last valid step
Compare within a row, never across rows.

  python -m motion_expert.summarize_trajectory_field \
      --runs eval_logs/motion_field_local/calvin_h15_h15_s10_seed0 \
      --zarr /path/to/val_rechunked8.zarr --threshold 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run stems (without .json/.npz) produced by eval_trajectory_field.",
    )
    p.add_argument(
        "--zarr",
        default=None,
        help="Label zarr for the zero-motion reference. Defaults to each run's recorded val_zarr.",
    )
    p.add_argument("--threshold", type=float, default=2.0, help="Moved cut in cm.")
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--row-chunk", type=int, default=500)
    p.add_argument(
        "--rows",
        default=None,
        help="Optional 'start:end' row slice, or a .npy of row indices (RoboTwin2 task filter).",
    )
    return p.parse_args()


def load_row_indices(spec: Optional[str], num_rows: int) -> Optional[np.ndarray]:
    if spec is None:
        return None
    if spec.endswith(".npy"):
        return np.load(spec)
    start, end = spec.split(":")
    return np.arange(int(start or 0), int(end or num_rows))


def zero_motion_reference(
    zarr_path: str,
    horizon: int,
    threshold_cm: float,
    row_chunk: int,
    row_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    store = zarr.open(zarr_path, mode="r")
    total = store["traj_3d_xyz"].shape[0]
    rows = np.arange(total) if row_indices is None else np.asarray(row_indices)

    sum_end = cnt_a = sum_a_steps = cnt_a_steps = 0.0
    sum_b = cnt_b = 0.0
    for start in range(0, rows.size, row_chunk):
        idx = rows[start : start + row_chunk]
        sel = slice(int(idx[0]), int(idx[-1]) + 1) if np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)) else idx
        delta = np.asarray(store["traj_3d_xyz"][sel], dtype=np.float64)[:, :, :horizon]
        mask = np.asarray(store["valid_mask"][sel]).astype(bool)[:, :, :horizon]
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        delta[~mask] = 0.0

        prefix = np.cumprod(mask.astype(np.uint8), axis=2).astype(bool)
        disp = np.linalg.norm(np.cumsum(delta, axis=2), axis=-1) * 100.0

        full = prefix[:, :, -1]
        sub_a = full & (disp[:, :, -1] > threshold_cm)
        sum_end += disp[:, :, -1][sub_a].sum()
        cnt_a += sub_a.sum()
        sum_a_steps += disp[sub_a].sum()
        cnt_a_steps += sub_a.sum() * horizon

        counts = prefix.sum(axis=2)
        has_any = counts > 0
        last = np.clip(counts - 1, 0, None)
        disp_last = np.take_along_axis(disp, last[..., None], axis=2)[..., 0]
        sub_b = has_any & (disp_last > threshold_cm)
        point_mean = (disp * prefix).sum(axis=2) / np.maximum(counts, 1)
        sum_b += point_mean[sub_b].sum()
        cnt_b += sub_b.sum()

    return {
        "endpoint_cm": sum_end / max(cnt_a, 1.0),
        "mean_steps_cm_subsetA": sum_a_steps / max(cnt_a_steps, 1.0),
        "mean_steps_cm_subsetB": sum_b / max(cnt_b, 1.0),
        "num_points_A": int(cnt_a),
        "num_points_B": int(cnt_b),
    }


def _delta(model: float, ref: float) -> str:
    if not np.isfinite(ref) or ref == 0:
        return "     n/a"
    return f"{(model / ref - 1) * 100:+7.1f}%"


def main() -> None:
    args = parse_args()
    thr = args.threshold

    for stem in args.runs:
        stem_path = Path(stem)
        summary = json.loads((stem_path.with_suffix(".json")).read_text())
        arrays = np.load(stem_path.with_suffix(".npz"))
        run = summary["run"]

        end_err = arrays["endpoint_err_cm"].astype(np.float64)
        end_disp = arrays["endpoint_gt_disp_cm"].astype(np.float64)
        pt_err = arrays["point_mean_err_cm"].astype(np.float64)
        pt_disp = arrays["point_gt_disp_cm"].astype(np.float64)

        moved_a = end_disp > thr
        moved_b = pt_disp > thr

        zarr_path = args.zarr or run["val_zarr"]
        row_idx = load_row_indices(args.rows, 0)
        if row_idx is None and run.get("tasks"):
            print(
                f"[warn] {run['tag']}: run used a task filter but --rows was not given; "
                "the zero-motion reference would cover all rows. Skipping it.",
                flush=True,
            )
            ref = None
        else:
            ref = zero_motion_reference(zarr_path, args.horizon, thr, args.row_chunk, row_idx)

        print("=" * 78)
        print(f"{run['tag']}   ({summary['num_rows']} rows, {run['num_inference_steps']} decode steps)")
        print(f"  all points   : mean {summary['mean_first_K_cm']:.3f} cm | endpoint {summary['endpoint_K_cm']:.3f} cm")
        print(f"  moved >{thr:g}cm  : {moved_a.mean() * 100:.1f}% of scored points")
        print(f"     ENDPOINT @{args.horizon}      : {end_err[moved_a].mean():7.3f} cm", end="")
        if ref:
            print(f"   [zero-motion {ref['endpoint_cm']:7.3f} cm  {_delta(end_err[moved_a].mean(), ref['endpoint_cm'])}]")
        else:
            print()
        print(f"     MEAN over steps   : {pt_err[moved_b].mean():7.3f} cm", end="")
        if ref:
            print(f"   [zero-motion {ref['mean_steps_cm_subsetB']:7.3f} cm  {_delta(pt_err[moved_b].mean(), ref['mean_steps_cm_subsetB'])}]")
        else:
            print()
        print(f"     GT displacement   : {end_disp[moved_a].mean():7.3f} cm  "
              f"-> relative error {end_err[moved_a].mean() / end_disp[moved_a].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
