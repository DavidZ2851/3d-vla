"""Normalization statistics for the calvin_motion_uvd_zarr dataset.

The UVD MotionExpert keeps each query point in the *absolute* camera frame:
``u, v`` are mapped to the [-1, 1] model grid (exactly like the 2D track) and the
static-camera ``depth`` is globally standardized ``(d - mean) / std``. We do NOT
per-sample centroid-center depth: with CALVIN's fixed camera the per-frame mean
depth barely moves (~3.6 cm), so centering removes nothing useful and would
desync depth from the absolute u,v / RGB content (see the 3D ``model.py`` path,
which only centers because unprojected world-xyz has large scene-dependent
translation).

The *target* is the anchor-relative future delta ``traj_uvd - query_uvd``:
``du, dv`` in model-grid units, ``dd`` in metres. Each channel is divided by its
own std (``traj_delta_uvd_scale``) so the rectified-flow target is ~unit scale.

Outputs a JSON with::

    {
      "query_depth_mean": <float>,
      "query_depth_std":  <float>,
      "traj_delta_uvd_scale": [du_std, dv_std, dd_std],
      ...
    }

Run::

    python -m motion_expert.compute_calvin_uvd_norm_stats \
        --zarr /data/.../calvin_motion_uvd_zarr/train_rechunked8.zarr \
        --target-hw 224 224 --num-rows 8000 \
        --out motion_expert/configs/calvin_uvd_norm_stats.json
"""
import argparse
import json
from typing import Tuple

import numpy as np
import zarr

from .geometry import compute_resize_pad_params


def _source_px_to_model_grid(
    uv_px: np.ndarray,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Replicate aspect_resize_and_pad's source_pixels -> [-1, 1] grid mapping (numpy)."""
    scale, _, _, pad_top, pad_left = compute_resize_pad_params(src_hw, target_hw)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    uv = uv_px.astype(np.float64).copy()
    uv = uv * scale
    uv[..., 0] += pad_left
    uv[..., 1] += pad_top
    denom = np.asarray([max(target_w - 1, 1), max(target_h - 1, 1)], dtype=np.float64)
    return uv / denom * 2.0 - 1.0


def compute_calvin_uvd_norm_stats(
    zarr_path: str,
    target_hw: Tuple[int, int] = (224, 224),
    num_rows: int = 8000,
    seed: int = 0,
) -> dict:
    z = zarr.open(zarr_path, mode="r")
    for key in ("rgb", "query_uv", "traj_3d_uvd", "valid_mask"):
        if key not in z:
            raise KeyError(f"uvd zarr missing required key `{key}` in {zarr_path}.")

    n_total = int(z["query_uv"].shape[0])
    src_h = int(z["rgb"].shape[-2])
    src_w = int(z["rgb"].shape[-1])
    src_hw = (src_h, src_w)
    target_hw = (int(target_hw[0]), int(target_hw[1]))

    num_rows = min(int(num_rows), n_total)
    rows = np.unique(np.linspace(0, n_total - 1, num_rows).astype(np.int64))

    # query-depth (absolute, metres) running stats
    qd_sum = 0.0
    qd_sumsq = 0.0
    qd_count = 0
    # per-axis anchor-relative delta running stats [du(grid), dv(grid), dd(metres)]
    d_sum = np.zeros(3, dtype=np.float64)
    d_sumsq = np.zeros(3, dtype=np.float64)
    d_count = 0

    block = 256
    for start in range(0, len(rows), block):
        idx = rows[start : start + block]
        qv = np.asarray(z["query_uv"].oindex[idx]).astype(np.float64)   # (S,P,3)
        tr = np.asarray(z["traj_3d_uvd"].oindex[idx]).astype(np.float64)  # (S,P,T,3)
        vm = np.asarray(z["valid_mask"].oindex[idx]).astype(bool)         # (S,P,T)

        # ---- query depth (use finite values only) ----
        qd = qv[..., 2]
        finite = np.isfinite(qd)
        qd_v = qd[finite]
        qd_sum += float(qd_v.sum())
        qd_sumsq += float(np.square(qd_v).sum())
        qd_count += int(qd_v.size)

        # ---- anchor-relative deltas over valid future steps ----
        q_grid = _source_px_to_model_grid(qv[..., :2], src_hw, target_hw)   # (S,P,2)
        t_grid = _source_px_to_model_grid(tr[..., :2], src_hw, target_hw)   # (S,P,T,2)
        delta_uv = t_grid - q_grid[:, :, None, :]                          # (S,P,T,2)
        delta_d = tr[..., 2] - qv[..., 2][:, :, None]                       # (S,P,T)
        delta = np.concatenate([delta_uv, delta_d[..., None]], axis=-1)     # (S,P,T,3)

        finite_d = np.isfinite(delta).all(axis=-1)
        m = vm & finite_d
        vals = delta[m].reshape(-1, 3)
        if vals.size:
            d_sum += vals.sum(axis=0)
            d_sumsq += np.square(vals).sum(axis=0)
            d_count += int(vals.shape[0])

    if qd_count == 0 or d_count == 0:
        raise RuntimeError("No valid query depth / delta samples found.")

    qd_mean = qd_sum / qd_count
    qd_std = float(np.sqrt(max(qd_sumsq / qd_count - qd_mean * qd_mean, 1e-12)))

    d_mean = d_sum / d_count
    d_var = np.maximum(d_sumsq / d_count - np.square(d_mean), 1e-12)
    d_std = np.sqrt(d_var)

    return {
        "zarr": str(zarr_path),
        "num_rows_sampled": int(len(rows)),
        "src_hw": [src_h, src_w],
        "target_hw": [target_hw[0], target_hw[1]],
        "query_depth_mean": float(qd_mean),
        "query_depth_std": float(qd_std),
        "query_depth_count": int(qd_count),
        "traj_delta_uvd_mean": [float(x) for x in d_mean],
        "traj_delta_uvd_scale": [float(x) for x in d_std],
        "traj_delta_count": int(d_count),
        "channels": "u_grid,v_grid,depth_metres",
        "notes": "query depth is GLOBAL-standardized (no centroid centering); "
        "delta uv in [-1,1] grid units, delta depth in metres.",
    }


def main():
    p = argparse.ArgumentParser(description="Compute calvin UVD motion-expert norm stats.")
    p.add_argument(
        "--zarr",
        default="/data/user_data/ziyinx/datasets/zarr_datasets/calvin_motion_uvd_zarr/train_rechunked8.zarr",
    )
    p.add_argument("--target-hw", nargs=2, type=int, default=[224, 224])
    p.add_argument("--num-rows", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional path to write JSON stats.")
    args = p.parse_args()

    stats = compute_calvin_uvd_norm_stats(
        zarr_path=args.zarr,
        target_hw=tuple(args.target_hw),
        num_rows=args.num_rows,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as h:
            json.dump(stats, h, indent=2)
        print(f"[uvd-stats] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
