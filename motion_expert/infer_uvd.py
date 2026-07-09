"""Inference for the UVD MotionExpert.

Loads a trained checkpoint and predicts future 15-step UVD point trajectories for
query points on a frame. Two entry points:

  * ``load_uvd_pipeline(checkpoint)`` -> (pipeline, config, raw) for programmatic use.
  * ``python -m motion_expert.infer_uvd --checkpoint ... [--zarr ...]`` runs
    prediction over rows of a calvin_motion_uvd_zarr, reconstructs absolute future
    UVD (u/v pixels + depth metres), reports EPE vs. ground truth and optionally
    saves an ``.npz``.

Prediction returns anchor-relative deltas ``(du, dv, dd)`` with du/dv in [-1,1]
model-grid units and dd in metres. We add them back to the query to recover
absolute model-grid uv + metric depth, then map uv back to source pixels.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import MotionExpertConfig
from .data import MotionExpertBatch
from .train import _build_motion_expert_config
from .geometry import compute_resize_pad_params
from .pipeline_uvd import MotionExpertUVDPipeline
from .zarr_dataset import MotionExpertQwenTokenizer
from .zarr_dataset_uvd import (
    DEFAULT_QUERY_DEPTH_MEAN,
    DEFAULT_QUERY_DEPTH_STD,
    MotionExpertUVDCollator,
    MotionExpertUVDZarrDataset,
)


def model_grid_to_source_px(
    uv_grid: torch.Tensor,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Inverse of zarr_dataset_uvd._source_px_to_model_grid: [-1,1] grid -> source pixels."""
    scale, _, _, pad_top, pad_left = compute_resize_pad_params(src_hw, target_hw)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    denom = uv_grid.new_tensor([max(target_w - 1, 1), max(target_h - 1, 1)])
    uv = (uv_grid + 1.0) * 0.5 * denom
    uv[..., 0] -= pad_left
    uv[..., 1] -= pad_top
    return uv / max(scale, 1e-8)


def load_uvd_pipeline(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    config_path: Optional[str] = None,
) -> Tuple[MotionExpertUVDPipeline, MotionExpertConfig, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw = ckpt.get("config")
    if raw is None:
        if config_path is None:
            raise ValueError("Checkpoint has no embedded config; pass --config.")
        import yaml
        with open(config_path) as h:
            raw = yaml.safe_load(h)
    config = _build_motion_expert_config(raw["model"])
    if config.florence.model_name_or_path:
        os.environ.setdefault("FLORENCE2_MODEL_PATH", config.florence.model_name_or_path)

    pipeline = MotionExpertUVDPipeline.from_backbone(config)
    missing, unexpected = pipeline.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Frozen florence2 params are intentionally absent from the slim checkpoint.
    real_missing = [k for k in missing if not k.startswith("feature_extractor.")]
    if real_missing:
        print(f"[uvd-infer] WARNING missing keys: {real_missing[:8]}{'...' if len(real_missing) > 8 else ''}")
    if unexpected:
        print(f"[uvd-infer] WARNING unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    pipeline = pipeline.to(device).eval()
    print(f"[uvd-infer] loaded {checkpoint_path} step={ckpt.get('step')} "
          f"traj_delta_uvd_scale={pipeline.model.traj_delta_uvd_scale.tolist()}", flush=True)
    return pipeline, config, raw


@torch.no_grad()
def predict_absolute_uvd(
    pipeline: MotionExpertUVDPipeline,
    batch: MotionExpertBatch,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    query_depth_mean: float,
    query_depth_std: float,
    num_steps: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Run the flow sampler and reconstruct absolute future UVD in source pixels + metres.

    Returns dict of CPU tensors:
      pred_delta_uvd  (B,P,T,3) du/dv grid, dd metres
      pred_uv_px      (B,P,T,2) future u,v in source pixels
      pred_depth_m    (B,P,T)   future depth in metres
      query_uv_px     (B,P,2)   query u,v in source pixels
      query_depth_m   (B,P)     query depth in metres
    """
    device = device or next(pipeline.parameters()).device
    batch = batch.to(device)
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        pred_delta = pipeline.predict(batch, num_steps=num_steps).float()   # (B,P,T,3)

    query_uvd = batch.query_world_xyz.float()                                # (B,P,3) grid uv + std depth
    query_grid = query_uvd[..., :2]
    query_depth_m = query_uvd[..., 2] * query_depth_std + query_depth_mean   # (B,P) metres

    future_grid = query_grid.unsqueeze(2) + pred_delta[..., :2]              # (B,P,T,2)
    future_depth_m = query_depth_m.unsqueeze(2) + pred_delta[..., 2]         # (B,P,T)

    query_uv_px = model_grid_to_source_px(query_grid, src_hw, target_hw)
    pred_uv_px = model_grid_to_source_px(future_grid, src_hw, target_hw)

    return {
        "pred_delta_uvd": pred_delta.cpu(),
        "pred_uv_px": pred_uv_px.cpu(),
        "pred_depth_m": future_depth_m.cpu(),
        "query_uv_px": query_uv_px.cpu(),
        "query_depth_m": query_depth_m.cpu(),
    }


def _epe_metrics(pred_uv_px, pred_depth_m, traj_gt, valid):
    """2D pixel EPE + depth-L1 over valid steps. traj_gt (B,P,T,3) px,px,metres."""
    gt_uv = torch.from_numpy(traj_gt[..., :2]).float()
    gt_d = torch.from_numpy(traj_gt[..., 2]).float()
    m = torch.from_numpy(valid).bool()
    epe = torch.linalg.norm(pred_uv_px - gt_uv, dim=-1)        # (B,P,T) pixels
    d_l1 = (pred_depth_m - gt_d).abs()                          # (B,P,T) metres
    if m.any():
        return {
            "uv_epe_px": float(epe[m].mean()),
            "depth_l1_m": float(d_l1[m].mean()),
            "valid_frac": float(m.float().mean()),
        }
    return {"uv_epe_px": float("nan"), "depth_l1_m": float("nan"), "valid_frac": 0.0}


def main():
    p = argparse.ArgumentParser(description="UVD MotionExpert inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="Fallback config if not embedded in ckpt.")
    p.add_argument(
        "--zarr",
        default="/data/user_data/ziyinx/datasets/zarr_datasets/calvin_motion_uvd_zarr/val_rechunked8.zarr",
    )
    p.add_argument("--instructions", default=None)
    p.add_argument("--indices", type=int, nargs="*", default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--out", default=None, help="Optional .npz to save predictions.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline, config, raw = load_uvd_pipeline(args.checkpoint, device=device, config_path=args.config)
    data_cfg = raw.get("data", {})
    instructions = args.instructions or data_cfg.get("val_instructions_path",
                                                     data_cfg.get("instructions_path"))
    if instructions is None:
        raise ValueError("No instructions path (pass --instructions or embed in config).")

    depth_mean = float(data_cfg.get("query_depth_mean", DEFAULT_QUERY_DEPTH_MEAN))
    depth_std = float(data_cfg.get("query_depth_std", DEFAULT_QUERY_DEPTH_STD))
    target_hw = (config.input_height, config.input_width)

    dataset = MotionExpertUVDZarrDataset(
        root=args.zarr, instructions_path=instructions, image_size=target_hw,
        future_steps=config.future_steps, image_pad_value=config.image_pad_value,
        query_depth_mean=depth_mean, query_depth_std=depth_std,
    )
    tokenizer = MotionExpertQwenTokenizer(
        backbone_type="florence2", model_name_or_path=config.florence.model_name_or_path,
        max_length=config.text_max_length,
    )
    collator = MotionExpertUVDCollator(tokenizer)

    indices: List[int] = args.indices if args.indices else list(range(min(args.num_samples, len(dataset))))
    samples = [dataset[i] for i in indices]
    batch = collator(samples)

    import zarr
    z = zarr.open(args.zarr, mode="r")
    src_hw = (int(z["rgb"].shape[-2]), int(z["rgb"].shape[-1]))
    traj_gt = np.stack([np.asarray(z["traj_3d_uvd"][i], dtype=np.float32) for i in indices])  # (B,P,T,3)
    valid = np.stack([np.asarray(z["valid_mask"][i]) for i in indices]).astype(bool)          # (B,P,T)

    out = predict_absolute_uvd(
        pipeline, batch, src_hw=src_hw, target_hw=target_hw,
        query_depth_mean=depth_mean, query_depth_std=depth_std,
        num_steps=args.num_steps, device=device,
    )
    metrics = _epe_metrics(out["pred_uv_px"], out["pred_depth_m"], traj_gt, valid)
    print("[uvd-infer] metrics: " + json.dumps(metrics, indent=2), flush=True)

    if args.out:
        np.savez_compressed(
            args.out,
            indices=np.asarray(indices),
            instructions=np.asarray([s["instruction_text"] for s in samples]),
            pred_delta_uvd=out["pred_delta_uvd"].numpy(),
            pred_uv_px=out["pred_uv_px"].numpy(),
            pred_depth_m=out["pred_depth_m"].numpy(),
            query_uv_px=out["query_uv_px"].numpy(),
            query_depth_m=out["query_depth_m"].numpy(),
            traj_gt_uvd=traj_gt,
            valid_mask=valid,
            **{f"metric_{k}": v for k, v in metrics.items()},
        )
        print(f"[uvd-infer] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
