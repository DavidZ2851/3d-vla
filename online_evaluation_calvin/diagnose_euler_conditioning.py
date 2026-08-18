tat#!/usr/bin/env python3
"""Is the camera frame's Euler parameterisation what hurts the cam policy?

Two error measures per predicted rotation delta:

  euler_err    - L2 in the space the loss actually optimises (what training sees)
  geodesic_err - angle of R_pred^T R_gt in SO(3) (what the robot actually does)

Near the gimbal singularity (|cos e_y| -> 0) the Euler->SO(3) Jacobian blows up,
so a model can look fine on `euler_err` while being badly wrong on
`geodesic_err`. Binning both by |cos e_y| of the anchor pose shows whether a
checkpoint's excess rotation error is concentrated where the parameterisation is
ill-conditioned -- the signature of an Euler problem rather than a generically
weaker model.

Runs offline on the val zarr; no rollout, no retraining.

  python -m online_evaluation_calvin.diagnose_euler_conditioning \
      --checkpoint <policy.pth> --dataset CalvinCamContact10th \
      --model-type flower_actor_3d_motion_latent_cam --embedding-dim 960
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import fetch_dataset_class
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.data_preprocessors import fetch_data_preprocessor
from utils.depth2cloud import fetch_depth2cloud
from utils.trainers.base import base_collate_fn
from online_evaluation_calvin.model_wrapper import (
    _extract_state_dict,
    _strip_module_prefix,
    _reconcile_compile_prefix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="CalvinCamContact10th")
    p.add_argument("--model-type", default="flower_actor_3d_motion_latent_cam")
    p.add_argument("--embedding-dim", type=int, default=960)
    p.add_argument(
        "--val-zarr",
        default="/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr/val_rechunked8.zarr",
    )
    p.add_argument("--val-instructions", default="instructions/calvin/val_instructions.json")
    p.add_argument("--motion-expert-config", default=None)
    p.add_argument("--motion-expert-checkpoint", default=None)
    p.add_argument("--motion-expert-frame", default="camera")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tag", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def euler_to_matrix_np(e: np.ndarray) -> np.ndarray:
    """Extrinsic xyz: R = Rz(e_z) Ry(e_y) Rx(e_x). Matches utils.calvin_camera_frame."""
    ex, ey, ez = e[..., 0], e[..., 1], e[..., 2]
    cx, sx = np.cos(ex), np.sin(ex)
    cy, sy = np.cos(ey), np.sin(ey)
    cz, sz = np.cos(ez), np.sin(ez)
    R = np.empty(e.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle of Ra^T Rb, in degrees."""
    Rrel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    tr = np.trace(Rrel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def build_model(args, device):
    model_cls = fetch_model_class(args.model_type)
    kwargs = dict(
        backbone="florence2",
        num_finetune_vision_layers=0,
        num_vis_instr_attn_layers=3,
        fps_subsampling_factor=1,
        embedding_dim=args.embedding_dim,
        num_attn_heads=16,
        nhist=1,
        nhand=1,
        num_shared_attn_layers=16,
        relative=True,
        rotation_format="euler",
        denoise_timesteps=5,
        denoise_model="rectified_flow",
    )
    import inspect

    params = inspect.signature(model_cls.__init__).parameters
    for k, v in {
        "motion_expert_config": args.motion_expert_config,
        "motion_expert_checkpoint": args.motion_expert_checkpoint,
        "motion_solver_steps": 10,
        "motion_steps_to_run": 1,
        "motion_finetune": True,
        "motion_scene_fusion": True,
        "motion_action_fusion": False,
        "motion_expert_frame": args.motion_expert_frame,
    }.items():
        if k in params:
            kwargs[k] = v
    model = model_cls(**kwargs)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = _strip_module_prefix(_extract_state_dict(ckpt))
    sd = _reconcile_compile_prefix(sd, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[diag] loaded {args.checkpoint}")
    print(f"[diag]   missing={len(missing)} unexpected={len(unexpected)} iter={ckpt.get('iter')}")
    del ckpt, sd
    return model.to(device).eval()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")

    preprocessor = fetch_data_preprocessor(args.dataset)(
        False, 1, custom_imsize=224, depth2cloud=fetch_depth2cloud(args.dataset)
    )
    dataset = fetch_dataset_class(args.dataset)(
        root=args.val_zarr,
        instructions=args.val_instructions,
        copies=1,
        relative_action=True,
        mem_limit=0.1,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size // args.chunk_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=base_collate_fn,
    )
    tokenizer = fetch_tokenizers("florence2")
    model = build_model(args, device)

    eul_err, geo_err, cos_ey, gt_mag = [], [], [], []
    for bi, sample in enumerate(loader):
        raw_proprio = sample["proprioception"]
        action = preprocessor.process_rel_actions(sample["action"], raw_proprio)
        proprio = preprocessor.process_proprio(raw_proprio)
        rgbs, pcds = preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"], sample["depth"],
            sample["wrist_depth"], sample["extrinsics_wrist"], augment=False,
        )
        # Mirrors BaseTrainTester._model_forward: some tokenizers return a bare
        # tensor of ids, others a BatchEncoding with an attention mask.
        tok = tokenizer(sample["instr"])
        if isinstance(tok, torch.Tensor):
            instr, instr_mask, instr_lens = tok.cuda(non_blocking=True), None, None
        else:
            instr = tok.input_ids.cuda(non_blocking=True)
            instr_mask = tok.attention_mask.cuda(non_blocking=True)
            instr_lens = tuple(int(x) for x in tok.attention_mask.sum(dim=1).tolist())

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(
                action, torch.zeros(action.shape[:-1], dtype=bool, device="cuda"),
                rgbs, None, pcds, instr, proprio,
                run_inference=True, instr_mask=instr_mask, instr_lens=instr_lens,
                grounding_3d=None, grounding_keep_mask=None,
            )

        p = pred.float().cpu().numpy()
        g = action.float().cpu().numpy()
        # (B, T, nhand, 3+3+1) -> per-step euler deltas
        pe, ge = p[..., 3:6].reshape(-1, 3), g[..., 3:6].reshape(-1, 3)
        eul_err.append(np.linalg.norm(pe - ge, axis=-1))
        geo_err.append(geodesic_deg(euler_to_matrix_np(pe), euler_to_matrix_np(ge)))
        gt_mag.append(np.linalg.norm(ge, axis=-1))
        # anchor pose e_y in the policy frame, broadcast to every step of the chunk
        anchor_ey = proprio.float().cpu().numpy()[..., 4].reshape(-1)
        reps = pe.shape[0] // anchor_ey.shape[0]
        cos_ey.append(np.repeat(np.abs(np.cos(anchor_ey)), reps))

        if bi % 20 == 0:
            print(f"[diag] batch {bi+1}/{len(loader)}", flush=True)

    eul = np.concatenate(eul_err)
    geo = np.concatenate(geo_err)
    cey = np.concatenate(cos_ey)
    mag = np.concatenate(gt_mag)

    tag = args.tag or Path(args.checkpoint).parent.name[:40]
    print(f"\n===== {tag} =====")
    print(f"steps scored: {eul.size:,}")
    print(f"  euler_err    mean {eul.mean():.4f}  p90 {np.percentile(eul,90):.4f}  p99 {np.percentile(eul,99):.4f}")
    print(f"  geodesic_deg mean {geo.mean():.3f}  p90 {np.percentile(geo,90):.3f}  p99 {np.percentile(geo,99):.3f}")
    print(f"\n  {'|cos e_y| bin':>16} {'frac':>7} {'euler_err':>10} {'geo_deg':>9} {'geo/euler':>10}")
    edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cey >= lo) & (cey < hi)
        if m.sum() == 0:
            continue
        ratio = geo[m].mean() / max(eul[m].mean(), 1e-9)
        rows.append({"lo": lo, "hi": hi, "frac": float(m.mean()),
                     "euler": float(eul[m].mean()), "geo": float(geo[m].mean())})
        print(f"  [{lo:.2f},{hi:.2f})".rjust(16)
              + f" {m.mean()*100:6.1f}% {eul[m].mean():10.4f} {geo[m].mean():9.3f} {ratio:10.2f}")

    summary = {"tag": tag, "checkpoint": args.checkpoint, "n_steps": int(eul.size),
               "euler_mean": float(eul.mean()), "geo_mean": float(geo.mean()),
               "geo_p99": float(np.percentile(geo, 99)), "bins": rows,
               "gt_delta_mag_mean": float(mag.mean())}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\n[diag] wrote {args.output}")


if __name__ == "__main__":
    main()
