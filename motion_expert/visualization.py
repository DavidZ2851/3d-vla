from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def _flatten_query_xyz(query_world_xyz: torch.Tensor) -> torch.Tensor:
    if query_world_xyz.dim() == 4:
        return query_world_xyz.flatten(1, 2)
    if query_world_xyz.dim() == 3:
        return query_world_xyz
    raise ValueError(
        f"query_world_xyz must be [B,N,3] or [B,G,G,3], got {tuple(query_world_xyz.shape)}"
    )


def _flatten_future_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 5:
        return tensor.flatten(1, 2)
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 3:
        return tensor
    raise ValueError(
        f"future tensor must be [B,N,T,*] or [B,G,G,T,*], got {tuple(tensor.shape)}"
    )


def _set_axes_equal_3d(ax, points_xyz: np.ndarray) -> None:
    if points_xyz.size == 0:
        return
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if not np.isfinite(radius) or radius <= 1e-8:
        radius = 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_motion_expert_spaghetti_batch(
    *,
    query_world_xyz: torch.Tensor,
    gt_delta_xyz: torch.Tensor,
    pred_delta_xyz: torch.Tensor,
    traj_mask: torch.Tensor,
    output_dir: str | Path,
    prefix: str,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    motion_threshold: float = 0.005,
    max_points_per_sample: int = 200,
    instruction_text: Optional[List[str]] = None,
    rgb: Optional[torch.Tensor] = None,
    rgb_output_dir: Optional[str | Path] = None,
) -> List[Path]:
    # Local import so training works even if matplotlib is not installed.
    import matplotlib.pyplot as plt

    query_world_xyz = _flatten_query_xyz(query_world_xyz).float()
    gt_delta_xyz = _flatten_future_tensor(gt_delta_xyz).float()
    pred_delta_xyz = _flatten_future_tensor(pred_delta_xyz).float()
    traj_mask = _flatten_future_tensor(traj_mask).bool()

    if gt_delta_xyz.shape != pred_delta_xyz.shape:
        raise ValueError(
            f"GT/PRED delta shape mismatch: {tuple(gt_delta_xyz.shape)} vs {tuple(pred_delta_xyz.shape)}"
        )
    if gt_delta_xyz.shape[:3] != traj_mask.shape:
        raise ValueError(
            f"traj_mask shape {tuple(traj_mask.shape)} must match delta prefix {tuple(gt_delta_xyz.shape[:3])}"
        )
    if query_world_xyz.shape[:2] != gt_delta_xyz.shape[:2]:
        raise ValueError(
            f"query_world_xyz shape {tuple(query_world_xyz.shape)} incompatible with deltas {tuple(gt_delta_xyz.shape)}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = None
    if rgb_output_dir is not None:
        rgb_dir = Path(rgb_output_dir)
        rgb_dir.mkdir(parents=True, exist_ok=True)

    # Absolute trajectories from anchor + cumulative stepwise deltas.
    anchor = query_world_xyz
    gt_abs = anchor.unsqueeze(2) + torch.cumsum(gt_delta_xyz, dim=2)
    pred_abs = anchor.unsqueeze(2) + torch.cumsum(pred_delta_xyz, dim=2)

    gt_disp = torch.linalg.norm(torch.cumsum(gt_delta_xyz, dim=2), dim=-1)
    gt_disp = torch.where(traj_mask, gt_disp, torch.zeros_like(gt_disp))
    pred_disp = torch.linalg.norm(torch.cumsum(pred_delta_xyz, dim=2), dim=-1)
    pred_disp = torch.where(traj_mask, pred_disp, torch.zeros_like(pred_disp))

    # Filter each trajectory source independently:
    # - GT trajectories by GT motion
    # - PRED trajectories by predicted motion
    valid_points = traj_mask.any(dim=2)
    gt_moved_mask = (gt_disp.max(dim=2).values > float(motion_threshold)) & valid_points
    pred_moved_mask = (pred_disp.max(dim=2).values > float(motion_threshold)) & valid_points

    batch_size = int(anchor.shape[0])
    to_render = batch_size if max_samples is None else min(batch_size, int(max_samples))
    saved_paths: List[Path] = []

    for batch_idx in range(to_render):
        gt_moved_idx = torch.nonzero(gt_moved_mask[batch_idx], as_tuple=False).squeeze(-1)
        pred_moved_idx = torch.nonzero(pred_moved_mask[batch_idx], as_tuple=False).squeeze(-1)
        moved_idx = torch.unique(torch.cat([gt_moved_idx, pred_moved_idx], dim=0))
        if moved_idx.numel() == 0:
            continue

        if moved_idx.numel() > int(max_points_per_sample):
            point_motion = torch.maximum(
                gt_disp[batch_idx].max(dim=1).values,
                pred_disp[batch_idx].max(dim=1).values,
            )
            topk = torch.topk(point_motion[moved_idx], k=int(max_points_per_sample), largest=True).indices
            moved_idx = moved_idx[topk]

        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        all_pts = []
        for point_idx in moved_idx.tolist():
            valid_steps = traj_mask[batch_idx, point_idx]
            if not bool(valid_steps.any()):
                continue
            valid_count = int(valid_steps.sum().item())
            anchor_xyz = anchor[batch_idx, point_idx].detach().cpu().numpy()
            gt_traj = gt_abs[batch_idx, point_idx, :valid_count].detach().cpu().numpy()
            pred_traj = pred_abs[batch_idx, point_idx, :valid_count].detach().cpu().numpy()

            gt_poly = np.concatenate([anchor_xyz[None, :], gt_traj], axis=0)
            pred_poly = np.concatenate([anchor_xyz[None, :], pred_traj], axis=0)
            all_pts.append(gt_poly)
            all_pts.append(pred_poly)

            # GT pink, PRED green; each polyline is rendered only if its own source moved.
            if bool(gt_moved_mask[batch_idx, point_idx]):
                ax.plot(
                    gt_poly[:, 0],
                    gt_poly[:, 1],
                    gt_poly[:, 2],
                    color="#ff69b4",
                    alpha=0.75,
                    linewidth=1.2,
                )
                all_pts.append(gt_poly)
            if bool(pred_moved_mask[batch_idx, point_idx]):
                ax.plot(
                    pred_poly[:, 0],
                    pred_poly[:, 1],
                    pred_poly[:, 2],
                    color="#00cc66",
                    alpha=0.75,
                    linewidth=1.2,
                )
                all_pts.append(pred_poly)

        if all_pts:
            cloud = np.concatenate(all_pts, axis=0)
            _set_axes_equal_3d(ax, cloud)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            (
                f"{prefix} sample={start_index + batch_idx} "
                f"moving_gt={int(gt_moved_idx.numel())} "
                f"moving_pred={int(pred_moved_idx.numel())}\n"
                f"task: {instruction_text[batch_idx]}"
            )
            if instruction_text is not None and batch_idx < len(instruction_text)
            else (
                f"{prefix} sample={start_index + batch_idx} "
                f"moving_gt={int(gt_moved_idx.numel())} "
                f"moving_pred={int(pred_moved_idx.numel())}"
            )
        )

        path = out_dir / f"{prefix}_sample_{start_index + batch_idx:05d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved_paths.append(path)

        if rgb is not None and batch_idx < int(rgb.shape[0]) and rgb_dir is not None:
            rgb_img = rgb[batch_idx].detach().cpu().float()
            if rgb_img.dim() == 3 and rgb_img.shape[0] in (1, 3):
                rgb_img = rgb_img.permute(1, 2, 0).contiguous()
            rgb_img = rgb_img.clamp(0.0, 1.0).numpy()
            rgb_path = rgb_dir / f"{prefix}_sample_{start_index + batch_idx:05d}_rgb.png"
            plt.imsave(rgb_path, rgb_img)

    return saved_paths
