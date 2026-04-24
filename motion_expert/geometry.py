from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_query_uv(
    query_uv: torch.Tensor,
    image_hw: Tuple[int, int],
    uv_is_normalized: bool,
) -> torch.Tensor:
    """Convert UV to raw pixel coordinates in the source image."""
    if uv_is_normalized:
        height, width = int(image_hw[0]), int(image_hw[1])
        scale = query_uv.new_tensor(
            [max(width - 1, 1), max(height - 1, 1)],
            dtype=query_uv.dtype,
        )
        return query_uv * scale
    return query_uv


def aspect_resize_and_pad_rgb_and_query_uv(
    rgb: torch.Tensor,
    query_uv: torch.Tensor,
    target_hw: Tuple[int, int],
    uv_is_normalized: bool = False,
    pad_value: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize while preserving aspect ratio, then center-pad to the target canvas.

    Returns the padded image and the transformed UV normalized to [0, 1] in the
    padded image coordinate frame.
    """
    if rgb.dim() != 3:
        raise ValueError(f"Expected rgb to have shape [3, H, W], got {tuple(rgb.shape)}.")

    src_h, src_w = int(rgb.shape[-2]), int(rgb.shape[-1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size {(target_h, target_w)}.")

    query_uv_px = normalize_query_uv(
        query_uv.float(),
        image_hw=(src_h, src_w),
        uv_is_normalized=uv_is_normalized,
    )

    scale = min(target_h / float(src_h), target_w / float(src_w))
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))

    resized_rgb = F.interpolate(
        rgb.unsqueeze(0).float(),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    pad_top = (target_h - resized_h) // 2
    pad_left = (target_w - resized_w) // 2

    padded_rgb = rgb.new_full((3, target_h, target_w), float(pad_value)).float()
    padded_rgb[:, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_rgb

    query_uv_resized = query_uv_px * scale
    query_uv_resized[..., 0] += pad_left
    query_uv_resized[..., 1] += pad_top

    denom = query_uv_resized.new_tensor(
        [max(target_w - 1, 1), max(target_h - 1, 1)],
        dtype=query_uv_resized.dtype,
    )
    query_uv_norm = torch.clamp(query_uv_resized / denom, 0.0, 1.0)
    return padded_rgb, query_uv_norm


def fill_invalid_traj_with_last_valid(
    traj_3d_xyz: torch.Tensor,
    traj_mask: torch.Tensor,
    anchor_xyz: torch.Tensor,
) -> torch.Tensor:
    """Repeat the last valid absolute waypoint; use the anchor for invalid prefixes."""
    filled = torch.nan_to_num(traj_3d_xyz.clone().float())
    traj_mask = traj_mask.bool()

    for point_idx in range(filled.shape[0]):
        last_valid = anchor_xyz[point_idx].float()
        for step_idx in range(filled.shape[1]):
            if traj_mask[point_idx, step_idx]:
                last_valid = filled[point_idx, step_idx]
            else:
                filled[point_idx, step_idx] = last_valid
    return filled


def compute_stepwise_future_delta(
    traj_3d_xyz: torch.Tensor,
    query_world_xyz: torch.Tensor,
) -> torch.Tensor:
    """Convert absolute future XYZ to stepwise deltas with the query point as anchor."""
    prev = torch.cat(
        [query_world_xyz.unsqueeze(1), traj_3d_xyz[:, :-1]],
        dim=1,
    )
    return traj_3d_xyz - prev
