from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def center_query_world_xyz(query_world_xyz: torch.Tensor) -> torch.Tensor:
    """Center query XYZ by the per-sample finite-point mean.

    Supports both [N, 3] and [B, N, 3]. Non-finite points do not contribute to
    the center and are zeroed after centering.
    """
    if query_world_xyz is None:
        return query_world_xyz
    if query_world_xyz.shape[-1] != 3:
        raise ValueError(f"Expected query_world_xyz last dim to be 3, got {tuple(query_world_xyz.shape)}.")

    xyz = query_world_xyz.float()
    finite = torch.isfinite(xyz).all(dim=-1, keepdim=True)
    safe_xyz = torch.where(finite, xyz, torch.zeros_like(xyz))
    counts = finite.to(xyz.dtype).sum(dim=-2, keepdim=True).clamp_min(1.0)
    center = safe_xyz.sum(dim=-2, keepdim=True) / counts
    centered = torch.where(finite, xyz - center, torch.zeros_like(xyz))
    return torch.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)


def compute_resize_pad_params(
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> Tuple[float, int, int, int, int]:
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if src_h <= 0 or src_w <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid resize/pad sizes: src={src_hw}, target={target_hw}.")
    scale = min(target_h / float(src_h), target_w / float(src_w))
    resized_h = max(1, int(round(src_h * scale)))
    resized_w = max(1, int(round(src_w * scale)))
    pad_top = (target_h - resized_h) // 2
    pad_left = (target_w - resized_w) // 2
    return scale, resized_h, resized_w, pad_top, pad_left


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

    Returns the padded image and the transformed UV normalized to [-1, 1] in the
    padded image coordinate frame, matching torch.nn.functional.grid_sample.
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

    scale, resized_h, resized_w, pad_top, pad_left = compute_resize_pad_params(
        (src_h, src_w),
        (target_h, target_w),
    )

    resized_rgb = F.interpolate(
        rgb.unsqueeze(0).float(),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    padded_rgb = rgb.new_full((3, target_h, target_w), float(pad_value)).float()
    padded_rgb[:, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_rgb

    query_uv_resized = query_uv_px * scale
    query_uv_resized[..., 0] += pad_left
    query_uv_resized[..., 1] += pad_top

    denom = query_uv_resized.new_tensor(
        [max(target_w - 1, 1), max(target_h - 1, 1)],
        dtype=query_uv_resized.dtype,
    )
    query_uv_01 = torch.clamp(query_uv_resized / denom, 0.0, 1.0)
    query_uv_grid = query_uv_01.mul(2.0).sub(1.0)
    return padded_rgb, query_uv_grid


def aspect_resize_and_pad_depth(
    depth: torch.Tensor,
    target_hw: Tuple[int, int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    if depth.dim() == 2:
        depth_chw = depth.unsqueeze(0)
    elif depth.dim() == 3 and depth.shape[0] == 1:
        depth_chw = depth
    else:
        raise ValueError(f"Expected depth shape [H, W] or [1, H, W], got {tuple(depth.shape)}.")

    src_h, src_w = int(depth_chw.shape[-2]), int(depth_chw.shape[-1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    scale, resized_h, resized_w, pad_top, pad_left = compute_resize_pad_params(
        (src_h, src_w),
        (target_h, target_w),
    )
    del scale

    resized_depth = F.interpolate(
        depth_chw.unsqueeze(0).float(),
        size=(resized_h, resized_w),
        mode="nearest",
    ).squeeze(0)
    padded_depth = depth_chw.new_full(
        (1, target_h, target_w),
        float(pad_value),
    ).float()
    padded_depth[:, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_depth
    return padded_depth


def transform_intrinsics_for_resize_pad(
    intrinsics: torch.Tensor,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    scale, _, _, pad_top, pad_left = compute_resize_pad_params(src_hw, target_hw)
    out = intrinsics.clone().float()
    out[..., 0, 0] *= scale
    out[..., 1, 1] *= scale
    out[..., 0, 2] = out[..., 0, 2] * scale + pad_left
    out[..., 1, 2] = out[..., 1, 2] * scale + pad_top
    return out


def sample_depth_at_query_uv(
    depth: torch.Tensor,
    query_uv: torch.Tensor,
) -> torch.Tensor:
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected depth shape [B, H, W] or [B, 1, H, W], got {tuple(depth.shape)}.")
    grid = query_uv.float().unsqueeze(2)
    sampled = F.grid_sample(
        depth.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(1).squeeze(-1)
    finite_positive = torch.isfinite(sampled) & (sampled > 0)
    return torch.where(finite_positive, sampled, torch.zeros_like(sampled))


def query_uv_depth_to_world_xyz(
    query_uv: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depth.dim() == 3:
        _, height, width = depth.shape
    elif depth.dim() == 4:
        _, _, height, width = depth.shape
    else:
        raise ValueError(f"Expected depth shape [B, H, W] or [B, 1, H, W], got {tuple(depth.shape)}.")

    batch_size = query_uv.shape[0]
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0).expand(batch_size, -1, -1)

    query_depth = sample_depth_at_query_uv(depth, query_uv)
    query_uv_01 = query_uv.float().add(1.0).mul(0.5)
    uv_px = query_uv_01 * query_uv.new_tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=query_uv.dtype,
    )

    fx = intrinsics[:, 0, 0].clamp_min(1e-6)[:, None]
    fy = intrinsics[:, 1, 1].clamp_min(1e-6)[:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]
    z = query_depth
    x = (uv_px[..., 0] - cx) * z / fx
    y = (uv_px[..., 1] - cy) * z / fy
    cam_xyz = torch.stack([x, y, z], dim=-1)

    ones = torch.ones_like(cam_xyz[..., :1])
    cam_xyz_h = torch.cat([cam_xyz, ones], dim=-1)
    world_xyz_h = torch.matmul(cam_xyz_h, extrinsics.float().transpose(-1, -2))
    world_xyz = world_xyz_h[..., :3]
    world_xyz = torch.nan_to_num(world_xyz, nan=0.0, posinf=0.0, neginf=0.0)
    return world_xyz, query_depth.unsqueeze(-1)


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


def fill_invalid_delta_with_zero(
    traj_delta_xyz: torch.Tensor,
    traj_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero invalid deltas so masked steps stay numerically stable."""
    filled = torch.nan_to_num(
        traj_delta_xyz.clone().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    filled[~traj_mask.bool()] = 0.0
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
