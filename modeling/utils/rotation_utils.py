"""
Vectorized rotation utilities for efficient batched operations.
All operations work directly on rotation matrices or 6D representations.
"""
import torch

from .utils import (
    quaternion_to_matrix,
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
)

try:
    from torch._higher_order_ops.associative_scan import associative_scan  # type: ignore
except Exception:  # pragma: no cover
    associative_scan = None

_ASSOCIATIVE_SCAN_COMBINE_MODE = "generic"


def batched_cumulative_rotation_matrices(
    delta_rot_6d: torch.Tensor,
    start_rot_6d: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cumulative rotations using batched matrix multiplication.
    Fully vectorized on CUDA when associative_scan is available.
    """
    B, T = delta_rot_6d.shape[:2]
    extra_dims = delta_rot_6d.shape[2:-1]

    n_extra = 1 if len(extra_dims) == 0 else int(torch.prod(torch.tensor(list(extra_dims), device=delta_rot_6d.device)))

    delta_flat = delta_rot_6d.reshape(B * T * n_extra, 6)
    start_flat = start_rot_6d.reshape(B * n_extra, 6)

    delta_mats = compute_rotation_matrix_from_ortho6d(delta_flat).reshape(B, T, n_extra, 3, 3)
    start_mat = compute_rotation_matrix_from_ortho6d(start_flat).reshape(B, 1, n_extra, 3, 3)

    if associative_scan is not None and delta_mats.is_cuda and start_mat.is_cuda:
        delta_scan = delta_mats.permute(1, 0, 2, 3, 4).reshape(T, B * n_extra, 3, 3)

        def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

        prefix = associative_scan(
            mm,
            delta_scan,
            dim=0,
            combine_mode=_ASSOCIATIVE_SCAN_COMBINE_MODE,
        )
        prefix = prefix.reshape(T, B, n_extra, 3, 3).permute(1, 0, 2, 3, 4)
        cumulative_mats = torch.matmul(start_mat, prefix)
    else:
        cumulative_mats_list = []
        current = start_mat
        for t in range(T):
            delta_t = delta_mats[:, t:t + 1]
            current = torch.matmul(current, delta_t)
            cumulative_mats_list.append(current)
        cumulative_mats = torch.cat(cumulative_mats_list, dim=1)

    if len(extra_dims) > 0:
        cumulative_mats = cumulative_mats.reshape(B, T, *extra_dims, 3, 3)
    else:
        cumulative_mats = cumulative_mats.reshape(B, T, 3, 3)

    cumulative_flat = cumulative_mats.reshape(B * T * n_extra, 3, 3)
    cumulative_6d_flat = get_ortho6d_from_rotation_matrix(cumulative_flat)
    if len(extra_dims) > 0:
        return cumulative_6d_flat.reshape(B, T, *extra_dims, 6)
    return cumulative_6d_flat.reshape(B, T, 6)


def relative_to_absolute_6d(action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """
    Convert relative actions to absolute using 6D rotation representation.

    Handles:
      - quaternion-relative action (8 dims):  [pos_delta(3), quat_delta_wxyz(4), gripper(1)]
      - rot6d-relative action (10 dims):      [pos_delta(3), rot6d_delta(6), gripper(1)]
    """
    if len(proprio.shape) == 3:
        start_pos = proprio[:, -1:, :3]
        start_quat = proprio[:, -1:, 3:7]  # wxyz
        action_pos_delta = action[..., :3]
        action_gripper = action[..., -1:]
        if action.shape[-1] == 10:
            action_rot_delta = action[..., 3:9]
        elif action.shape[-1] == 8:
            action_quat_delta = normalise_quat(action[..., 3:7])
            B, T = action_quat_delta.shape[:2]
            mats = quaternion_to_matrix(action_quat_delta.reshape(B * T, 4))
            action_rot_delta = get_ortho6d_from_rotation_matrix(mats).reshape(B, T, 6)
        else:
            raise ValueError(f"Unexpected action dim {action.shape[-1]} for single-hand relative_to_absolute_6d")
    elif len(proprio.shape) == 4:
        start_pos = proprio[:, -1:, :, :3]
        start_quat = proprio[:, -1:, :, 3:7]  # wxyz
        action_pos_delta = action[..., :3]
        action_gripper = action[..., -1:]
        if action.shape[-1] == 10:
            action_rot_delta = action[..., 3:9]
        elif action.shape[-1] == 8:
            action_quat_delta = normalise_quat(action[..., 3:7])
            B, T, nhand = action_quat_delta.shape[:3]
            mats = quaternion_to_matrix(action_quat_delta.reshape(B * T * nhand, 4))
            rot6d = get_ortho6d_from_rotation_matrix(mats).reshape(B, T, nhand, 6)
            action_rot_delta = rot6d
        else:
            raise ValueError(f"Unexpected action dim {action.shape[-1]} for bimanual relative_to_absolute_6d")
    else:
        raise ValueError(f"Unexpected proprioception shape: {proprio.shape}")

    start_quat = normalise_quat(start_quat)
    B = start_quat.shape[0]
    if len(start_quat.shape) == 3:
        mats = quaternion_to_matrix(start_quat.reshape(B, 4))
        start_rot_6d = get_ortho6d_from_rotation_matrix(mats).reshape(B, 1, 6)
    else:
        nhand = start_quat.shape[2]
        mats = quaternion_to_matrix(start_quat.reshape(B * nhand, 4))
        start_rot_6d = get_ortho6d_from_rotation_matrix(mats).reshape(B, 1, nhand, 6)

    cumulative_rot_6d = batched_cumulative_rotation_matrices(action_rot_delta, start_rot_6d)
    pos = start_pos + action_pos_delta.cumsum(dim=1)
    return torch.cat([pos, cumulative_rot_6d, action_gripper], dim=-1)


def rotation_geodesic_distance(rot_mat1: torch.Tensor, rot_mat2: torch.Tensor) -> torch.Tensor:
    """
    Fast surrogate for geodesic distance between two rotation matrices.
    Returns 1 - cos(theta), monotonic on [0, pi].
    """
    rot_rel = torch.matmul(rot_mat1.transpose(-2, -1), rot_mat2)
    trace = rot_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return 1.0 - cos_angle

