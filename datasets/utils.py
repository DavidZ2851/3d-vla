import numpy as np
import torch
import zarr
from zarr.storage import DirectoryStore
from zarr import LRUStoreCache

import utils.pytorch3d_transforms as pytorch3d_transforms


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.as_tensor(x)


def read_zarr_with_cache(fname, mem_gb=16):
    # Configure the underlying store
    store = DirectoryStore(fname)

    # Wrap the store with a cache
    cached_store = LRUStoreCache(store, max_size=mem_gb * 2**30)  # GB cache

    # Open Zarr file with caching
    return zarr.open_group(cached_store, mode="r")


def to_relative_action(actions, anchor_action, qform='xyzw'):
    """
    Compute delta actions where the first delta is relative to anchor,
    and subsequent deltas are relative to the previous timestep.

    Args:
        actions: (..., N, 8)  — future trajectory
        anchor_action: (..., 1, 8) — current pose to treat as timestep -1
        qform: 'xyzw' or 'wxyz' — quaternion format

    Returns:
        delta_actions: (..., N, 8)
    """
    assert actions.shape[-1] == 8
    # Stitch anchor in front and shift everything by one
    prev = torch.cat([anchor_action, actions[..., :-1, :]], -2)  # (..., N, 8)

    rel_pos = actions[..., :3] - prev[..., :3]

    if qform == 'xyzw':
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., [6, 3, 4, 5]],
            pytorch3d_transforms.quaternion_invert(prev[..., [6, 3, 4, 5]])
        )[..., [1, 2, 3, 0]]
    elif qform == 'wxyz':
        rel_orn = pytorch3d_transforms.quaternion_multiply(
            actions[..., 3:7],
            pytorch3d_transforms.quaternion_invert(prev[..., 3:7])
        )
    else:
        raise ValueError("Invalid quaternion format")

    gripper = actions[..., -1:]

    return torch.cat([rel_pos, rel_orn, gripper], -1)  # (..., N, 8)


def matrix_to_euler_xyz(matrix: torch.Tensor):
    """Convert rotation matrices to XYZ Euler angles in radians."""
    m00 = matrix[..., 0, 0]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    sy = torch.sqrt(m00 * m00 + m10 * m10)
    singular = sy < 1e-6

    x = torch.atan2(m21, m22)
    y = torch.atan2(-m20, sy)
    z = torch.atan2(m10, m00)

    x_s = torch.atan2(-m12, m11)
    y_s = torch.atan2(-m20, sy)
    z_s = torch.zeros_like(z)

    x = torch.where(singular, x_s, x)
    y = torch.where(singular, y_s, y)
    z = torch.where(singular, z_s, z)
    return torch.stack([x, y, z], dim=-1)


def quat_wxyz_to_euler_xyz(quat: torch.Tensor):
    """Convert wxyz quaternions to canonical XYZ Euler angles in radians."""
    quat = quat / quat.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-10)
    matrix = pytorch3d_transforms.quaternion_to_matrix(quat)
    return matrix_to_euler_xyz(matrix)
