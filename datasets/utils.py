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


def convert_quaternion_format(data, from_format: str = 'wxyz', to_format: str = 'xyzw', *, inplace: bool = True):
    """
    Convert quaternion format between wxyz and xyzw.
    Args:
        data: Tensor with shape (..., 8) where indices 3:7 contain quaternion
              or shape (..., 7) where indices 3:7 contain quaternion
    """
    if from_format == to_format:
        return data
    if not inplace:
        data = data.clone()
    quat = data[..., 3:7].clone()
    if from_format == 'wxyz' and to_format == 'xyzw':
        data[..., 3:7] = quat[..., [1, 2, 3, 0]]  # [x, y, z, w]
    elif from_format == 'xyzw' and to_format == 'wxyz':
        data[..., 3:7] = quat[..., [3, 0, 1, 2]]  # [w, x, y, z]
    else:
        raise ValueError(f"Invalid conversion: {from_format} -> {to_format}")
    return data


def to_relative_action(actions, anchor_action, qform='xyzw'):
    """
    Compute delta actions where the first delta is relative to anchor,
    and subsequent deltas are relative to the previous timestep.

    Args:
        actions: (B, nhands, T, 8)  — future trajectory
        anchor_action: (B, nhands, 1, 8) — current pose to treat as timestep -1
        qform: 'xyzw' or 'wxyz' — quaternion format

    Returns:
        delta_actions: (B, nhands, N, 8)
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

    return torch.cat([rel_pos, rel_orn, gripper], -1)  # (B, nhands, T, 8)
