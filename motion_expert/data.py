from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

import torch


ImageHW = Union[Tuple[int, int], torch.Tensor]


@dataclass
class MotionExpertBatch:
    rgb: Optional[torch.Tensor] = None
    instruction_text: Optional[List[str]] = None
    instruction_id: Optional[torch.Tensor] = None
    text_token_ids: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None
    text_valid_lens: Optional[torch.Tensor] = None
    query_uv: Optional[torch.Tensor] = None
    query_world_xyz: Optional[torch.Tensor] = None
    traj_3d_xyz: Optional[torch.Tensor] = None
    future_delta_xyz: Optional[torch.Tensor] = None
    traj_mask: Optional[torch.Tensor] = None
    visual_tokens: Optional[torch.Tensor] = None
    text_tokens: Optional[torch.Tensor] = None
    image_hw: Optional[ImageHW] = None

    def to(self, device: torch.device) -> "MotionExpertBatch":
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                moved[field.name] = value.to(device)
            else:
                moved[field.name] = value
        return MotionExpertBatch(**moved)


def build_regular_query_uv(
    batch_size: int,
    image_hw: Tuple[int, int],
    grid_size: int = 20,
    normalized: bool = True,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Build a regular query grid in UV order [u, v]."""
    height, width = int(image_hw[0]), int(image_hw[1])

    if normalized:
        xs = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
        ys = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    else:
        xs = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * (
            width / float(grid_size)
        )
        ys = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * (
            height / float(grid_size)
        )

    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((uu, vv), dim=-1).reshape(1, grid_size * grid_size, 2)
    return grid.expand(batch_size, -1, -1).contiguous()


def flatten_query_grid(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 4:
        return tensor.flatten(1, 2)
    return tensor


def flatten_future_grid(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 5:
        return tensor.flatten(1, 2)
    return tensor
