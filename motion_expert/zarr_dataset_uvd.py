"""UVD MotionExpert dataset for calvin_motion_uvd_zarr.

Each row::

    rgb          (3, H, W) uint8        query-frame image
    query_uv     (P, 3) f32            query points: u(px), v(px), depth(metres)
    traj_3d_uvd  (P, T, 3) f32         absolute future u(px), v(px), depth(metres)
    valid_mask   (P, T) bool           per-point per-step validity
    instr_id     ()  int32             index into the instruction side-table

P = grid_size**2 (20x20 = 400), T = future_steps (15).

Normalization (see ``compute_calvin_uvd_norm_stats.py`` for the rationale):
  * query ``u, v``  -> [-1, 1] model grid (same as the 2D track).
  * query ``depth`` -> GLOBAL standardize ``(d - mean) / std`` (no centroid
    centering; CALVIN's static camera keeps absolute depth informative/bounded).
  * target = anchor-relative future delta ``traj_uvd - query_uvd``: ``du, dv`` in
    model-grid units, ``dd`` in metres. Invalid steps are anchored at the query
    point (delta 0) and flagged in ``traj_mask``. The per-axis std
    (``traj_delta_uvd_scale``) is applied inside the model.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .data import MotionExpertBatch
from .geometry import (
    aspect_resize_and_pad_rgb_and_query_uv,
    compute_resize_pad_params,
)
from .zarr_dataset import InstructionTable, MotionExpertQwenTokenizer

# Defaults from compute_calvin_uvd_norm_stats.py on the calvin_motion_uvd_zarr
# training split (target_hw = 224x224). Override via the dataset constructor /
# config when the data changes.
DEFAULT_QUERY_DEPTH_MEAN = 4.290678619501367
DEFAULT_QUERY_DEPTH_STD = 0.5424240862196633


def _source_px_to_model_grid(
    uv_px: torch.Tensor,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Replicate aspect_resize_and_pad's source_pixels -> [-1, 1] grid mapping."""
    scale, _, _, pad_top, pad_left = compute_resize_pad_params(src_hw, target_hw)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    uv = uv_px.float().clone()
    uv = uv * scale
    uv[..., 0] += pad_left
    uv[..., 1] += pad_top
    denom = uv.new_tensor([max(target_w - 1, 1), max(target_h - 1, 1)])
    return uv / denom * 2.0 - 1.0


class MotionExpertUVDZarrDataset(Dataset):
    def __init__(
        self,
        root: str,
        instructions_path: str,
        image_size: Tuple[int, int] = (224, 224),
        future_steps: int = 15,
        image_pad_value: float = 0.5,
        query_depth_mean: float = DEFAULT_QUERY_DEPTH_MEAN,
        query_depth_std: float = DEFAULT_QUERY_DEPTH_STD,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root = str(root)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.future_steps = int(future_steps)
        self.image_pad_value = float(image_pad_value)
        self.query_depth_mean = float(query_depth_mean)
        self.query_depth_std = max(float(query_depth_std), 1e-6)
        self.instructions = InstructionTable(instructions_path)
        self._z = None  # opened lazily per worker

        z = zarr.open(self.root, mode="r")
        for k in ("rgb", "query_uv", "traj_3d_uvd", "valid_mask", "instr_id"):
            if k not in z:
                raise KeyError(f"uvd motion zarr missing required key `{k}` in {self.root}.")
        if int(z["query_uv"].shape[-1]) != 3:
            raise ValueError("query_uv must have 3 channels (u, v, depth).")
        if int(z["traj_3d_uvd"].shape[-1]) != 3:
            raise ValueError("traj_3d_uvd must have 3 channels (u, v, depth).")
        self._length = int(z["traj_3d_uvd"].shape[0])
        if max_samples is not None:
            self._length = min(self._length, int(max_samples))
        t = int(z["traj_3d_uvd"].shape[2])
        if t != self.future_steps:
            raise ValueError(f"zarr future_steps={t} != configured {self.future_steps}.")

    def __len__(self) -> int:
        return self._length

    def _zarr(self):
        if self._z is None:
            self._z = zarr.open(self.root, mode="r")
        return self._z

    def __getitem__(self, idx: int) -> Dict[str, object]:
        z = self._zarr()
        idx = int(idx) % self._length

        rgb = torch.from_numpy(np.asarray(z["rgb"][idx], dtype=np.float32))  # (3,H,W) 0..255
        rgb = rgb / 255.0
        src_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))

        query = torch.from_numpy(np.asarray(z["query_uv"][idx], dtype=np.float32))     # (P,3)
        traj = torch.from_numpy(np.asarray(z["traj_3d_uvd"][idx], dtype=np.float32))   # (P,T,3)
        valid = torch.from_numpy(np.asarray(z["valid_mask"][idx])).bool()              # (P,T)

        query_px = query[:, :2]                                  # (P,2) u,v pixels
        query_depth = query[:, 2]                                # (P,)  metres

        # rgb -> 224 canvas; query px -> model grid [-1,1]
        rgb, query_grid = aspect_resize_and_pad_rgb_and_query_uv(
            rgb=rgb, query_uv=query_px, target_hw=self.image_size,
            query_uv_format="source_pixels", pad_value=self.image_pad_value,
        )

        # invalid future positions -> anchor at the query point (delta becomes 0)
        traj_px = torch.where(valid.unsqueeze(-1), traj[..., :2], query_px.unsqueeze(1))
        traj_px = torch.nan_to_num(traj_px, nan=0.0, posinf=0.0, neginf=0.0)
        traj_depth = torch.where(valid, traj[..., 2], query_depth.unsqueeze(1))
        traj_depth = torch.nan_to_num(traj_depth, nan=0.0, posinf=0.0, neginf=0.0)

        traj_grid = _source_px_to_model_grid(traj_px, src_hw, self.image_size)          # (P,T,2)

        delta_uv = traj_grid - query_grid.unsqueeze(1)                                  # (P,T,2) grid units
        delta_d = (traj_depth - query_depth.unsqueeze(1)).unsqueeze(-1)                 # (P,T,1) metres
        future_delta = torch.cat([delta_uv, delta_d], dim=-1)                           # (P,T,3)
        future_delta = torch.where(valid.unsqueeze(-1), future_delta, torch.zeros_like(future_delta))

        # query geometry coordinate: [u_grid, v_grid, depth_standardized]
        query_depth_norm = (query_depth - self.query_depth_mean) / self.query_depth_std
        query_uvd = torch.cat([query_grid, query_depth_norm.unsqueeze(-1)], dim=-1)     # (P,3)

        instruction_id = int(np.asarray(z["instr_id"][idx]).reshape(-1)[0])
        instruction_text = self.instructions.resolve(instruction_id)

        return {
            "rgb": rgb.float(),
            "instruction_text": instruction_text,
            "instruction_id": instruction_id,
            "query_uv": query_grid.float(),       # (P,2) for visual grid_sample
            "query_uvd": query_uvd.float(),        # (P,3) geometry coordinate
            "future_delta_uvd": future_delta.float(),
            "traj_mask": valid.bool(),
            "image_hw": torch.tensor(self.image_size, dtype=torch.long),
        }


class MotionExpertUVDCollator:
    def __init__(self, tokenizer: MotionExpertQwenTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, object]]) -> MotionExpertBatch:
        instructions = [s["instruction_text"] for s in batch]
        tok = self.tokenizer(instructions)
        return MotionExpertBatch(
            rgb=torch.stack([s["rgb"] for s in batch]),
            instruction_text=instructions,
            instruction_id=torch.tensor([s["instruction_id"] for s in batch], dtype=torch.long),
            text_token_ids=tok.input_ids,
            text_attention_mask=tok.attention_mask,
            text_valid_lens=tok.attention_mask.sum(dim=1),
            query_uv=torch.stack([s["query_uv"] for s in batch]),            # (B,P,2)
            query_world_xyz=torch.stack([s["query_uvd"] for s in batch]),    # (B,P,3) carries uvd
            future_delta_xyz=torch.stack([s["future_delta_uvd"] for s in batch]),  # (B,P,T,3)
            traj_mask=torch.stack([s["traj_mask"] for s in batch]),
            image_hw=torch.stack([s["image_hw"] for s in batch]),
        )


@torch.no_grad()
def compute_delta_uvd_scale(dataset: MotionExpertUVDZarrDataset, max_samples: int = 2000) -> torch.Tensor:
    """Per-axis std of valid anchor-relative UVD deltas (du/dv grid, dd metres)."""
    n = min(len(dataset), int(max_samples))
    sums = torch.zeros(3, dtype=torch.float64)
    sumsq = torch.zeros(3, dtype=torch.float64)
    count = 0
    stride = max(1, len(dataset) // max(1, n))
    for i in range(0, len(dataset), stride):
        s = dataset[i]
        d = s["future_delta_uvd"]                # (P,T,3)
        m = s["traj_mask"].unsqueeze(-1)         # (P,T,1)
        vals = d[m.expand_as(d)].reshape(-1, 3).double()
        if vals.numel() == 0:
            continue
        sums += vals.sum(0)
        sumsq += (vals * vals).sum(0)
        count += vals.shape[0]
    if count == 0:
        return torch.ones(3)
    mean = sums / count
    var = (sumsq / count) - mean * mean
    return torch.sqrt(var.clamp_min(1e-8)).float()
