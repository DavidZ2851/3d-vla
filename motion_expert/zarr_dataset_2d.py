"""2D MotionExpert dataset for calvin_2d_motion_zarr.

Each row: rgb (3,H,W uint8), query_uv (P,2) source pixels on a 20x20 grid,
traj_2d_uv (P,T,2) absolute source-pixel trajectories (``-inf`` where invalid),
valid_mask (P,T), instr_id (index into a position-indexed instruction list).

Produces model-grid ([-1,1]) query UV + per-point per-step UV deltas anchored at
the query point, with invalid steps zeroed and flagged in ``traj_mask``.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def _source_px_to_model_grid(uv_px: torch.Tensor, src_hw: Tuple[int, int],
                             target_hw: Tuple[int, int]) -> torch.Tensor:
    """Replicate aspect_resize_and_pad's source_pixels -> [-1,1] grid mapping."""
    scale, _, _, pad_top, pad_left = compute_resize_pad_params(src_hw, target_hw)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    uv = uv_px.float().clone()
    uv = uv * scale
    uv[..., 0] += pad_left
    uv[..., 1] += pad_top
    denom = uv.new_tensor([max(target_w - 1, 1), max(target_h - 1, 1)])
    return uv / denom * 2.0 - 1.0


class MotionExpert2DZarrDataset(Dataset):
    def __init__(
        self,
        root: str,
        instructions_path: str,
        image_size: Tuple[int, int] = (224, 224),
        future_steps: int = 15,
        image_pad_value: float = 0.5,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.root = str(root)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.future_steps = int(future_steps)
        self.image_pad_value = float(image_pad_value)
        self.instructions = InstructionTable(instructions_path)
        self._z = None  # opened lazily per worker

        z = zarr.open(self.root, mode="r")
        for k in ("rgb", "query_uv", "traj_2d_uv", "valid_mask", "instr_id"):
            if k not in z:
                raise KeyError(f"2D motion zarr missing required key `{k}` in {self.root}.")
        self._length = int(z["traj_2d_uv"].shape[0])
        if max_samples is not None:
            self._length = min(self._length, int(max_samples))
        t = int(z["traj_2d_uv"].shape[2])
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

        query_px = torch.from_numpy(np.asarray(z["query_uv"][idx], dtype=np.float32))      # (P,2)
        traj_px = torch.from_numpy(np.asarray(z["traj_2d_uv"][idx], dtype=np.float32))     # (P,T,2)
        valid = torch.from_numpy(np.asarray(z["valid_mask"][idx])).bool()                  # (P,T)

        # rgb -> 224, query px -> model grid [-1,1]
        rgb, query_grid = aspect_resize_and_pad_rgb_and_query_uv(
            rgb=rgb, query_uv=query_px, target_hw=self.image_size,
            query_uv_format="source_pixels", pad_value=self.image_pad_value,
        )

        # invalid trajectory positions -> anchor at the query point (delta becomes 0)
        traj_px = torch.where(valid.unsqueeze(-1), traj_px, query_px.unsqueeze(1))
        traj_px = torch.nan_to_num(traj_px, nan=0.0, posinf=0.0, neginf=0.0)
        traj_grid = _source_px_to_model_grid(traj_px, src_hw, self.image_size)             # (P,T,2)

        future_delta = traj_grid - query_grid.unsqueeze(1)                                  # (P,T,2)
        future_delta = torch.where(valid.unsqueeze(-1), future_delta, torch.zeros_like(future_delta))

        instruction_id = int(np.asarray(z["instr_id"][idx]).reshape(-1)[0])
        instruction_text = self.instructions.resolve(instruction_id)

        return {
            "rgb": rgb.float(),
            "instruction_text": instruction_text,
            "instruction_id": instruction_id,
            "query_uv": query_grid.float(),
            "future_delta_uv": future_delta.float(),
            "traj_mask": valid.bool(),
            "image_hw": torch.tensor(self.image_size, dtype=torch.long),
        }


class MotionExpert2DCollator:
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
            query_uv=torch.stack([s["query_uv"] for s in batch]),
            future_delta_xyz=torch.stack([s["future_delta_uv"] for s in batch]),  # reused field: UV delta
            traj_mask=torch.stack([s["traj_mask"] for s in batch]),
            image_hw=torch.stack([s["image_hw"] for s in batch]),
        )


@torch.no_grad()
def compute_delta_uv_scale(dataset: MotionExpert2DZarrDataset, max_samples: int = 2000) -> torch.Tensor:
    """Std of valid per-step UV deltas (model-grid units), for flow-target scaling."""
    n = min(len(dataset), int(max_samples))
    sums = torch.zeros(2, dtype=torch.float64)
    sumsq = torch.zeros(2, dtype=torch.float64)
    count = 0
    stride = max(1, len(dataset) // n)
    for i in range(0, len(dataset), stride):
        s = dataset[i]
        d = s["future_delta_uv"]                 # (P,T,2)
        m = s["traj_mask"].unsqueeze(-1)         # (P,T,1)
        vals = d[m.expand_as(d)].reshape(-1, 2).double()
        if vals.numel() == 0:
            continue
        sums += vals.sum(0)
        sumsq += (vals * vals).sum(0)
        count += vals.shape[0]
    if count == 0:
        return torch.ones(2)
    mean = sums / count
    var = (sumsq / count) - mean * mean
    return torch.sqrt(var.clamp_min(1e-8)).float()
