import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from datasets.utils import read_zarr_with_cache, to_tensor

from .data import MotionExpertBatch, flatten_future_grid, flatten_query_grid
from .geometry import (
    aspect_resize_and_pad_rgb_and_query_uv,
    compute_stepwise_future_delta,
    fill_invalid_traj_with_last_valid,
)


@dataclass
class TokenizedInstructionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class InstructionTable:
    def __init__(self, instruction_source: str):
        with open(instruction_source, "r") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            self._instructions = {int(idx): str(text) for idx, text in enumerate(raw)}
        elif isinstance(raw, dict):
            mapped = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    value = value.get("text", value.get("instruction", value))
                mapped[int(key)] = str(value)
            self._instructions = mapped
        else:
            raise ValueError(
                f"Unsupported instruction table type: {type(raw).__name__}."
            )

    def resolve(self, instruction_id: int) -> str:
        if instruction_id not in self._instructions:
            raise KeyError(f"Instruction id {instruction_id} not found in side table.")
        return self._instructions[instruction_id]


class MotionExpertQwenTokenizer:
    def __init__(
        self,
        backbone_type: str = "qwen",
        model_name_or_path: Optional[str] = None,
        max_length: int = 128,
    ):
        self.backbone_type = str(backbone_type).lower()
        if model_name_or_path is None:
            if self.backbone_type == "qwen":
                model_name_or_path = os.environ.get("QWEN3VL_MODEL_PATH")
            elif self.backbone_type == "florence2":
                model_name_or_path = os.environ.get(
                    "FLORENCE2_MODEL_PATH",
                    "microsoft/Florence-2-large",
                )
        if model_name_or_path is None:
            raise ValueError(
                "No tokenizer path provided. Set the matching backbone env var or "
                "pass model_name_or_path explicitly."
            )
        self.max_length = int(max_length)
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, instructions: Sequence[str]) -> TokenizedInstructionBatch:
        out = self.tokenizer(
            list(instructions),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        return TokenizedInstructionBatch(
            input_ids=out["input_ids"],
            attention_mask=out["attention_mask"],
        )


MotionExpertTextTokenizer = MotionExpertQwenTokenizer


class MotionExpertZarrDataset(Dataset):
    def __init__(
        self,
        root: str,
        instructions_path: str,
        image_size: Tuple[int, int] = (384, 640),
        instruction_id_key: Optional[str] = None,
        query_uv_is_normalized: bool = False,
        image_pad_value: float = 0.5,
        mem_limit_gb: float = 16.0,
    ):
        super().__init__()
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit_gb)
        self.instructions = InstructionTable(instructions_path)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.query_uv_is_normalized = bool(query_uv_is_normalized)
        self.image_pad_value = float(image_pad_value)
        self.instruction_id_key = self._resolve_instruction_key(instruction_id_key)

    def _resolve_instruction_key(self, requested_key: Optional[str]) -> str:
        if requested_key is not None:
            if requested_key not in self.annos:
                raise KeyError(
                    f"Instruction id key `{requested_key}` not found in zarr."
                )
            return requested_key
        for candidate in ("instr_id", "instruction_id", "task_id"):
            if candidate in self.annos:
                return candidate
        raise KeyError(
            "Could not find an instruction id key. Tried `instr_id`, "
            "`instruction_id`, and `task_id`."
        )

    def __len__(self) -> int:
        return len(self.annos["rgb"])

    def _load_tensor(self, key: str, idx: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        value = to_tensor(self.annos[key][idx])
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb = self._load_tensor("rgb", idx, dtype=torch.float32)
        if rgb.dim() != 3:
            raise ValueError(f"Expected rgb to have shape [3, H, W], got {tuple(rgb.shape)}.")
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        query_uv = self._load_tensor("query_uv", idx, dtype=torch.float32)
        if query_uv.dim() == 3:
            query_uv = query_uv.unsqueeze(0)
        query_uv = flatten_query_grid(query_uv)

        query_world_xyz = self._load_tensor("query_world_xyz", idx, dtype=torch.float32)
        if query_world_xyz.dim() == 3:
            query_world_xyz = query_world_xyz.unsqueeze(0)
        query_world_xyz = flatten_query_grid(query_world_xyz)

        traj_3d_xyz = self._load_tensor("traj_3d_xyz", idx, dtype=torch.float32)
        if traj_3d_xyz.dim() == 4:
            traj_3d_xyz = traj_3d_xyz.unsqueeze(0)
        traj_3d_xyz = flatten_future_grid(traj_3d_xyz)

        traj_mask = self._load_tensor("traj_mask", idx).bool()
        if traj_mask.dim() == 3:
            traj_mask = traj_mask.all(dim=-1)
        if traj_mask.dim() == 3:
            traj_mask = traj_mask.unsqueeze(0)
        traj_mask = flatten_future_grid(traj_mask).bool()

        rgb, query_uv_norm = aspect_resize_and_pad_rgb_and_query_uv(
            rgb=rgb,
            query_uv=query_uv.float(),
            target_hw=self.image_size,
            uv_is_normalized=self.query_uv_is_normalized,
            pad_value=self.image_pad_value,
        )
        traj_3d_xyz = fill_invalid_traj_with_last_valid(
            traj_3d_xyz=traj_3d_xyz.float(),
            traj_mask=traj_mask,
            anchor_xyz=query_world_xyz.float(),
        )
        future_delta_xyz = compute_stepwise_future_delta(
            traj_3d_xyz=traj_3d_xyz,
            query_world_xyz=query_world_xyz.float(),
        )

        instruction_id = int(self._load_tensor(self.instruction_id_key, idx).item())
        instruction_text = self.instructions.resolve(instruction_id)

        return {
            "rgb": rgb,
            "instruction_text": instruction_text,
            "instruction_id": instruction_id,
            "query_uv": query_uv_norm.float(),
            "query_world_xyz": query_world_xyz.float(),
            "traj_3d_xyz": traj_3d_xyz.float(),
            "future_delta_xyz": future_delta_xyz.float(),
            "traj_mask": traj_mask.bool(),
            "image_hw": torch.tensor(self.image_size, dtype=torch.long),
        }


class MotionExpertCollator:
    def __init__(self, tokenizer: MotionExpertQwenTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> MotionExpertBatch:
        instructions = [sample["instruction_text"] for sample in batch]
        tokenized = self.tokenizer(instructions)
        attention_mask = tokenized.attention_mask
        return MotionExpertBatch(
            rgb=torch.stack([sample["rgb"] for sample in batch]),
            instruction_text=instructions,
            instruction_id=torch.tensor(
                [sample["instruction_id"] for sample in batch],
                dtype=torch.long,
            ),
            text_token_ids=tokenized.input_ids,
            text_attention_mask=attention_mask,
            text_valid_lens=attention_mask.sum(dim=1),
            query_uv=torch.stack([sample["query_uv"] for sample in batch]),
            query_world_xyz=torch.stack([sample["query_world_xyz"] for sample in batch]),
            traj_3d_xyz=torch.stack([sample["traj_3d_xyz"] for sample in batch]),
            future_delta_xyz=torch.stack([sample["future_delta_xyz"] for sample in batch]),
            traj_mask=torch.stack([sample["traj_mask"] for sample in batch]),
            image_hw=torch.stack([sample["image_hw"] for sample in batch]),
        )


def compute_query_xyz_statistics(
    zarr_paths: Sequence[str],
    query_key: str = "query_world_xyz",
    mem_limit_gb: float = 16.0,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xyz_min = None
    xyz_max = None

    for zarr_path in zarr_paths:
        annos = read_zarr_with_cache(zarr_path, mem_gb=mem_limit_gb)
        arr = annos[query_key]
        for start in range(0, len(arr), chunk_size):
            chunk = np.asarray(arr[start : start + chunk_size], dtype=np.float32).reshape(-1, 3)
            finite = np.isfinite(chunk).all(axis=1)
            if not finite.any():
                continue
            chunk = chunk[finite]
            chunk_min = torch.from_numpy(chunk.min(axis=0))
            chunk_max = torch.from_numpy(chunk.max(axis=0))
            xyz_min = chunk_min if xyz_min is None else torch.minimum(xyz_min, chunk_min)
            xyz_max = chunk_max if xyz_max is None else torch.maximum(xyz_max, chunk_max)

    if xyz_min is None or xyz_max is None:
        raise RuntimeError("Could not compute query XYZ statistics from the provided zarr files.")

    center = (xyz_min + xyz_max) / 2.0
    scale = torch.clamp((xyz_max - xyz_min) / 2.0, min=1e-6)
    return center.float(), scale.float()
