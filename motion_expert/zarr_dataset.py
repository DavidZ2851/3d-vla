import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, BartTokenizerFast

from datasets.utils import read_zarr_with_cache, to_tensor

from .data import MotionExpertBatch, flatten_future_grid, flatten_query_grid
from .geometry import (
    QUERY_UV_FORMAT_SOURCE_PIXELS,
    aspect_resize_and_pad_rgb_and_query_uv,
    aspect_resize_and_pad_depth,
    center_query_world_xyz,
    compute_stepwise_future_delta,
    fill_invalid_delta_with_zero,
    fill_invalid_traj_with_last_valid,
    normalize_query_uv_format,
    query_uv_depth_to_world_xyz,
    transform_intrinsics_for_resize_pad,
)


def _center_finite_query_xyz_np(query_world_xyz: np.ndarray) -> np.ndarray:
    query_world_xyz = np.asarray(query_world_xyz, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(query_world_xyz).all(axis=1)
    if not finite.any():
        return np.empty((0, 3), dtype=np.float32)
    valid_xyz = query_world_xyz[finite]
    center = valid_xyz.mean(axis=0, keepdims=True)
    return valid_xyz - center


def _update_axis_sums(
    values: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
) -> int:
    if values.size == 0:
        return 0
    values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    sums[:] += values.sum(axis=0)
    sumsq[:] += np.square(values).sum(axis=0)
    return int(values.shape[0])


def _mean_std_from_sums(
    sums: np.ndarray,
    sumsq: np.ndarray,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise RuntimeError("Cannot compute normalization statistics without valid XYZ values.")
    mean = sums / float(count)
    var = np.maximum(sumsq / float(count) - np.square(mean), 1e-12)
    std = np.maximum(np.sqrt(var), 1e-6)
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(std.astype(np.float32)),
    )


def _axis_summary_from_sums(
    sums: np.ndarray,
    sumsq: np.ndarray,
    count: int,
) -> Dict[str, object]:
    mean, std = _mean_std_from_sums(sums, sumsq, count)
    return {
        "count": int(count),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "center": mean.tolist(),
        "scale": std.tolist(),
    }


@dataclass
class TokenizedInstructionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class InstructionTable:
    def __init__(self, instruction_source: str):
        with open(instruction_source, "r") as handle:
            raw = json.load(handle)
        self._instructions = self._parse_instruction_source(raw)

    @staticmethod
    def _coerce_text(value: object) -> str:
        if isinstance(value, dict):
            value = value.get("text", value.get("instruction", value))
        return str(value)

    @classmethod
    def _parse_instruction_source(cls, raw: object) -> Dict[int, str]:
        # New schema: {"schema_version": ..., "instructions": [...]}
        if isinstance(raw, dict) and "instructions" in raw:
            raw = raw["instructions"]

        if isinstance(raw, list):
            mapped: Dict[int, str] = {}
            # Preferred list format: [{"instruction_id": ..., "text": ...}, ...]
            if raw and isinstance(raw[0], dict):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    if "instruction_id" not in entry:
                        continue
                    mapped[int(entry["instruction_id"])] = cls._coerce_text(entry)
                if mapped:
                    return mapped
            # Fallback list format: plain text list indexed by position.
            return {int(idx): cls._coerce_text(text) for idx, text in enumerate(raw)}

        if isinstance(raw, dict):
            mapped = {}
            for key, value in raw.items():
                try:
                    instruction_id = int(key)
                except (TypeError, ValueError):
                    continue
                mapped[instruction_id] = cls._coerce_text(value)
            if mapped:
                return mapped

        raise ValueError(
            f"Unsupported instruction table structure: {type(raw).__name__}."
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
        if self.backbone_type == "florence2":
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                )
                self.tokenizer = self.processor.tokenizer
            except Exception:
                self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
        else:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self.tokenizer = self.processor.tokenizer
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    use_fast=False,
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
        query_uv_format: str = QUERY_UV_FORMAT_SOURCE_PIXELS,
        image_pad_value: float = 0.5,
        mem_limit_gb: float = 16.0,
        traj_3d_xyz_is_delta: Optional[bool] = None,
        max_samples: Optional[int] = None,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit_gb)
        self.instructions = InstructionTable(instructions_path)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.query_uv_format = normalize_query_uv_format(query_uv_format)
        self.image_pad_value = float(image_pad_value)
        self.max_samples = None if max_samples is None else int(max_samples)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self._raw_length = len(self.annos["rgb"])
        self._effective_raw_length = (
            self._raw_length
            if self.max_samples is None
            else min(self._raw_length, self.max_samples)
        )
        self.instruction_id_key = self._resolve_instruction_key(instruction_id_key)
        self.traj_mask_key = self._resolve_traj_mask_key()
        self.traj_3d_xyz_is_delta = self._resolve_traj_delta_semantics(
            root=root,
            instructions_path=instructions_path,
            requested=traj_3d_xyz_is_delta,
        )
        self.depth_key = self._resolve_optional_key(("depth",))
        self.intrinsics_key = self._resolve_optional_key(("intrinsics", "camera_intrinsics"))
        self.extrinsics_key = self._resolve_optional_key(("extrinsics", "camera_extrinsics"))

    def _resolve_optional_key(self, candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in self.annos:
                return candidate
        return None

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

    def _resolve_traj_mask_key(self) -> str:
        for candidate in ("traj_mask", "valid_mask"):
            if candidate in self.annos:
                return candidate
        raise KeyError("Could not find a trajectory mask key. Tried `traj_mask` and `valid_mask`.")

    def _resolve_traj_delta_semantics(
        self,
        root: str,
        instructions_path: str,
        requested: Optional[bool],
    ) -> bool:
        if requested is not None:
            return bool(requested)
        joined = f"{root} {instructions_path}".lower()
        return "calvin" in joined

    def __len__(self) -> int:
        if self._effective_raw_length <= 0:
            return 0
        if self.chunk_size == 1:
            return self._effective_raw_length
        return int(math.ceil(self._effective_raw_length / float(self.chunk_size)))

    def _load_tensor(self, key: str, idx: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        value = to_tensor(self.annos[key][idx])
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _load_tensor_slice(
        self,
        key: str,
        start: int,
        end: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        value = to_tensor(self.annos[key][start:end])
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _build_single_sample_from_prefetched(
        self,
        local_idx: int,
        prefetched: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        rgb = prefetched["rgb"][local_idx].to(dtype=torch.float32)
        if rgb.dim() != 3:
            raise ValueError(f"Expected rgb to have shape [3, H, W], got {tuple(rgb.shape)}.")
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        query_uv = prefetched["query_uv"][local_idx].to(dtype=torch.float32)
        if query_uv.dim() == 3:
            query_uv = query_uv.unsqueeze(0)
        query_uv = flatten_query_grid(query_uv)

        query_world_xyz = None
        if "query_world_xyz" in prefetched:
            query_world_xyz = prefetched["query_world_xyz"][local_idx].to(dtype=torch.float32)
            if query_world_xyz.dim() == 3:
                query_world_xyz = query_world_xyz.unsqueeze(0)
            query_world_xyz = flatten_query_grid(query_world_xyz)

        traj_3d_xyz = prefetched["traj_3d_xyz"][local_idx].to(dtype=torch.float32)
        if traj_3d_xyz.dim() == 4:
            traj_3d_xyz = traj_3d_xyz.unsqueeze(0)
        traj_3d_xyz = flatten_future_grid(traj_3d_xyz)

        traj_mask = prefetched["traj_mask"][local_idx].bool()
        if traj_mask.dim() == 4:
            traj_mask = traj_mask.all(dim=-1)
        if (
            traj_mask.dim() == 3
            and traj_mask.shape[-1] == 3
            and traj_mask.shape[:2] == traj_3d_xyz.shape[:2]
        ):
            traj_mask = traj_mask.all(dim=-1)
        if traj_mask.dim() == 3:
            traj_mask = traj_mask.unsqueeze(0)
        traj_mask = flatten_future_grid(traj_mask).bool()

        src_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        rgb, query_uv_grid = aspect_resize_and_pad_rgb_and_query_uv(
            rgb=rgb,
            query_uv=query_uv.float(),
            target_hw=self.image_size,
            query_uv_format=self.query_uv_format,
            pad_value=self.image_pad_value,
        )
        depth = None
        intrinsics = None
        extrinsics = None
        if "depth" in prefetched:
            depth = prefetched["depth"][local_idx].to(dtype=torch.float32)
            depth = aspect_resize_and_pad_depth(depth, self.image_size)
        if "intrinsics" in prefetched:
            intrinsics = prefetched["intrinsics"][local_idx].to(dtype=torch.float32)
            intrinsics = transform_intrinsics_for_resize_pad(intrinsics, src_hw, self.image_size)
        if "extrinsics" in prefetched:
            extrinsics = prefetched["extrinsics"][local_idx].to(dtype=torch.float32)

        if query_world_xyz is None and (depth is None or intrinsics is None or extrinsics is None):
            raise KeyError(
                "Zarr sample must contain query_world_xyz, or depth + intrinsics + extrinsics."
            )
        if query_world_xyz is None:
            query_world_xyz, _ = query_uv_depth_to_world_xyz(
                query_uv=query_uv_grid.unsqueeze(0),
                depth=depth.unsqueeze(0),
                intrinsics=intrinsics.unsqueeze(0),
                extrinsics=extrinsics.unsqueeze(0),
            )
            query_world_xyz = query_world_xyz.squeeze(0)

        query_world_xyz_raw = query_world_xyz.float()
        if self.traj_3d_xyz_is_delta:
            future_delta_xyz = fill_invalid_delta_with_zero(
                traj_delta_xyz=traj_3d_xyz.float(),
                traj_mask=traj_mask,
            )
            traj_3d_xyz = future_delta_xyz
        else:
            if query_world_xyz is None:
                raise KeyError("Absolute traj_3d_xyz requires query_world_xyz for stepwise deltas.")
            traj_3d_xyz = fill_invalid_traj_with_last_valid(
                traj_3d_xyz=traj_3d_xyz.float(),
                traj_mask=traj_mask,
                anchor_xyz=query_world_xyz_raw,
            )
            future_delta_xyz = compute_stepwise_future_delta(
                traj_3d_xyz=traj_3d_xyz,
                query_world_xyz=query_world_xyz_raw,
            )
        query_world_xyz = center_query_world_xyz(query_world_xyz_raw)

        instruction_id = int(prefetched["instruction_id"][local_idx].item())
        instruction_text = self.instructions.resolve(instruction_id)

        sample = {
            "rgb": rgb,
            "instruction_text": instruction_text,
            "instruction_id": instruction_id,
            "query_uv": query_uv_grid.float(),
            "traj_3d_xyz": traj_3d_xyz.float(),
            "future_delta_xyz": future_delta_xyz.float(),
            "traj_mask": traj_mask.bool(),
            "image_hw": torch.tensor(self.image_size, dtype=torch.long),
        }
        sample["query_world_xyz"] = query_world_xyz.float()
        if depth is not None:
            sample["depth"] = depth.float()
        if intrinsics is not None:
            sample["intrinsics"] = intrinsics.float()
        if extrinsics is not None:
            sample["extrinsics"] = extrinsics.float()
        return sample

    @staticmethod
    def _pack_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(samples) == 1:
            return samples[0]
        packed = {
            "rgb": torch.stack([sample["rgb"] for sample in samples]),
            "instruction_text": [sample["instruction_text"] for sample in samples],
            "instruction_id": torch.tensor(
                [sample["instruction_id"] for sample in samples],
                dtype=torch.long,
            ),
            "query_uv": torch.stack([sample["query_uv"] for sample in samples]),
            "traj_3d_xyz": torch.stack([sample["traj_3d_xyz"] for sample in samples]),
            "future_delta_xyz": torch.stack([sample["future_delta_xyz"] for sample in samples]),
            "traj_mask": torch.stack([sample["traj_mask"] for sample in samples]),
            "image_hw": torch.stack([sample["image_hw"] for sample in samples]),
        }
        if all("query_world_xyz" in sample for sample in samples):
            packed["query_world_xyz"] = torch.stack([sample["query_world_xyz"] for sample in samples])
        if all("depth" in sample for sample in samples):
            packed["depth"] = torch.stack([sample["depth"] for sample in samples])
        if all("intrinsics" in sample for sample in samples):
            packed["intrinsics"] = torch.stack([sample["intrinsics"] for sample in samples])
        if all("extrinsics" in sample for sample in samples):
            packed["extrinsics"] = torch.stack([sample["extrinsics"] for sample in samples])
        return packed

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._effective_raw_length <= 0:
            raise IndexError("Dataset is empty.")

        if self.chunk_size == 1:
            raw_start = idx % self._effective_raw_length
            raw_end = raw_start + 1
        else:
            chunk_idx = idx % len(self)
            raw_start = chunk_idx * self.chunk_size
            raw_end = min(raw_start + self.chunk_size, self._effective_raw_length)

        prefetched: Dict[str, torch.Tensor] = {
            "rgb": self._load_tensor_slice("rgb", raw_start, raw_end, dtype=torch.float32),
            "query_uv": self._load_tensor_slice("query_uv", raw_start, raw_end, dtype=torch.float32),
            "traj_3d_xyz": self._load_tensor_slice("traj_3d_xyz", raw_start, raw_end, dtype=torch.float32),
            "traj_mask": self._load_tensor_slice(self.traj_mask_key, raw_start, raw_end),
            "instruction_id": self._load_tensor_slice(self.instruction_id_key, raw_start, raw_end),
        }
        if "query_world_xyz" in self.annos:
            prefetched["query_world_xyz"] = self._load_tensor_slice(
                "query_world_xyz",
                raw_start,
                raw_end,
                dtype=torch.float32,
            )
        if self.depth_key is not None:
            prefetched["depth"] = self._load_tensor_slice(
                self.depth_key,
                raw_start,
                raw_end,
                dtype=torch.float32,
            )
        if self.intrinsics_key is not None:
            prefetched["intrinsics"] = self._load_tensor_slice(
                self.intrinsics_key,
                raw_start,
                raw_end,
                dtype=torch.float32,
            )
        if self.extrinsics_key is not None:
            prefetched["extrinsics"] = self._load_tensor_slice(
                self.extrinsics_key,
                raw_start,
                raw_end,
                dtype=torch.float32,
            )

        sample_count = int(raw_end - raw_start)
        samples = [
            self._build_single_sample_from_prefetched(local_idx=local_idx, prefetched=prefetched)
            for local_idx in range(sample_count)
        ]
        return self._pack_samples(samples)


class MotionExpertCollator:
    def __init__(self, tokenizer: MotionExpertQwenTokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _expand_packed_samples(sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        rgb = sample["rgb"]
        if not torch.is_tensor(rgb) or rgb.dim() == 3:
            return [sample]

        chunk_len = int(rgb.shape[0])
        expanded: List[Dict[str, torch.Tensor]] = []
        for sample_idx in range(chunk_len):
            flat_sample: Dict[str, torch.Tensor] = {}
            for key, value in sample.items():
                if key == "instruction_text":
                    flat_sample[key] = value[sample_idx]
                    continue
                if torch.is_tensor(value):
                    flat_sample[key] = value[sample_idx]
                    continue
                if isinstance(value, (list, tuple)):
                    flat_sample[key] = value[sample_idx]
                else:
                    flat_sample[key] = value
            expanded.append(flat_sample)
        return expanded

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> MotionExpertBatch:
        flat_batch: List[Dict[str, torch.Tensor]] = []
        for sample in batch:
            flat_batch.extend(self._expand_packed_samples(sample))

        instructions = [sample["instruction_text"] for sample in flat_batch]
        tokenized = self.tokenizer(instructions)
        attention_mask = tokenized.attention_mask
        return MotionExpertBatch(
            rgb=torch.stack([sample["rgb"] for sample in flat_batch]),
            instruction_text=instructions,
            instruction_id=torch.tensor(
                [sample["instruction_id"] for sample in flat_batch],
                dtype=torch.long,
            ),
            text_token_ids=tokenized.input_ids,
            text_attention_mask=attention_mask,
            text_valid_lens=attention_mask.sum(dim=1),
            depth=(
                torch.stack([sample["depth"] for sample in flat_batch])
                if all("depth" in sample for sample in flat_batch)
                else None
            ),
            intrinsics=(
                torch.stack([sample["intrinsics"] for sample in flat_batch])
                if all("intrinsics" in sample for sample in flat_batch)
                else None
            ),
            extrinsics=(
                torch.stack([sample["extrinsics"] for sample in flat_batch])
                if all("extrinsics" in sample for sample in flat_batch)
                else None
            ),
            query_uv=torch.stack([sample["query_uv"] for sample in flat_batch]),
            query_world_xyz=(
                torch.stack([sample["query_world_xyz"] for sample in flat_batch])
                if all("query_world_xyz" in sample for sample in flat_batch)
                else None
            ),
            traj_3d_xyz=torch.stack([sample["traj_3d_xyz"] for sample in flat_batch]),
            future_delta_xyz=torch.stack([sample["future_delta_xyz"] for sample in flat_batch]),
            traj_mask=torch.stack([sample["traj_mask"] for sample in flat_batch]),
            image_hw=torch.stack([sample["image_hw"] for sample in flat_batch]),
        )


def compute_query_xyz_statistics(
    zarr_paths: Sequence[str],
    query_key: str = "query_world_xyz",
    mem_limit_gb: float = 16.0,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    count = 0

    for zarr_path in zarr_paths:
        annos = read_zarr_with_cache(zarr_path, mem_gb=mem_limit_gb)
        arr = annos[query_key]
        for start in range(0, len(arr), chunk_size):
            chunk_arr = np.asarray(arr[start : start + chunk_size], dtype=np.float32)
            chunk = chunk_arr.reshape(chunk_arr.shape[0], -1, 3)
            for sample in chunk:
                centered_sample = _center_finite_query_xyz_np(sample)
                count += _update_axis_sums(centered_sample, sums, sumsq)

    if count <= 0:
        raise RuntimeError("Could not compute query XYZ statistics from the provided zarr files.")

    center, scale = _mean_std_from_sums(sums, sumsq, count)
    return center.float(), scale.float()


def compute_query_xyz_statistics_from_dataset(
    dataset: MotionExpertZarrDataset,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(3, dtype=torch.float64)
    sumsq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for idx in range(len(dataset)):
        query_world_xyz = center_query_world_xyz(dataset[idx]["query_world_xyz"]).reshape(-1, 3)
        finite = torch.isfinite(query_world_xyz).all(dim=-1)
        if not finite.any():
            continue
        query_world_xyz = query_world_xyz[finite].to(torch.float64)
        sums += query_world_xyz.sum(dim=0)
        sumsq += query_world_xyz.square().sum(dim=0)
        count += int(query_world_xyz.shape[0])

    if count <= 0:
        raise RuntimeError("Could not compute query XYZ statistics from the dataset.")
    center = sums / float(count)
    var = torch.clamp(sumsq / float(count) - center.square(), min=1e-12)
    scale = torch.clamp(var.sqrt(), min=1e-6)
    return center.float(), scale.float()


def compute_calvin_zarr_normalization_stats(
    zarr_paths: Sequence[str],
    query_key: str = "query_world_xyz",
    traj_key: str = "traj_3d_xyz",
    mem_limit_gb: float = 16.0,
    chunk_size: int = 512,
) -> Dict[str, object]:
    query_sums = np.zeros(3, dtype=np.float64)
    query_sumsq = np.zeros(3, dtype=np.float64)
    traj_sums = np.zeros(3, dtype=np.float64)
    traj_sumsq = np.zeros(3, dtype=np.float64)
    query_count = 0
    traj_count = 0

    for zarr_path in zarr_paths:
        annos = read_zarr_with_cache(zarr_path, mem_gb=mem_limit_gb)
        if query_key not in annos:
            raise KeyError(f"Calvin zarr normalization requires `{query_key}`.")
        if traj_key not in annos:
            raise KeyError(f"Calvin zarr normalization requires `{traj_key}`.")
        mask_key = "traj_mask" if "traj_mask" in annos else "valid_mask"
        if mask_key not in annos:
            raise KeyError("Calvin zarr normalization requires `traj_mask` or `valid_mask`.")

        query_arr = annos[query_key]
        traj_arr = annos[traj_key]
        mask_arr = annos[mask_key]
        if len(query_arr) != len(traj_arr) or len(query_arr) != len(mask_arr):
            raise ValueError("query, trajectory, and mask arrays must have the same length.")

        for start in range(0, len(query_arr), chunk_size):
            query_chunk_arr = np.asarray(query_arr[start : start + chunk_size], dtype=np.float32)
            query_chunk = query_chunk_arr.reshape(query_chunk_arr.shape[0], -1, 3)
            for query_sample in query_chunk:
                centered_query = _center_finite_query_xyz_np(query_sample)
                query_count += _update_axis_sums(centered_query, query_sums, query_sumsq)

            traj_chunk_arr = np.asarray(traj_arr[start : start + chunk_size], dtype=np.float32)
            traj_chunk = traj_chunk_arr.reshape(traj_chunk_arr.shape[0], -1, traj_chunk_arr.shape[-2], 3)
            mask_chunk_arr = np.asarray(mask_arr[start : start + chunk_size]).astype(bool)
            if mask_chunk_arr.shape == traj_chunk_arr.shape:
                mask_chunk_arr = mask_chunk_arr.all(axis=-1)
            mask_chunk = mask_chunk_arr.reshape(
                traj_chunk_arr.shape[0],
                -1,
                traj_chunk_arr.shape[-2],
            )
            finite_traj = np.isfinite(traj_chunk).all(axis=-1)
            valid_traj = mask_chunk & finite_traj
            traj_values = traj_chunk[valid_traj]
            traj_count += _update_axis_sums(traj_values, traj_sums, traj_sumsq)

    if query_count <= 0 or traj_count <= 0:
        raise RuntimeError("Could not compute Calvin zarr normalization statistics.")

    return {
        "source_zarr_paths": [str(path) for path in zarr_paths],
        "estimation_method": {
            "type": "streaming_mean_std",
            "note": "query_world_xyz is centered per sample before accumulating statistics.",
        },
        "query_centered_xyz": _axis_summary_from_sums(
            query_sums,
            query_sumsq,
            query_count,
        ),
        "traj_delta_xyz": _axis_summary_from_sums(
            traj_sums,
            traj_sumsq,
            traj_count,
        ),
    }
