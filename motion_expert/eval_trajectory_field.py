#!/usr/bin/env python3
"""Decode the MotionExpert trajectory field on a held-out zarr and score it
against the SpatialTrackerV2 point tracks stored in that zarr.

The zarr keeps ``traj_3d_xyz`` as *stepwise* deltas, so the position of a query
point after ``t`` future steps is ``cumsum(delta)[t]`` relative to the query
point. Errors below are on those cumulative positions, i.e. "where does the
model think this point ends up", which is what a trajectory-field figure shows.

Two headline numbers per run, both in centimetres:

  * ``mean_first_K``  - error averaged over every valid (point, step) pair with
                        step < K. "How good is the field over the first K steps."
  * ``endpoint_K``    - error at step K, averaged over the points whose track is
                        valid for the whole K-step prefix.

A model trained with a longer horizon (H=32) is scored by truncating its
prediction to the first K steps, so H=15 and H=32 are compared on identical
inputs and identical ground truth.

Example
-------
  python -m motion_expert.eval_trajectory_field \
      --config motion_expert/configs/calvin_florence2.yaml \
      --checkpoint train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt \
      --horizon 15 --output eval_logs/motion_field/calvin_h15.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .config import build_motion_expert_config
from .pipeline import MotionExpertPipeline
from .zarr_dataset import (
    MotionExpertCollator,
    MotionExpertQwenTokenizer,
    MotionExpertZarrDataset,
)

#: GT displacement (cm, over the scored horizon) above which a point counts as
#: "moved". Mirrors ``motion_weight_tau`` = 0.005 m used by the training loss.
DEFAULT_MOVED_THRESHOLD_CM = 0.5


class ConstantInstructionTable:
    """Stand-in for ``InstructionTable`` that returns one fixed string.

    Swapped onto the dataset to ablate language: every query frame gets the same
    uninformative text, so whatever accuracy survives is what the field recovers
    from geometry and vision alone. Defined at module level so DataLoader worker
    processes can pickle it.
    """

    def __init__(self, text: str):
        self.text = str(text)

    def resolve(self, instruction_id: int) -> str:  # noqa: ARG002 - id is ignored
        return self.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="MotionExpert yaml config.")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint (.pt).")
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Number of future steps to score (K). Clipped to what model and GT provide.",
    )
    parser.add_argument("--val-zarr", default=None, help="Override data.val_zarr.")
    parser.add_argument(
        "--val-instructions",
        default=None,
        help="Override data.val_instructions_path.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "split_manifest.json next to the zarr. Combined with --tasks, restricts "
            "scoring to the rows whose `val_files` entry belongs to those tasks."
        ),
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names to keep (requires --split-manifest).",
    )
    parser.add_argument(
        "--manifest-split-key",
        default="val_files",
        help="Which manifest key lists the per-row source files.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Score at most this many rows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Rows per forward pass.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=None, help="Defaults to data.chunk_size.")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Autocast dtype for the decode. Flash-attn kernels stay bf16 either way.",
    )
    parser.add_argument(
        "--moved-threshold-cm",
        type=float,
        default=DEFAULT_MOVED_THRESHOLD_CM,
        help="GT displacement over the horizon above which a point counts as moving.",
    )
    parser.add_argument(
        "--instruction-override",
        default=None,
        help=(
            "Replace every instruction with this exact string (language ablation). "
            "The instruction id is ignored, so all frames share one prompt."
        ),
    )
    parser.add_argument("--output", default=None, help="Write the metric json here.")
    parser.add_argument(
        "--dump-npz",
        default=None,
        help="Optional per-point error dump (endpoint/mean errors + GT displacement).",
    )
    parser.add_argument("--tag", default=None, help="Label recorded in the output json.")
    return parser.parse_args()


def load_raw_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def task_of_manifest_path(path: str) -> Optional[str]:
    """``.../<task>/<segment>/annotations/<frame>.npz`` -> ``<task>``."""
    parts = path.split("/")
    try:
        anchor = parts.index("annotations")
    except ValueError:
        return None
    if anchor < 2:
        return None
    return parts[anchor - 2]


def resolve_row_indices(
    args: argparse.Namespace,
    num_rows: int,
) -> Optional[np.ndarray]:
    if args.tasks is None:
        return None
    if args.split_manifest is None:
        raise ValueError("--tasks requires --split-manifest.")
    wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
    with open(args.split_manifest, "r") as handle:
        manifest = json.load(handle)
    files = manifest[args.manifest_split_key]
    if len(files) != num_rows:
        raise ValueError(
            f"Manifest `{args.manifest_split_key}` has {len(files)} entries but the zarr "
            f"has {num_rows} rows; the manifest does not describe this store."
        )
    labels = [task_of_manifest_path(path) for path in files]
    keep = np.asarray([idx for idx, task in enumerate(labels) if task in wanted], dtype=np.int64)
    found = sorted({task for task in labels if task in wanted})
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"Tasks not present in the manifest split: {missing}")
    print(f"[eval] task filter kept {keep.size}/{num_rows} rows for tasks {found}", flush=True)
    return keep


def chunk_loader_batch_size(raw_batch_size: int, chunk_size: int) -> int:
    return max(1, int(raw_batch_size) // max(1, int(chunk_size)))


class TrajectoryFieldAccumulator:
    """Streams per-(point, step) cumulative-position errors into summary stats.

    ``prefix_valid[t]`` = every step up to and including ``t`` is labelled valid.
    Cumulative positions are only meaningful over such a prefix, because a gap in
    the SpatialTrackerV2 track breaks the running sum.
    """

    def __init__(self, horizon: int, moved_threshold_cm: float):
        self.horizon = int(horizon)
        self.moved_threshold_cm = float(moved_threshold_cm)
        self.step_err_sum = np.zeros(self.horizon, dtype=np.float64)
        self.step_err_sqsum = np.zeros(self.horizon, dtype=np.float64)
        self.step_count = np.zeros(self.horizon, dtype=np.float64)
        self.step_err_sum_rawmask = np.zeros(self.horizon, dtype=np.float64)
        self.step_count_rawmask = np.zeros(self.horizon, dtype=np.float64)
        self.step_delta_err_sum = np.zeros(self.horizon, dtype=np.float64)
        self.num_rows = 0
        self.num_points = 0
        self.num_points_full_horizon = 0
        self.num_gap_points = 0
        self._endpoint_err: List[np.ndarray] = []
        self._endpoint_gt_disp: List[np.ndarray] = []
        self._point_mean_err: List[np.ndarray] = []
        self._point_gt_disp: List[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        pred_delta: torch.Tensor,
        gt_delta: torch.Tensor,
        traj_mask: torch.Tensor,
    ) -> None:
        """pred/gt: (B, P, T, 3) stepwise deltas in metres. mask: (B, P, T)."""
        horizon = self.horizon
        pred = pred_delta[:, :, :horizon, :].double()
        gt = gt_delta[:, :, :horizon, :].double()
        mask = traj_mask[:, :, :horizon].bool()

        prefix = torch.cumprod(mask.to(torch.uint8), dim=2).bool()
        # A step that is valid but sits behind an invalid one: cumulative sums
        # cannot be trusted there, so it is excluded and counted separately.
        self.num_gap_points += int((mask & ~prefix).any(dim=2).sum().item())

        pred_cum = pred.cumsum(dim=2)
        gt_cum = gt.cumsum(dim=2)
        err_cm = torch.linalg.norm(pred_cum - gt_cum, dim=-1) * 100.0
        delta_err_cm = torch.linalg.norm(pred - gt, dim=-1) * 100.0
        gt_disp_cm = torch.linalg.norm(gt_cum, dim=-1) * 100.0

        prefix_f = prefix.to(torch.float64)
        mask_f = mask.to(torch.float64)
        self.step_err_sum += (err_cm * prefix_f).sum(dim=(0, 1)).cpu().numpy()
        self.step_err_sqsum += (err_cm.pow(2) * prefix_f).sum(dim=(0, 1)).cpu().numpy()
        self.step_count += prefix_f.sum(dim=(0, 1)).cpu().numpy()
        self.step_err_sum_rawmask += (err_cm * mask_f).sum(dim=(0, 1)).cpu().numpy()
        self.step_count_rawmask += mask_f.sum(dim=(0, 1)).cpu().numpy()
        self.step_delta_err_sum += (delta_err_cm * prefix_f).sum(dim=(0, 1)).cpu().numpy()

        self.num_rows += int(pred.shape[0])
        self.num_points += int(pred.shape[0] * pred.shape[1])

        # Endpoint: the K-th step, restricted to points valid across the prefix.
        endpoint_valid = prefix[:, :, horizon - 1]
        self.num_points_full_horizon += int(endpoint_valid.sum().item())
        if bool(endpoint_valid.any()):
            self._endpoint_err.append(
                err_cm[:, :, horizon - 1][endpoint_valid].float().cpu().numpy()
            )
            self._endpoint_gt_disp.append(
                gt_disp_cm[:, :, horizon - 1][endpoint_valid].float().cpu().numpy()
            )

        # Per-point mean over its own valid prefix, plus that point's GT motion.
        counts = prefix_f.sum(dim=2)
        has_any = counts > 0
        if bool(has_any.any()):
            point_mean = (err_cm * prefix_f).sum(dim=2) / counts.clamp_min(1.0)
            last_idx = (counts.long() - 1).clamp_min(0)
            point_disp = gt_disp_cm.gather(2, last_idx.unsqueeze(-1)).squeeze(-1)
            self._point_mean_err.append(point_mean[has_any].float().cpu().numpy())
            self._point_gt_disp.append(point_disp[has_any].float().cpu().numpy())

    @staticmethod
    def _concat(chunks: Sequence[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "endpoint_err_cm": self._concat(self._endpoint_err),
            "endpoint_gt_disp_cm": self._concat(self._endpoint_gt_disp),
            "point_mean_err_cm": self._concat(self._point_mean_err),
            "point_gt_disp_cm": self._concat(self._point_gt_disp),
        }

    def summary(self) -> Dict[str, object]:
        arrays = self.arrays()
        endpoint = arrays["endpoint_err_cm"].astype(np.float64)
        endpoint_disp = arrays["endpoint_gt_disp_cm"].astype(np.float64)
        point_mean = arrays["point_mean_err_cm"].astype(np.float64)
        point_disp = arrays["point_gt_disp_cm"].astype(np.float64)

        total_count = float(self.step_count.sum())
        mean_first_k = float(self.step_err_sum.sum() / max(total_count, 1.0))
        per_step_mean = (self.step_err_sum / np.maximum(self.step_count, 1.0)).tolist()
        per_step_delta_mean = (
            self.step_delta_err_sum / np.maximum(self.step_count, 1.0)
        ).tolist()

        def _stats(values: np.ndarray) -> Dict[str, float]:
            if values.size == 0:
                return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
            return {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }

        moved_end = endpoint_disp > self.moved_threshold_cm
        moved_point = point_disp > self.moved_threshold_cm

        return {
            "horizon": self.horizon,
            "num_rows": self.num_rows,
            "num_points": self.num_points,
            "num_points_full_horizon": self.num_points_full_horizon,
            "num_points_with_mask_gap": self.num_gap_points,
            "moved_threshold_cm": self.moved_threshold_cm,
            # headline numbers
            "mean_first_K_cm": mean_first_k,
            "endpoint_K_cm": _stats(endpoint)["mean"],
            # distributions
            "endpoint_cm": _stats(endpoint),
            "point_mean_cm": _stats(point_mean),
            "endpoint_moved_cm": _stats(endpoint[moved_end]),
            "endpoint_static_cm": _stats(endpoint[~moved_end]),
            "point_mean_moved_cm": _stats(point_mean[moved_point]),
            "point_mean_static_cm": _stats(point_mean[~moved_point]),
            "moved_fraction_endpoint": float(moved_end.mean()) if moved_end.size else float("nan"),
            "gt_endpoint_disp_cm": _stats(endpoint_disp),
            # per-step curves
            "per_step_mean_cm": per_step_mean,
            "per_step_count": self.step_count.tolist(),
            "per_step_stepwise_delta_mean_cm": per_step_delta_mean,
            # sanity variant that ignores the prefix requirement
            "mean_first_K_cm_rawmask": float(
                self.step_err_sum_rawmask.sum() / max(float(self.step_count_rawmask.sum()), 1.0)
            ),
        }


def build_pipeline(
    raw_cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> MotionExpertPipeline:
    model_cfg = build_motion_expert_config(raw_cfg["model"])
    if model_cfg.florence.model_name_or_path is not None:
        os.environ.setdefault("FLORENCE2_MODEL_PATH", model_cfg.florence.model_name_or_path)
    if model_cfg.qwen.model_name_or_path is not None:
        os.environ.setdefault("QWEN3VL_MODEL_PATH", model_cfg.qwen.model_name_or_path)

    pipeline = MotionExpertPipeline.from_backbone(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    # Florence2 buffers are rebuilt from the pretrained snapshot, so a few
    # non-persistent entries can legitimately be absent; anything under
    # `model.` (the trained decoder) must match exactly.
    critical_missing = [key for key in missing if key.startswith("model.")]
    if critical_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch. missing(model.*)={critical_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    if missing:
        print(f"[eval] {len(missing)} backbone buffers restored from pretrained weights", flush=True)
    print(
        "[eval] loaded step="
        f"{checkpoint.get('step')} future_steps={model_cfg.future_steps} "
        f"percentile_norm={bool(pipeline.model.use_calvin_percentile_normalization.item())} "
        f"traj_scale={pipeline.model.traj_delta_xyz_scale.tolist()}",
        flush=True,
    )
    del checkpoint
    pipeline = pipeline.to(device)
    pipeline.eval()
    return pipeline


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    raw_cfg = load_raw_config(args.config)
    data_cfg = raw_cfg["data"]
    model_cfg = build_motion_expert_config(raw_cfg["model"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    val_zarr = args.val_zarr or data_cfg.get("val_zarr")
    if not val_zarr:
        raise ValueError("No validation zarr: pass --val-zarr or set data.val_zarr.")
    val_instructions = (
        args.val_instructions
        or data_cfg.get("val_instructions_path")
        or data_cfg.get("train_instructions_path")
        or data_cfg["instructions_path"]
    )
    chunk_size = int(args.chunk_size if args.chunk_size is not None else data_cfg.get("chunk_size", 1))

    backbone_type = str(model_cfg.backbone_type).lower()
    tokenizer = MotionExpertQwenTokenizer(
        backbone_type=backbone_type,
        model_name_or_path=(
            model_cfg.qwen.model_name_or_path
            if backbone_type == "qwen"
            else model_cfg.florence.model_name_or_path
        ),
        max_length=model_cfg.text_max_length,
    )
    collator = MotionExpertCollator(tokenizer)

    # A row filter needs row-level indexing, which the packed chunk layout hides.
    needs_row_indices = args.tasks is not None
    if needs_row_indices:
        chunk_size = 1

    dataset = MotionExpertZarrDataset(
        root=val_zarr,
        instructions_path=val_instructions,
        image_size=(model_cfg.input_height, model_cfg.input_width),
        instruction_id_key=data_cfg.get("instruction_id_key"),
        query_uv_format=model_cfg.query_uv_format,
        image_pad_value=model_cfg.image_pad_value,
        mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
        traj_3d_xyz_is_delta=data_cfg.get("traj_3d_xyz_is_delta"),
        chunk_size=chunk_size,
        camera_frame=data_cfg.get("camera_frame"),
    )
    if args.instruction_override is not None:
        dataset.instructions = ConstantInstructionTable(args.instruction_override)
        print(f"[eval] LANGUAGE ABLATION: every instruction -> {args.instruction_override!r}", flush=True)

    num_rows_total = dataset._raw_length

    eval_set = dataset
    row_indices = resolve_row_indices(args, num_rows_total)
    if row_indices is not None:
        if args.max_rows is not None:
            row_indices = row_indices[: args.max_rows]
        eval_set = Subset(dataset, row_indices.tolist())
    elif args.max_rows is not None:
        limit = int(np.ceil(args.max_rows / float(chunk_size)))
        eval_set = Subset(dataset, list(range(min(limit, len(dataset)))))

    loader = DataLoader(
        eval_set,
        batch_size=chunk_loader_batch_size(args.batch_size, chunk_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    pipeline = build_pipeline(raw_cfg, args.checkpoint, device)

    gt_horizon = int(dataset.annos["traj_3d_xyz"].shape[2])
    horizon = min(int(args.horizon), gt_horizon, int(model_cfg.future_steps))
    if horizon != args.horizon:
        print(
            f"[eval] horizon clipped to {horizon} "
            f"(requested={args.horizon} gt={gt_horizon} model={model_cfg.future_steps})",
            flush=True,
        )

    accumulator = TrajectoryFieldAccumulator(horizon, args.moved_threshold_cm)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.precision == "bf16" and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            # Drop the targets before decoding: pipeline.predict would otherwise
            # score its own metrics, which breaks when the model horizon and the
            # GT horizon differ (H=32 model vs 15-step SpatialTrackerV2 labels).
            decode_batch = dataclasses.replace(batch, future_delta_xyz=None, traj_mask=None)
            with autocast_ctx:
                outputs = pipeline.predict(
                    decode_batch,
                    num_inference_steps=args.num_inference_steps,
                )
            accumulator.update(
                pred_delta=outputs["pred_delta_xyz"].float(),
                gt_delta=batch.future_delta_xyz.float().flatten(1, 2)
                if batch.future_delta_xyz.dim() == 5
                else batch.future_delta_xyz.float(),
                traj_mask=batch.traj_mask.flatten(1, 2)
                if batch.traj_mask.dim() == 4
                else batch.traj_mask,
            )
            if batch_idx % 20 == 0:
                elapsed = time.time() - start
                print(
                    f"[eval] batch {batch_idx + 1}/{len(loader)} "
                    f"rows={accumulator.num_rows} elapsed={elapsed:.0f}s",
                    flush=True,
                )

    summary = accumulator.summary()
    summary["run"] = {
        "tag": args.tag,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "val_zarr": str(val_zarr),
        "val_instructions": str(val_instructions),
        "tasks": args.tasks,
        "model_future_steps": int(model_cfg.future_steps),
        "gt_future_steps": gt_horizon,
        "instruction_override": args.instruction_override,
        "num_inference_steps": int(args.num_inference_steps),
        "precision": args.precision,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "elapsed_s": time.time() - start,
    }

    print(json.dumps(summary, indent=2))
    print(
        f"\n[RESULT] {args.tag or Path(args.config).stem}: "
        f"mean over first {horizon} steps = {summary['mean_first_K_cm']:.3f} cm | "
        f"endpoint at step {horizon} = {summary['endpoint_K_cm']:.3f} cm",
        flush=True,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[eval] wrote {out_path}", flush=True)
    if args.dump_npz:
        npz_path = Path(args.dump_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **accumulator.arrays())
        print(f"[eval] wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
