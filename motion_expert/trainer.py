from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import MotionExpertConfig
from .data import MotionExpertBatch
from .pipeline import MotionExpertPipeline
from .tracegen_compat import build_default_inference_scheduler
from .visualization import save_motion_expert_spaghetti_batch


class MotionExpertTrainer:
    def __init__(
        self,
        pipeline: MotionExpertPipeline,
        config: MotionExpertConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        output_dir: str,
        lr: float = 1e-4,
        weight_decay: float = 5e-3,
        clip_grad_norm: float = 1.0,
        mixed_precision: bool = False,
        local_rank: int = 0,
        distributed: bool = False,
        find_unused_parameters: bool = False,
        train_sampler: Optional[DistributedSampler] = None,
    ):
        pipeline = pipeline.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.train_sampler = train_sampler
        if self.distributed:
            pipeline = DistributedDataParallel(
                pipeline,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        self.pipeline = pipeline
        trainable_params = [param for param in self.pipeline.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("MotionExpertTrainer requires at least one trainable parameter.")
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = None
        self.clip_grad_norm = float(clip_grad_norm)
        self.mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.best_val_loss = float("inf")
        self._train_iter = None
        self._train_epoch = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._train_epoch)

    @property
    def module(self) -> MotionExpertPipeline:
        if isinstance(self.pipeline, DistributedDataParallel):
            return self.pipeline.module
        return self.pipeline

    def _autocast(self):
        if not self.mixed_precision:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _is_main_process(self) -> bool:
        if not self.distributed:
            return True
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _next_train_batch(self) -> MotionExpertBatch:
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_epoch += 1
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self._train_epoch)
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return batch

    def configure_lr_scheduler(
        self,
        total_steps: int,
        scheduler_type: str = "cosine_with_warmup",
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
    ) -> None:
        total_steps = max(1, int(total_steps))
        warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
        scheduler_type = str(scheduler_type).lower()
        if scheduler_type in ("none", "constant"):
            self.lr_scheduler = None
            return
        if scheduler_type == "cosine":
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=float(min_lr),
            )
            return
        if scheduler_type == "cosine_with_warmup":
            if warmup_steps <= 0:
                self.lr_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                    eta_min=float(min_lr),
                )
                return
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=float(min_lr),
            )
            self.lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            return
        raise ValueError(
            "Unsupported lr scheduler type. Use one of: "
            "`none`, `constant`, `cosine`, `cosine_with_warmup`."
        )

    @staticmethod
    def _update_scalar_totals(
        totals: Dict[str, float],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        for key, value in outputs.items():
            if not torch.is_tensor(value) or value.ndim != 0:
                continue
            totals[key] = totals.get(key, 0.0) + float(value.item())

    def train_step(
        self,
        batch: Optional[MotionExpertBatch] = None,
    ) -> Dict[str, float]:
        self.pipeline.train()
        batch = self._next_train_batch() if batch is None else batch
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            outputs = self.pipeline(batch)
            loss = outputs["loss"]
        loss.backward()
        if self.clip_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        totals: Dict[str, float] = {}
        self._update_scalar_totals(totals, outputs)
        return totals

    def train_epoch(self) -> Dict[str, float]:
        self.pipeline.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._train_epoch)
        self._train_iter = None
        totals: Dict[str, float] = {}
        total_batches = 0
        for batch in self.train_loader:
            step_metrics = self.train_step(batch)
            for key, value in step_metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            total_batches += 1
        self._train_epoch += 1
        self._train_iter = None
        return {
            key: value / max(total_batches, 1)
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate_loss(
        self,
        loader: Optional[DataLoader] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        loader = self.val_loader if loader is None else loader
        if loader is None:
            return {}
        self.module.eval()
        totals: Dict[str, float] = {}
        total_batches = 0
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(self.device)
            with self._autocast():
                outputs = self.module(batch)
            self._update_scalar_totals(totals, outputs)
            total_batches += 1
        return {
            key: value / max(total_batches, 1)
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate_generation(
        self,
        loader: Optional[DataLoader] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        max_batches: int = 1,
        spaghetti_viz_output_dir: Optional[str] = None,
        spaghetti_viz_prefix: str = "val",
        spaghetti_viz_max_samples: int = 0,
        spaghetti_motion_threshold: float = 0.005,
        spaghetti_max_points_per_sample: int = 200,
        spaghetti_rgb_output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        loader = self.val_loader if loader is None else loader
        if loader is None:
            return {}
        self.module.eval()
        scheduler = build_default_inference_scheduler(
            num_train_timesteps=self.module.loss_fn.num_train_timesteps
        )
        totals: Dict[str, float] = {}
        count = 0
        viz_saved = 0
        viz_enabled = (
            spaghetti_viz_output_dir is not None
            and int(spaghetti_viz_max_samples) > 0
        )
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = batch.to(self.device)
            autocast_ctx = self._autocast() if self.mixed_precision else nullcontext()
            with autocast_ctx:
                outputs = self.module.predict(
                    batch,
                    scheduler=scheduler,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            for key, value in outputs.items():
                if key == "pred_delta_xyz":
                    continue
                totals[key] = totals.get(key, 0.0) + float(value.item())

            if viz_enabled and viz_saved < int(spaghetti_viz_max_samples):
                pred_delta_xyz = outputs.get("pred_delta_xyz")
                if (
                    pred_delta_xyz is not None
                    and batch.query_world_xyz is not None
                    and batch.future_delta_xyz is not None
                    and batch.traj_mask is not None
                ):
                    remaining = int(spaghetti_viz_max_samples) - viz_saved
                    try:
                        saved_paths = save_motion_expert_spaghetti_batch(
                            query_world_xyz=batch.query_world_xyz.detach().cpu(),
                            gt_delta_xyz=batch.future_delta_xyz.detach().cpu(),
                            pred_delta_xyz=pred_delta_xyz.detach().cpu(),
                            traj_mask=batch.traj_mask.detach().cpu(),
                            output_dir=spaghetti_viz_output_dir,
                            prefix=spaghetti_viz_prefix,
                            start_index=viz_saved,
                            max_samples=remaining,
                            motion_threshold=float(spaghetti_motion_threshold),
                            max_points_per_sample=int(spaghetti_max_points_per_sample),
                            instruction_text=batch.instruction_text,
                            rgb=batch.rgb.detach().cpu() if batch.rgb is not None else None,
                            rgb_output_dir=spaghetti_rgb_output_dir,
                        )
                        viz_saved += len(saved_paths)
                    except Exception as exc:
                        print(f"[motion_expert][viz] failed to render spaghetti plots: {exc}")
            count += 1
        averaged = {key: value / max(count, 1) for key, value in totals.items()}
        if viz_enabled:
            averaged["spaghetti_viz_saved"] = float(viz_saved)
        return averaged

    def save_checkpoint(
        self,
        step: int,
        *,
        is_best: bool = False,
        save_latest: bool = True,
        archive_step: bool = False,
    ) -> Path:
        last_path = self.output_dir / "last_model.pt"
        best_path = self.output_dir / "best_model.pt"
        archive_path = self.output_dir / f"checkpoint_step_{int(step)}.pt"
        if not self._is_main_process():
            if is_best:
                return best_path
            if save_latest:
                return last_path
            if archive_step:
                return archive_path
            return best_path
        checkpoint = {
            "step": int(step),
            "epoch": int(self._train_epoch),
            "train_epoch": int(self._train_epoch),
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_val_loss": self.best_val_loss,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if save_latest:
            torch.save(checkpoint, last_path)
        if archive_step:
            torch.save(checkpoint, archive_path)
        if is_best:
            torch.save(checkpoint, best_path)

        if save_latest:
            return last_path
        if archive_step:
            return archive_path
        return best_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self._train_iter = None
        if "train_epoch" in checkpoint:
            self._train_epoch = int(checkpoint["train_epoch"])
        elif "step" not in checkpoint and "epoch" in checkpoint:
            self._train_epoch = int(checkpoint["epoch"])
        else:
            self._train_epoch = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self._train_epoch)
        if "step" in checkpoint:
            return int(checkpoint["step"])
        return int(checkpoint.get("epoch", -1)) + 1
