from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import MotionExpertConfig
from .data import MotionExpertBatch
from .pipeline import MotionExpertPipeline
from .tracegen_compat import build_default_inference_scheduler


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
        self.optimizer = AdamW(
            self.pipeline.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
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

        totals: Dict[str, float] = {}
        self._update_scalar_totals(totals, outputs)
        return totals

    def train_epoch(self) -> Dict[str, float]:
        self.pipeline.train()
        totals: Dict[str, float] = {}
        total_batches = 0
        for batch in self.train_loader:
            step_metrics = self.train_step(batch)
            for key, value in step_metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            total_batches += 1
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
            count += 1
        return {key: value / max(count, 1) for key, value in totals.items()}

    def save_checkpoint(self, step: int, is_best: bool = False) -> Path:
        checkpoint = {
            "step": int(step),
            "epoch": int(step),
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_val_loss": self.best_val_loss,
        }
        path = self.output_dir / f"checkpoint_step_{int(step)}.pt"
        torch.save(checkpoint, path)
        last_path = self.output_dir / "last_model.pt"
        torch.save(checkpoint, last_path)
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        return path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        if "step" in checkpoint:
            return int(checkpoint["step"])
        return int(checkpoint.get("epoch", -1)) + 1
