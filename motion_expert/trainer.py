from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
    ):
        self.pipeline = pipeline.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(
            self.pipeline.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.clip_grad_norm = float(clip_grad_norm)
        self.mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.best_val_loss = float("inf")

    def _autocast(self):
        if not self.mixed_precision:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def train_epoch(self) -> Dict[str, float]:
        self.pipeline.train()
        total_loss = 0.0
        total_batches = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = self.pipeline(batch)
                loss = outputs["loss"]
            loss.backward()
            if self.clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
        return {
            "loss": total_loss / max(total_batches, 1),
        }

    @torch.no_grad()
    def evaluate_loss(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.pipeline.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            with self._autocast():
                outputs = self.pipeline(batch)
            total_loss += float(outputs["loss"].item())
            total_batches += 1
        return {
            "loss": total_loss / max(total_batches, 1),
        }

    @torch.no_grad()
    def evaluate_generation(
        self,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        max_batches: int = 1,
    ) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.pipeline.eval()
        scheduler = build_default_inference_scheduler(
            num_train_timesteps=self.pipeline.loss_fn.num_train_timesteps
        )
        totals: Dict[str, float] = {}
        count = 0
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= max_batches:
                break
            batch = batch.to(self.device)
            outputs = self.pipeline.predict(
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

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "best_val_loss": self.best_val_loss,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        return path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.pipeline.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        return int(checkpoint.get("epoch", -1)) + 1
