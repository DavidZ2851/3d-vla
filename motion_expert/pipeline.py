from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import CogVideoXDDIMScheduler
from einops import rearrange

from .config import MotionExpertConfig
from .data import MotionExpertBatch, flatten_future_grid, flatten_query_grid
from .extractor import (
    FlorenceTraceFeatureExtractor,
    MotionExpertFeatureOutput,
    QwenTraceFeatureExtractor,
)
from .model import MotionExpertModel
from .tracegen_compat import build_default_inference_scheduler


class MotionExpertFlowLoss(nn.Module):
    def __init__(self, num_train_timesteps: int = 100):
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.noise_scheduler = CogVideoXDDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        traj_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        noise_pred = rearrange(predictions["noise_pred"], "b c t h w -> b (h w) t c")
        noise_target = rearrange(predictions["noise_target"], "b c t h w -> b (h w) t c")
        mask = traj_mask.unsqueeze(-1).to(noise_pred.dtype)
        diff_sq = (noise_pred - noise_target).pow(2) * mask
        valid_count = mask.sum().clamp_min(1.0) * noise_pred.shape[-1]
        diffusion_loss = diff_sq.sum() / valid_count
        return {
            "loss": diffusion_loss,
            "diffusion_loss": diffusion_loss,
        }


def compute_motion_expert_metrics(
    pred_delta_xyz: torch.Tensor,
    target_delta_xyz: torch.Tensor,
    traj_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    mask = traj_mask.bool().unsqueeze(-1)
    mask_f = mask.to(pred_delta_xyz.dtype)
    total_valid = mask_f.sum().clamp_min(1.0) * pred_delta_xyz.shape[-1]
    diff = pred_delta_xyz - target_delta_xyz
    delta_l1 = (diff.abs() * mask_f).sum() / total_valid
    delta_mse = (diff.pow(2) * mask_f).sum() / total_valid

    valid_counts = traj_mask.bool().sum(dim=-1)
    endpoint_valid = valid_counts > 0
    if endpoint_valid.any():
        gather_index = (valid_counts.clamp_min(1) - 1)[..., None, None].expand(-1, -1, 1, 3)
        pred_endpoint = pred_delta_xyz.gather(2, gather_index).squeeze(2)
        target_endpoint = target_delta_xyz.gather(2, gather_index).squeeze(2)
        endpoint_mask = endpoint_valid.unsqueeze(-1).to(pred_delta_xyz.dtype)
        endpoint_l1 = (
            (pred_endpoint - target_endpoint).abs() * endpoint_mask
        ).sum() / (endpoint_mask.sum().clamp_min(1.0) * pred_delta_xyz.shape[-1])
    else:
        endpoint_l1 = pred_delta_xyz.new_tensor(0.0)

    return {
        "delta_l1": delta_l1,
        "delta_mse": delta_mse,
        "endpoint_l1": endpoint_l1,
    }


class MotionExpertPipeline(nn.Module):
    def __init__(
        self,
        config: MotionExpertConfig,
        feature_extractor: Optional[torch.nn.Module] = None,
        model: Optional[MotionExpertModel] = None,
        loss_fn: Optional[MotionExpertFlowLoss] = None,
    ):
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.model = model or MotionExpertModel(config)
        self.loss_fn = loss_fn or MotionExpertFlowLoss()

    @classmethod
    def from_qwen(cls, config: MotionExpertConfig) -> "MotionExpertPipeline":
        feature_extractor = QwenTraceFeatureExtractor(config.qwen)
        return cls._from_feature_extractor(config, feature_extractor)

    @classmethod
    def from_florence(cls, config: MotionExpertConfig) -> "MotionExpertPipeline":
        feature_extractor = FlorenceTraceFeatureExtractor(config.florence)
        return cls._from_feature_extractor(config, feature_extractor)

    @classmethod
    def from_backbone(cls, config: MotionExpertConfig) -> "MotionExpertPipeline":
        backbone_type = str(config.backbone_type).lower()
        if backbone_type == "qwen":
            return cls.from_qwen(config)
        if backbone_type == "florence2":
            return cls.from_florence(config)
        raise ValueError(f"Unsupported MotionExpert backbone_type: {config.backbone_type}")

    @classmethod
    def _from_feature_extractor(
        cls,
        config: MotionExpertConfig,
        feature_extractor: torch.nn.Module,
    ) -> "MotionExpertPipeline":
        if hasattr(feature_extractor, "backbone") and hasattr(feature_extractor.backbone, "patch_size"):
            patch_size = int(feature_extractor.backbone.patch_size)
            spatial_merge_size = int(feature_extractor.backbone.spatial_merge_size)
            if config.input_height % patch_size != 0 or config.input_width % patch_size != 0:
                raise ValueError(
                    f"Canvas {(config.input_height, config.input_width)} must be divisible by "
                    f"the Qwen patch size {patch_size}."
                )
            if ((config.input_height // patch_size) % spatial_merge_size) != 0:
                raise ValueError("input_height is incompatible with Qwen spatial merge size.")
            if ((config.input_width // patch_size) % spatial_merge_size) != 0:
                raise ValueError("input_width is incompatible with Qwen spatial merge size.")

        visual_tokens = feature_extractor.estimate_visual_token_count(
            config.input_height,
            config.input_width,
        )
        max_conditioning_tokens = (
            config.num_query_tokens + visual_tokens + config.text_max_length
        )
        model = MotionExpertModel(
            config=config,
            visual_input_dim=feature_extractor.visual_token_dim,
            text_input_dim=feature_extractor.text_token_dim,
            max_conditioning_tokens=max_conditioning_tokens,
        )
        return cls(config=config, feature_extractor=feature_extractor, model=model)

    def set_query_xyz_statistics(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.model.set_query_xyz_statistics(center, scale)

    def set_delta_range(self, delta_range_xyz: torch.Tensor) -> None:
        self.model.set_delta_range(delta_range_xyz)

    def _extract_backbone_features(
        self,
        batch: MotionExpertBatch,
    ) -> Optional[MotionExpertFeatureOutput]:
        needs_extractor = batch.visual_tokens is None or batch.text_tokens is None
        if not needs_extractor:
            return None
        if self.feature_extractor is None:
            if batch.visual_tokens is None or batch.text_tokens is None:
                raise ValueError(
                    "MotionExpertPipeline needs either precomputed visual/text features "
                    "or a feature extractor. Use MotionExpertPipeline.from_backbone(...) "
                    "or pass visual_tokens/text_tokens in MotionExpertBatch."
                )
            return None
        if batch.rgb is None or batch.text_token_ids is None:
            raise ValueError("Backbone feature extraction requires images and text_token_ids.")
        return self.feature_extractor(
            batch.rgb,
            batch.text_token_ids,
            text_attention_mask=batch.text_attention_mask,
            text_valid_lens=batch.text_valid_lens,
        )

    def forward(self, batch: MotionExpertBatch) -> Dict[str, torch.Tensor]:
        query_uv = flatten_query_grid(batch.query_uv)
        query_world_xyz = flatten_query_grid(batch.query_world_xyz)
        future_delta_xyz = flatten_future_grid(batch.future_delta_xyz)
        traj_mask = flatten_future_grid(batch.traj_mask)

        extracted = self._extract_backbone_features(batch)
        visual_tokens = batch.visual_tokens
        text_tokens = batch.text_tokens

        if extracted is not None:
            if visual_tokens is None:
                visual_tokens = extracted.visual_tokens
            if text_tokens is None:
                text_tokens = extracted.text_tokens

        if visual_tokens is None or text_tokens is None:
            raise ValueError("visual_tokens and text_tokens are required for MotionExpert.")
        if future_delta_xyz is None or traj_mask is None:
            raise ValueError("Training requires future_delta_xyz and traj_mask.")

        future_delta_xyz_norm = self.model.normalize_future_delta(future_delta_xyz)
        trunk_conditioning = self.model.build_conditioning_tokens(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )
        outputs = self.model.forward_diffusion_training(
            trunk_conditioning=trunk_conditioning,
            target_delta_xyz_norm=future_delta_xyz_norm,
            diffusion_loss=self.loss_fn,
        )
        outputs.update(self.loss_fn(outputs, traj_mask))
        outputs["query_uv"] = query_uv
        outputs["query_world_xyz"] = query_world_xyz
        outputs["query_world_xyz_norm"] = self.model.normalize_query_xyz(query_world_xyz)
        outputs["future_delta_xyz"] = future_delta_xyz
        outputs["future_delta_xyz_norm"] = future_delta_xyz_norm
        outputs["traj_mask"] = traj_mask
        outputs["trunk_conditioning"] = trunk_conditioning
        return outputs

    @torch.no_grad()
    def predict(
        self,
        batch: MotionExpertBatch,
        scheduler=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        query_uv = flatten_query_grid(batch.query_uv)
        query_world_xyz = flatten_query_grid(batch.query_world_xyz)
        future_delta_xyz = flatten_future_grid(batch.future_delta_xyz)
        traj_mask = flatten_future_grid(batch.traj_mask)

        extracted = self._extract_backbone_features(batch)
        visual_tokens = batch.visual_tokens
        text_tokens = batch.text_tokens
        if extracted is not None:
            visual_tokens = extracted.visual_tokens if visual_tokens is None else visual_tokens
            text_tokens = extracted.text_tokens if text_tokens is None else text_tokens
        if visual_tokens is None or text_tokens is None:
            raise ValueError("visual_tokens and text_tokens are required for MotionExpert.")

        trunk_conditioning = self.model.build_conditioning_tokens(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )
        if scheduler is None:
            scheduler = build_default_inference_scheduler(
                num_train_timesteps=self.loss_fn.num_train_timesteps
            )
        pred_delta_xyz = self.model.predict(
            trunk_conditioning=trunk_conditioning,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        outputs = {"pred_delta_xyz": pred_delta_xyz}
        if future_delta_xyz is not None and traj_mask is not None:
            outputs.update(
                compute_motion_expert_metrics(
                    pred_delta_xyz=pred_delta_xyz,
                    target_delta_xyz=future_delta_xyz,
                    traj_mask=traj_mask,
                )
            )
        return outputs
