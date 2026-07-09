"""UVD MotionExpert pipeline: florence2 features + UVD rectified-flow point model."""
from typing import Dict, Optional

import torch
from torch import nn

from .config import MotionExpertConfig
from .data import MotionExpertBatch
from .extractor import FlorenceTraceFeatureExtractor
from .model_uvd import MotionExpertUVDModel
from .pipeline import MotionExpertFlowLoss


class MotionExpertUVDPipeline(nn.Module):
    def __init__(self, config: MotionExpertConfig, feature_extractor: nn.Module,
                 model: MotionExpertUVDModel, loss_fn: Optional[MotionExpertFlowLoss] = None):
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.model = model
        self.loss_fn = loss_fn or MotionExpertFlowLoss(
            num_train_timesteps=config.flow_timesteps,
            use_motion_weighted_loss=config.use_motion_weighted_loss,
            motion_weight_tau=config.motion_weight_tau,
            motion_weight_temp=config.motion_weight_temp,
            motion_weight_gamma=config.motion_weight_gamma,
            motion_weight_static_floor=config.motion_weight_static_floor,
        )

    @classmethod
    def from_backbone(cls, config: MotionExpertConfig) -> "MotionExpertUVDPipeline":
        if str(config.backbone_type).lower() != "florence2":
            raise ValueError("MotionExpertUVDPipeline currently supports backbone_type='florence2'.")
        feature_extractor = FlorenceTraceFeatureExtractor(config.florence)
        max_cond = (
            int(config.sparse_decoder.visual_global_tokens) + int(config.text_max_length)
        )
        model = MotionExpertUVDModel(
            config=config,
            visual_input_dim=feature_extractor.visual_token_dim,
            text_input_dim=feature_extractor.text_token_dim,
            max_conditioning_tokens=max_cond,
        )
        return cls(config=config, feature_extractor=feature_extractor, model=model)

    def _extract(self, batch: MotionExpertBatch):
        if batch.visual_tokens is not None and batch.text_tokens is not None:
            return batch.visual_tokens, batch.text_tokens, None
        if batch.rgb is None or batch.text_token_ids is None:
            raise ValueError("MotionExpertUVD needs rgb + text_token_ids (or precomputed features).")
        out = self.feature_extractor(
            batch.rgb, batch.text_token_ids,
            text_attention_mask=batch.text_attention_mask,
            text_valid_lens=batch.text_valid_lens,
        )
        return out.visual_tokens, out.text_tokens, out.visual_grid

    def forward(self, batch: MotionExpertBatch) -> Dict[str, torch.Tensor]:
        # query_uv: (B,P,2) for visual sampling; query_world_xyz carries the (B,P,3) UVD
        # geometry coord; future_delta_xyz carries the (B,P,T,3) UVD delta.
        if (batch.query_uv is None or batch.query_world_xyz is None
                or batch.future_delta_xyz is None or batch.traj_mask is None):
            raise ValueError("Training requires query_uv, query_world_xyz (UVD), future_delta_xyz (UVD) and traj_mask.")
        query_uv = batch.query_uv
        query_uvd = batch.query_world_xyz
        future_delta_uvd = batch.future_delta_xyz
        traj_mask = batch.traj_mask
        visual_tokens, text_tokens, visual_grid = self._extract(batch)

        future_delta_norm = self.model.normalize_future_delta(future_delta_uvd)
        outputs = self.model.forward_diffusion_training(
            query_uv=query_uv, query_uvd=query_uvd, visual_tokens=visual_tokens, text_tokens=text_tokens,
            target_delta_uvd_norm=future_delta_norm,
            text_attention_mask=batch.text_attention_mask, visual_grid=visual_grid,
        )
        outputs["future_delta_uvd"] = future_delta_uvd
        outputs["traj_mask"] = traj_mask
        outputs.update(self.loss_fn(outputs, traj_mask, future_delta_xyz=future_delta_uvd))
        return outputs

    @torch.no_grad()
    def predict(self, batch: MotionExpertBatch, num_steps: int = 10) -> torch.Tensor:
        """Return anchor-relative UVD deltas (B,P,T,3): du,dv in grid units, dd in metres."""
        visual_tokens, text_tokens, visual_grid = self._extract(batch)
        return self.model.predict(
            query_uv=batch.query_uv, query_uvd=batch.query_world_xyz,
            visual_tokens=visual_tokens, text_tokens=text_tokens,
            text_attention_mask=batch.text_attention_mask, visual_grid=visual_grid,
            num_steps=num_steps,
        )
