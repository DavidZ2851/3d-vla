from typing import Dict, Optional

import torch
from torch import nn

from .config import MotionExpertConfig
from .data import MotionExpertBatch, flatten_future_grid, flatten_query_grid
from .extractor import (
    FlorenceTraceFeatureExtractor,
    MotionExpertFeatureOutput,
    QwenTraceFeatureExtractor,
)
from .geometry import center_query_world_xyz, query_uv_depth_to_world_xyz, sample_depth_at_query_uv
from .model import MotionExpertModel


class MotionExpertFlowLoss(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 100,
        use_motion_weighted_loss: bool = True,
        motion_weight_tau: float = 0.005,
        motion_weight_temp: float = 5.0,
        motion_weight_gamma: float = 1.0,
        motion_weight_static_floor: float = 0.05,
    ):
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.use_motion_weighted_loss = bool(use_motion_weighted_loss)
        self.motion_weight_tau = float(motion_weight_tau)
        self.motion_weight_temp = float(motion_weight_temp)
        self.motion_weight_gamma = float(motion_weight_gamma)
        self.motion_weight_static_floor = float(motion_weight_static_floor)

    def _velocity_tensors(self, predictions: Dict[str, torch.Tensor]):
        if "pred_velocity" in predictions and "target_velocity" in predictions:
            return predictions["pred_velocity"], predictions["target_velocity"]
        if "noise_pred" in predictions and "noise_target" in predictions:
            noise_pred = predictions["noise_pred"]
            noise_target = predictions["noise_target"]
            if noise_pred.dim() == 5:
                noise_pred = noise_pred.permute(0, 3, 4, 2, 1).flatten(1, 2)
                noise_target = noise_target.permute(0, 3, 4, 2, 1).flatten(1, 2)
            return noise_pred, noise_target
        raise KeyError("Predictions must contain pred_velocity/target_velocity.")

    def _loss_weight(
        self,
        traj_mask: torch.Tensor,
        future_delta_xyz: Optional[torch.Tensor],
    ) -> torch.Tensor:
        weights = traj_mask.to(dtype=torch.float32)
        if not self.use_motion_weighted_loss or future_delta_xyz is None:
            return weights
        delta_norm = torch.linalg.norm(future_delta_xyz.float(), dim=-1)
        tau = max(self.motion_weight_tau, 1e-12)
        motion_score = torch.sigmoid((delta_norm - tau) * (self.motion_weight_temp / tau))
        motion_weight = motion_score.pow(self.motion_weight_gamma)
        motion_weight = torch.clamp(motion_weight, min=self.motion_weight_static_floor)
        return weights * motion_weight.to(weights.dtype)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        traj_mask: torch.Tensor,
        future_delta_xyz: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_velocity, target_velocity = self._velocity_tensors(predictions)
        loss_weight = self._loss_weight(traj_mask, future_delta_xyz).to(pred_velocity.device)
        loss_weight = loss_weight.to(pred_velocity.dtype)
        diff_sq = (pred_velocity - target_velocity).pow(2) * loss_weight.unsqueeze(-1)
        valid_count = loss_weight.sum().clamp_min(1.0) * pred_velocity.shape[-1]
        flow_loss = diff_sq.sum() / valid_count
        return {
            "loss": flow_loss,
            "flow_loss": flow_loss,
            "diffusion_loss": flow_loss,
            "loss_weight": loss_weight,
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
    delta_epe = torch.linalg.norm(diff, dim=-1)
    valid_delta = traj_mask.bool()
    if valid_delta.any():
        delta_acc_001 = (delta_epe[valid_delta] < 0.01).float().mean()
    else:
        delta_acc_001 = pred_delta_xyz.new_tensor(0.0)

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
        endpoint_err = torch.linalg.norm(pred_endpoint - target_endpoint, dim=-1)
        endpoint_acc_001 = (endpoint_err[endpoint_valid] < 0.01).float().mean()
    else:
        endpoint_l1 = pred_delta_xyz.new_tensor(0.0)
        endpoint_acc_001 = pred_delta_xyz.new_tensor(0.0)

    return {
        "delta_l1": delta_l1,
        "delta_mse": delta_mse,
        "delta_acc_001": delta_acc_001,
        "endpoint_l1": endpoint_l1,
        "endpoint_acc_001": endpoint_acc_001,
        **_compute_moved_static_metrics(pred_delta_xyz, target_delta_xyz, traj_mask),
    }


def _compute_moved_static_metrics(
    pred_delta_xyz: torch.Tensor,
    target_delta_xyz: torch.Tensor,
    traj_mask: torch.Tensor,
    tau: float = 0.005,
) -> Dict[str, torch.Tensor]:
    l1 = (pred_delta_xyz - target_delta_xyz).abs().mean(dim=-1)
    valid = traj_mask.bool()
    moved = torch.linalg.norm(target_delta_xyz, dim=-1) > tau
    static = ~moved

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        mask = mask & valid
        if mask.any():
            return l1[mask].mean()
        return pred_delta_xyz.new_tensor(0.0)

    return {
        "moved_delta_l1": _masked_mean(moved),
        "static_delta_l1": _masked_mean(static),
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
        self.loss_fn = loss_fn or MotionExpertFlowLoss(
            num_train_timesteps=config.flow_timesteps,
            use_motion_weighted_loss=config.use_motion_weighted_loss,
            motion_weight_tau=config.motion_weight_tau,
            motion_weight_temp=config.motion_weight_temp,
            motion_weight_gamma=config.motion_weight_gamma,
            motion_weight_static_floor=config.motion_weight_static_floor,
        )

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

        max_conditioning_tokens = config.num_query_tokens + 1 + config.text_max_length
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

    def set_calvin_normalization_statistics(
        self,
        query_center: torch.Tensor,
        query_scale: torch.Tensor,
        traj_center: torch.Tensor,
        traj_scale: torch.Tensor,
    ) -> None:
        self.model.set_calvin_normalization_statistics(
            query_center=query_center,
            query_scale=query_scale,
            traj_center=traj_center,
            traj_scale=traj_scale,
        )

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

    def _resolve_query_world_xyz_and_depth(
        self,
        batch: MotionExpertBatch,
        query_uv: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        query_world_xyz = flatten_query_grid(batch.query_world_xyz)
        query_depth = None
        if query_world_xyz is not None:
            query_world_xyz = center_query_world_xyz(query_world_xyz)
            if batch.depth is not None:
                query_depth = sample_depth_at_query_uv(batch.depth, query_uv).unsqueeze(-1)
            return query_world_xyz, query_depth
        if batch.depth is None or batch.intrinsics is None or batch.extrinsics is None:
            raise ValueError(
                "MotionExpert requires query_world_xyz, or depth + intrinsics + extrinsics "
                "to reconstruct query points."
            )
        query_world_xyz, query_depth = query_uv_depth_to_world_xyz(
            query_uv=query_uv,
            depth=batch.depth,
            intrinsics=batch.intrinsics,
            extrinsics=batch.extrinsics,
        )
        return center_query_world_xyz(query_world_xyz), query_depth

    def forward(self, batch: MotionExpertBatch) -> Dict[str, torch.Tensor]:
        query_uv = flatten_query_grid(batch.query_uv)
        future_delta_xyz = flatten_future_grid(batch.future_delta_xyz)
        traj_mask = flatten_future_grid(batch.traj_mask)
        if query_uv is None:
            raise ValueError("query_uv is required for MotionExpert.")
        query_world_xyz, query_depth = self._resolve_query_world_xyz_and_depth(batch, query_uv)

        extracted = self._extract_backbone_features(batch)
        visual_tokens = batch.visual_tokens
        text_tokens = batch.text_tokens
        visual_grid = None

        if extracted is not None:
            if visual_tokens is None:
                visual_tokens = extracted.visual_tokens
            if text_tokens is None:
                text_tokens = extracted.text_tokens
            visual_grid = extracted.visual_grid

        if visual_tokens is None or text_tokens is None:
            raise ValueError("visual_tokens and text_tokens are required for MotionExpert.")
        if future_delta_xyz is None or traj_mask is None:
            raise ValueError("Training requires future_delta_xyz and traj_mask.")

        future_delta_xyz_norm = self.model.normalize_future_delta(future_delta_xyz)
        outputs = self.model.forward_diffusion_training(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            target_delta_xyz_norm=future_delta_xyz_norm,
            visual_grid=visual_grid,
            query_depth=query_depth,
        )
        outputs["future_delta_xyz"] = future_delta_xyz
        outputs.update(
            self.loss_fn(
                outputs,
                traj_mask,
                future_delta_xyz=future_delta_xyz,
            )
        )
        outputs["query_uv"] = query_uv
        outputs["query_world_xyz"] = query_world_xyz
        outputs["query_world_xyz_norm"] = self.model.normalize_query_xyz(query_world_xyz)
        outputs["future_delta_xyz"] = future_delta_xyz
        outputs["future_delta_xyz_norm"] = future_delta_xyz_norm
        outputs["traj_mask"] = traj_mask
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
        future_delta_xyz = flatten_future_grid(batch.future_delta_xyz)
        traj_mask = flatten_future_grid(batch.traj_mask)
        if query_uv is None:
            raise ValueError("query_uv is required for MotionExpert.")
        query_world_xyz, query_depth = self._resolve_query_world_xyz_and_depth(batch, query_uv)

        extracted = self._extract_backbone_features(batch)
        visual_tokens = batch.visual_tokens
        text_tokens = batch.text_tokens
        visual_grid = None
        if extracted is not None:
            visual_tokens = extracted.visual_tokens if visual_tokens is None else visual_tokens
            text_tokens = extracted.text_tokens if text_tokens is None else text_tokens
            visual_grid = extracted.visual_grid
        if visual_tokens is None or text_tokens is None:
            raise ValueError("visual_tokens and text_tokens are required for MotionExpert.")

        del scheduler, guidance_scale
        pred_delta_xyz = self.model.predict(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            visual_grid=visual_grid,
            query_depth=query_depth,
            num_inference_steps=num_inference_steps,
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
