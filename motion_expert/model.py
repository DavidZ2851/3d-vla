from typing import Optional

import torch
from timm.models.vision_transformer import RmsNorm
from torch import nn

from .config import MotionExpertConfig
from .tracegen_compat import load_cogvideox_decoder_cls


class TokenProjector(nn.Module):
    def __init__(self, output_dim: int, input_dim: Optional[int] = None):
        super().__init__()
        if input_dim is None:
            self.norm = None
            self.proj = nn.LazyLinear(output_dim, bias=False)
            self.out_norm = nn.LayerNorm(output_dim)
        else:
            self.norm = RmsNorm(input_dim)
            self.proj = nn.Linear(input_dim, output_dim, bias=False)
            self.out_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        x = self.proj(x)
        return self.out_norm(x)


class MotionExpertModel(nn.Module):
    def __init__(
        self,
        config: MotionExpertConfig,
        visual_input_dim: Optional[int] = None,
        text_input_dim: Optional[int] = None,
        max_conditioning_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.token_dim = config.decoder.latent_dim

        self.query_uv_proj = nn.Sequential(
            nn.Linear(2, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.query_xyz_proj = nn.Sequential(
            nn.Linear(3, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.visual_proj = TokenProjector(self.token_dim, visual_input_dim)
        self.text_proj = TokenProjector(self.token_dim, text_input_dim)
        self.query_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.visual_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "query_xyz_center",
            torch.tensor(config.query_xyz_center, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "query_xyz_scale",
            torch.tensor(config.query_xyz_scale, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "delta_range_xyz",
            torch.tensor(config.delta_range_xyz, dtype=torch.float32),
            persistent=True,
        )

        decoder_cls = load_cogvideox_decoder_cls()
        decoder_cfg = config.decoder
        self.max_conditioning_tokens = (
            decoder_cfg.max_text_seq_length
            if max_conditioning_tokens is None
            else int(max_conditioning_tokens)
        )
        self.diffusion_decoder = decoder_cls(
            latent_dim=decoder_cfg.latent_dim,
            num_attention_heads=decoder_cfg.num_attention_heads,
            attention_head_dim=decoder_cfg.attention_head_dim,
            in_channels=decoder_cfg.in_channels,
            out_channels=decoder_cfg.out_channels,
            num_layers=decoder_cfg.num_layers,
            num_frames=config.future_steps,
            frame_size=config.grid_size,
            patch_size=decoder_cfg.patch_size,
            patch_size_t=decoder_cfg.patch_size_t,
            max_text_seq_length=self.max_conditioning_tokens,
            text_embed_dim=decoder_cfg.text_embed_dim,
            use_rotary_positional_embeddings=decoder_cfg.use_rotary_positional_embeddings,
            enable_encoder_hidden_states_grad=decoder_cfg.enable_encoder_hidden_states_grad,
            torch_dtype=decoder_cfg.torch_dtype,
            scale_factor=decoder_cfg.scale_factor,
            scale_factor_spatial=decoder_cfg.scale_factor_spatial,
            scale_factor_temporal=decoder_cfg.scale_factor_temporal,
            put_frames_in_channels=decoder_cfg.put_frames_in_channels,
        )
        self.set_delta_range(self.delta_range_xyz)

    def set_query_xyz_statistics(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=torch.float32, device=self.query_xyz_center.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.query_xyz_scale.device)
        self.query_xyz_center.copy_(center)
        self.query_xyz_scale.copy_(torch.clamp(scale, min=1e-6))
        self.config.query_xyz_center = tuple(float(x) for x in self.query_xyz_center.tolist())
        self.config.query_xyz_scale = tuple(float(x) for x in self.query_xyz_scale.tolist())

    def set_delta_range(self, delta_range: torch.Tensor) -> None:
        delta_range = torch.as_tensor(
            delta_range,
            dtype=torch.float32,
            device=self.delta_range_xyz.device,
        )
        delta_range = torch.clamp(delta_range, min=1e-6)
        self.delta_range_xyz.copy_(delta_range)
        self.diffusion_decoder.set_data_act_statistics(delta_range, -delta_range)
        self.config.delta_range_xyz = tuple(float(x) for x in self.delta_range_xyz.tolist())

    def normalize_query_xyz(self, query_world_xyz: torch.Tensor) -> torch.Tensor:
        return (query_world_xyz - self.query_xyz_center) / self.query_xyz_scale

    def normalize_future_delta(self, future_delta_xyz: torch.Tensor) -> torch.Tensor:
        return torch.clamp(future_delta_xyz / self.delta_range_xyz, -1.0, 1.0)

    def unnormalize_future_delta(self, future_delta_xyz_norm: torch.Tensor) -> torch.Tensor:
        return future_delta_xyz_norm * self.delta_range_xyz

    def build_query_tokens(
        self,
        query_uv: torch.Tensor,
        query_world_xyz_norm: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = self.query_uv_proj(query_uv) + self.query_xyz_proj(query_world_xyz_norm)
        return self.query_norm(self.dropout(query_tokens)) + self.query_type_embed

    def build_conditioning_tokens(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query_world_xyz_norm = self.normalize_query_xyz(query_world_xyz)
        query_tokens = self.build_query_tokens(query_uv, query_world_xyz_norm)
        visual_tokens = self.visual_proj(visual_tokens) + self.visual_type_embed
        text_tokens = self.text_proj(text_tokens) + self.text_type_embed
        return torch.cat([query_tokens, visual_tokens, text_tokens], dim=1)

    def build_decoder_target(self, future_delta_xyz_norm: torch.Tensor) -> torch.Tensor:
        anchor = torch.zeros(
            future_delta_xyz_norm.shape[0],
            future_delta_xyz_norm.shape[1],
            1,
            future_delta_xyz_norm.shape[-1],
            device=future_delta_xyz_norm.device,
            dtype=future_delta_xyz_norm.dtype,
        )
        return torch.cat([anchor, future_delta_xyz_norm], dim=2)

    def forward_diffusion_training(
        self,
        trunk_conditioning: torch.Tensor,
        target_delta_xyz_norm: torch.Tensor,
        diffusion_loss,
    ):
        decoder_target = self.build_decoder_target(target_delta_xyz_norm)
        return self.diffusion_decoder.forward_diffusion_training(
            trunk_conditioning=trunk_conditioning,
            target_video=decoder_target,
            diffusion_loss=diffusion_loss,
        )

    def predict(
        self,
        trunk_conditioning: torch.Tensor,
        scheduler,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self.diffusion_decoder.predict_trajectory(
            trunk_conditioning=trunk_conditioning,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
