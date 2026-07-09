"""Standalone UVD MotionExpert.

A 3D point-trajectory rectified-flow model whose "3D" coordinate is
``(u_grid, v_grid, depth_norm)`` instead of unprojected world XYZ. It mirrors the
2D model (``model_2d.py``) but the geometry coordinate is 3-dimensional:

  * KNN graph, relative-position bias and the point-scalar projection all operate
    on the 3D UVD coordinate (``COORD_DIM = 3``).
  * Visual features are still grid-sampled with the 2D ``(u, v)`` query (the
    depth channel does not index the image), exactly like the 2D / 3D models.
  * The velocity head regresses a 3D delta ``(du, dv, dd)``.

Coord-agnostic building blocks are reused from ``model.py``; only the
dim-hardcoded pieces are reimplemented for ``COORD_DIM = 3``. The 2D and 3D
models are left untouched.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from modeling.utils.flash_attention import FlashMultiheadAttention

from .config import MotionExpertConfig
from .geometry import validate_query_uv_grid
from .model import (
    FeedForward,
    TokenProjector,
    _sinusoidal_embedding,
    build_knn_graph,
    gather_neighbors,
)

COORD_DIM = 3


class LocalGeometryAttentionLayerUVD(nn.Module):
    """KNN local attention with a 3D relative-position bias (rel_uvd + dist)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by {num_heads} heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rel_bias = nn.Sequential(
            nn.Linear(COORD_DIM + 1, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, coord: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        neigh_x = gather_neighbors(x_norm, neighbor_idx)
        neigh_coord = gather_neighbors(coord, neighbor_idx)
        rel_coord = neigh_coord - coord.unsqueeze(2)
        rel_dist = rel_coord.norm(dim=-1, keepdim=True)
        rel_feat = torch.cat([rel_coord, rel_dist], dim=-1).to(x_norm.dtype)

        q = self.q_proj(x_norm).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        k = self.k_proj(neigh_x).view(
            x.shape[0], x.shape[1], neighbor_idx.shape[-1], self.num_heads, self.head_dim
        )
        v = self.v_proj(neigh_x).view(
            x.shape[0], x.shape[1], neighbor_idx.shape[-1], self.num_heads, self.head_dim
        )
        logits = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.rel_bias(rel_feat)
        attn = torch.softmax(logits, dim=2)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        out = out.reshape(x.shape[0], x.shape[1], self.dim)
        x = residual + self.dropout(self.out_proj(out))
        return x + self.ffn(self.norm2(x))


class SparsePointEncoderUVD(nn.Module):
    def __init__(self, dim, num_layers, num_heads, mlp_ratio, dropout, checkpoint_blocks=True):
        super().__init__()
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.layers = nn.ModuleList(
            [LocalGeometryAttentionLayerUVD(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, coord, neighbor_idx):
        for layer in self.layers:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(layer, x, coord, neighbor_idx, use_reentrant=False)
            else:
                x = layer(x, coord, neighbor_idx)
        return x


class SparseDecoderBlockUVD(nn.Module):
    """Spatial (KNN over UVD) -> temporal -> conditioning cross-attn -> FFN."""

    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.spatial = LocalGeometryAttentionLayerUVD(dim, num_heads, mlp_ratio, dropout)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = FlashMultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
            flash_attn_name="MotionExpertUVD temporal attention",
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.cond_norm = nn.LayerNorm(dim)
        self.cond_attn = FlashMultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
            flash_attn_name="MotionExpertUVD condition attention",
        )
        self.cond_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, coord, neighbor_idx, cond_tokens, cond_key_padding_mask):
        batch_size, num_points, num_steps, dim = x.shape
        x_bt = x.permute(0, 2, 1, 3).reshape(batch_size * num_steps, num_points, dim)
        coord_bt = coord[:, None].expand(-1, num_steps, -1, -1).reshape(
            batch_size * num_steps, num_points, COORD_DIM
        )
        idx_bt = neighbor_idx[:, None].expand(-1, num_steps, -1, -1).reshape(
            batch_size * num_steps, num_points, neighbor_idx.shape[-1]
        )
        x_bt = self.spatial(x_bt, coord_bt, idx_bt)
        x = x_bt.reshape(batch_size, num_steps, num_points, dim).permute(0, 2, 1, 3)

        x_bn = x.reshape(batch_size * num_points, num_steps, dim)
        temporal_in = self.temporal_norm(x_bn)
        temporal_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in, need_weights=False)
        x_bn = x_bn + self.temporal_dropout(temporal_out)
        x = x_bn.reshape(batch_size, num_points, num_steps, dim)

        x_flat = x.reshape(batch_size, num_points * num_steps, dim)
        cond_in = self.cond_norm(x_flat)
        cond_out, _ = self.cond_attn(
            cond_in, cond_tokens, cond_tokens,
            key_padding_mask=cond_key_padding_mask, need_weights=False,
        )
        x_flat = x_flat + self.cond_dropout(cond_out)
        x_flat = x_flat + self.ffn(self.ffn_norm(x_flat))
        return x_flat.reshape(batch_size, num_points, num_steps, dim)


class SparseFlowDecoderUVD(nn.Module):
    def __init__(self, dim, future_steps, num_layers, num_heads, mlp_ratio, dropout,
                 checkpoint_blocks=True):
        super().__init__()
        self.future_steps = int(future_steps)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.noisy_delta_proj = nn.Linear(COORD_DIM, dim)
        self.step_embed = nn.Embedding(self.future_steps, dim)
        self.tau_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [SparseDecoderBlockUVD(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.velocity_head = nn.Linear(dim, COORD_DIM)

    def forward(self, x_tau, tau, point_context, coord, neighbor_idx, cond_tokens,
                cond_key_padding_mask, return_hidden: bool = False):
        batch_size, num_points, num_steps, _ = x_tau.shape
        if num_steps != self.future_steps:
            raise ValueError(f"Expected {self.future_steps} future steps, got {num_steps}.")
        step_ids = torch.arange(num_steps, device=x_tau.device)
        step_embed = self.step_embed(step_ids)[None, None, :, :]
        tau_embed = self.tau_mlp(_sinusoidal_embedding(tau.to(x_tau.dtype), point_context.shape[-1]))
        x = (
            self.noisy_delta_proj(x_tau)
            + point_context[:, :, None, :]
            + step_embed.to(x_tau.dtype)
            + tau_embed[:, None, None, :].to(x_tau.dtype)
        )
        for block in self.blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(block, x, coord, neighbor_idx, cond_tokens, cond_key_padding_mask,
                               use_reentrant=False)
            else:
                x = block(x, coord, neighbor_idx, cond_tokens, cond_key_padding_mask)
        hidden = self.out_norm(x)
        velocity = self.velocity_head(hidden)
        if return_hidden:
            return velocity, hidden
        return velocity


class MotionExpertUVDModel(nn.Module):
    """UVD point-trajectory rectified-flow model (florence2 features + UVD queries).

    The model consumes ``query_uv`` (B, P, 2) for visual sampling and
    ``query_uvd`` (B, P, 3) for geometry. Targets are anchor-relative UVD deltas.
    """

    coord_dim = COORD_DIM

    def __init__(self, config: MotionExpertConfig, visual_input_dim: Optional[int] = None,
                 text_input_dim: Optional[int] = None, max_conditioning_tokens: Optional[int] = None):
        super().__init__()
        if config.future_steps != 15:
            raise ValueError("MotionExpertUVD is locked to future_steps=15.")
        self.config = config
        sparse_cfg = config.sparse_decoder
        self.token_dim = sparse_cfg.hidden_dim
        self.max_conditioning_tokens = max_conditioning_tokens
        self.visual_global_tokens = int(sparse_cfg.visual_global_tokens)
        if self.visual_global_tokens <= 0:
            raise ValueError("sparse_decoder.visual_global_tokens must be positive.")
        self.knn_k = int(config.knn_k)

        self.point_scalar_proj = nn.Sequential(
            nn.Linear(COORD_DIM, self.token_dim), nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.visual_proj = TokenProjector(self.token_dim, visual_input_dim)
        self.text_proj = TokenProjector(self.token_dim, text_input_dim)
        self.point_norm = nn.LayerNorm(self.token_dim)
        self.point_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.global_visual_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.point_encoder = SparsePointEncoderUVD(
            dim=self.token_dim, num_layers=sparse_cfg.point_encoder_layers,
            num_heads=sparse_cfg.num_attention_heads, mlp_ratio=sparse_cfg.mlp_ratio,
            dropout=config.dropout, checkpoint_blocks=config.checkpoint_activations,
        )
        self.flow_decoder = SparseFlowDecoderUVD(
            dim=self.token_dim, future_steps=config.future_steps,
            num_layers=sparse_cfg.decoder_layers, num_heads=sparse_cfg.num_attention_heads,
            mlp_ratio=sparse_cfg.mlp_ratio, dropout=config.dropout,
            checkpoint_blocks=config.checkpoint_activations,
        )
        # UVD-delta normalization scale: [du(grid), dv(grid), dd(metres)].
        self.register_buffer(
            "traj_delta_uvd_scale",
            torch.tensor([1.0] * COORD_DIM, dtype=torch.float32),
            persistent=True,
        )

    def set_traj_delta_scale(self, scale) -> None:
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.traj_delta_uvd_scale.device)
        self.traj_delta_uvd_scale.copy_(torch.clamp(scale, min=1e-4))

    def normalize_future_delta(self, delta_uvd: torch.Tensor) -> torch.Tensor:
        return delta_uvd / self.traj_delta_uvd_scale.to(delta_uvd.dtype)

    def unnormalize_future_delta(self, delta_uvd_norm: torch.Tensor) -> torch.Tensor:
        return delta_uvd_norm * self.traj_delta_uvd_scale.to(delta_uvd_norm.dtype)

    # ---- visual feature helpers (mirror MotionExpert2DModel; uses 2D uv) ----
    def _infer_visual_grid(self, visual_tokens):
        if visual_tokens is None:
            return None
        batch_size, num_tokens, channels = visual_tokens.shape
        patch_size = 32 if str(self.config.backbone_type).lower() == "florence2" else 16
        h = math.ceil(int(self.config.input_height) / float(patch_size))
        w = math.ceil(int(self.config.input_width) / float(patch_size))
        if h * w == num_tokens:
            return visual_tokens.reshape(batch_size, h, w, channels)
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            return visual_tokens.reshape(batch_size, side, side, channels)
        return None

    def _visual_pool_hw(self, num_tokens):
        pool_h = max(1, int(math.floor(math.sqrt(num_tokens))))
        pool_w = int(math.ceil(num_tokens / float(pool_h)))
        return pool_h, pool_w

    def _pool_visual_conditioning_tokens(self, visual_tokens, visual_grid=None):
        if self.visual_global_tokens == 1:
            return visual_tokens.mean(dim=1, keepdim=True)
        pool_h, pool_w = self._visual_pool_hw(self.visual_global_tokens)
        if visual_grid is not None and visual_grid.dim() in (4, 5):
            grid = visual_grid
            if grid.dim() == 5:
                b, nc, hh, ww, ch = grid.shape
                grid = grid.reshape(b, nc * hh, ww, ch)
            grid_nchw = grid.permute(0, 3, 1, 2).contiguous()
            pooled = F.adaptive_avg_pool2d(grid_nchw.float(), (pool_h, pool_w))
            pooled = pooled.flatten(2).transpose(1, 2)
            return pooled[:, : self.visual_global_tokens].to(dtype=visual_tokens.dtype)
        pooled = F.adaptive_avg_pool1d(visual_tokens.transpose(1, 2).float(), self.visual_global_tokens)
        return pooled.transpose(1, 2).to(dtype=visual_tokens.dtype)

    def sample_visual_features(self, query_uv, visual_tokens, visual_grid=None):
        query_uv = validate_query_uv_grid(query_uv)
        if visual_grid is None:
            visual_grid = self._infer_visual_grid(visual_tokens)
        global_visual = self._pool_visual_conditioning_tokens(visual_tokens, visual_grid)
        sampling_grid = visual_grid
        if sampling_grid is not None:
            if sampling_grid.dim() == 5:
                sampling_grid = sampling_grid[:, 0]
            if sampling_grid.dim() != 4:
                raise ValueError(f"visual_grid must be [B,H,W,C], got {tuple(sampling_grid.shape)}.")
            grid_nchw = sampling_grid.permute(0, 3, 1, 2).contiguous()
            sample_grid = query_uv.float().unsqueeze(2)
            sampled = F.grid_sample(grid_nchw.float(), sample_grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            sampled = sampled.squeeze(-1).transpose(1, 2).to(visual_tokens.dtype)
        else:
            sampled = visual_tokens.mean(dim=1, keepdim=True).expand(-1, query_uv.shape[1], -1)
        return sampled, global_visual

    def _build_conditioning_key_padding_mask(self, text_tokens, text_attention_mask, global_token_count):
        batch_size, text_token_count, _ = text_tokens.shape
        global_mask = torch.zeros(batch_size, int(global_token_count), dtype=torch.bool,
                                  device=text_tokens.device)
        if text_attention_mask is None:
            text_mask = torch.zeros(batch_size, text_token_count, dtype=torch.bool, device=text_tokens.device)
        else:
            text_mask = text_attention_mask.to(device=text_tokens.device) <= 0
        return torch.cat([global_mask, text_mask], dim=1)

    def build_point_context(self, query_uv, query_uvd, visual_tokens, text_tokens,
                            text_attention_mask=None, visual_grid=None):
        query_uv = validate_query_uv_grid(query_uv)
        query_uvd = query_uvd.float()
        sampled_visual, global_visual = self.sample_visual_features(query_uv, visual_tokens, visual_grid)
        point_tokens = self.point_scalar_proj(query_uvd) + self.visual_proj(sampled_visual)
        point_tokens = self.point_norm(self.dropout(point_tokens)) + self.point_type_embed
        neighbor_idx = build_knn_graph(query_uvd, self.knn_k)
        point_context = self.point_encoder(point_tokens, query_uvd, neighbor_idx)

        global_visual_tokens = self.visual_proj(global_visual) + self.global_visual_type_embed
        text_cond_tokens = self.text_proj(text_tokens) + self.text_type_embed
        cond_tokens = torch.cat([global_visual_tokens, text_cond_tokens], dim=1)
        cond_key_padding_mask = self._build_conditioning_key_padding_mask(
            text_tokens, text_attention_mask, global_visual_tokens.shape[1]
        )
        return point_context, neighbor_idx, cond_tokens, cond_key_padding_mask

    def forward_diffusion_training(self, query_uv, query_uvd, visual_tokens, text_tokens,
                                   target_delta_uvd_norm, text_attention_mask=None, visual_grid=None):
        query_uvd = query_uvd.float()
        batch_size = target_delta_uvd_norm.shape[0]
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv, query_uvd=query_uvd, visual_tokens=visual_tokens, text_tokens=text_tokens,
            text_attention_mask=text_attention_mask, visual_grid=visual_grid,
        )
        noise = torch.randn_like(target_delta_uvd_norm)
        tau = torch.rand(batch_size, device=target_delta_uvd_norm.device, dtype=target_delta_uvd_norm.dtype)
        tau_view = tau[:, None, None, None]
        x_tau = (1.0 - tau_view) * noise + tau_view * target_delta_uvd_norm
        target_velocity = target_delta_uvd_norm - noise
        pred_velocity = self.flow_decoder(
            x_tau=x_tau, tau=tau, point_context=point_context, coord=query_uvd,
            neighbor_idx=neighbor_idx, cond_tokens=cond_tokens,
            cond_key_padding_mask=cond_key_padding_mask,
        )
        return {
            "pred_velocity": pred_velocity,
            "target_velocity": target_velocity,
            "noisy_delta_uvd_norm": x_tau,
            "tau": tau,
            "point_context": point_context,
        }

    @torch.no_grad()
    def predict(self, query_uv, query_uvd, visual_tokens, text_tokens, text_attention_mask=None,
                visual_grid=None, num_steps: int = 10):
        """Euler-integrate the rectified flow noise -> normalized UVD delta -> UVD delta."""
        query_uvd = query_uvd.float()
        batch_size, num_points, _ = query_uvd.shape
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv, query_uvd=query_uvd, visual_tokens=visual_tokens, text_tokens=text_tokens,
            text_attention_mask=text_attention_mask, visual_grid=visual_grid,
        )
        x = torch.randn(batch_size, num_points, self.flow_decoder.future_steps, COORD_DIM,
                        device=query_uvd.device, dtype=point_context.dtype)
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            tau = torch.full((batch_size,), step * dt, device=query_uvd.device, dtype=x.dtype)
            v = self.flow_decoder(
                x_tau=x, tau=tau, point_context=point_context, coord=query_uvd,
                neighbor_idx=neighbor_idx, cond_tokens=cond_tokens,
                cond_key_padding_mask=cond_key_padding_mask,
            )
            x = x + dt * v
        return self.unnormalize_future_delta(x)

    def extract_one_ode_step_latent(
        self,
        query_uv: torch.Tensor,
        query_uvd: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        visual_grid: Optional[torch.Tensor] = None,
        solver_steps: int = 10,
        steps_to_run: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Decoder hidden at the noise point (tau=0), no integration.

        Mirrors ``MotionExpertModel.extract_one_ode_step_latent`` (model.py) but in
        UVD geometry: the geometry coordinate is ``query_uvd = (u_grid, v_grid,
        depth_standardized)`` instead of world XYZ. Returns the per-point,
        per-future-step decoder hidden that a downstream policy pools into a motion
        latent, and ``motion_pos`` = the (sanitized) UVD query coordinate.
        """
        del solver_steps
        if int(steps_to_run) != 1:
            raise ValueError(
                "Motion latent extraction currently supports exactly one ODE "
                f"solver step, got steps_to_run={steps_to_run}."
            )
        query_uvd = query_uvd.float()
        motion_pos = torch.nan_to_num(query_uvd, nan=0.0, posinf=0.0, neginf=0.0)
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv,
            query_uvd=query_uvd,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            visual_grid=visual_grid,
        )
        batch_size, num_points, _ = query_uvd.shape
        x = torch.randn(
            batch_size,
            num_points,
            self.flow_decoder.future_steps,
            COORD_DIM,
            device=query_uvd.device,
            dtype=point_context.dtype,
            generator=generator,
        )
        tau = torch.zeros(batch_size, device=query_uvd.device, dtype=x.dtype)
        _, motion_hidden = self.flow_decoder(
            x_tau=x,
            tau=tau,
            point_context=point_context,
            coord=query_uvd,
            neighbor_idx=neighbor_idx,
            cond_tokens=cond_tokens,
            cond_key_padding_mask=cond_key_padding_mask,
            return_hidden=True,
        )
        return {
            "motion_hidden": motion_hidden,
            "motion_pos": motion_pos,
        }
