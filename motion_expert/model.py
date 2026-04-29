import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm
from torch import nn

from .config import MotionExpertConfig
from .geometry import center_query_world_xyz


NORMALIZED_CLAMP_ABS = 2.0


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


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t[:, None]
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def build_knn_graph(xyz: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    if num_points <= 1:
        return torch.zeros(batch_size, num_points, 1, dtype=torch.long, device=xyz.device)
    k_eff = min(int(k), num_points - 1)
    xyz = torch.nan_to_num(xyz.float(), nan=0.0, posinf=0.0, neginf=0.0)
    dist = torch.cdist(xyz, xyz)
    eye = torch.eye(num_points, dtype=torch.bool, device=xyz.device).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k_eff, dim=-1, largest=False).indices


def gather_neighbors(x: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = x.shape[:2]
    trailing_shape = x.shape[2:]
    k = neighbor_idx.shape[-1]
    x_flat = x.reshape(batch_size, num_points, -1)
    gather_idx = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, x_flat.shape[-1])
    gathered = torch.gather(
        x_flat[:, None, :, :].expand(-1, num_points, -1, -1),
        dim=2,
        index=gather_idx,
    )
    return gathered.reshape(batch_size, num_points, k, *trailing_shape)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalGeometryAttentionLayer(nn.Module):
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
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        neigh_x = gather_neighbors(x_norm, neighbor_idx)
        neigh_xyz = gather_neighbors(xyz, neighbor_idx)
        rel_xyz = neigh_xyz - xyz.unsqueeze(2)
        rel_dist = rel_xyz.norm(dim=-1, keepdim=True)
        rel_feat = torch.cat([rel_xyz, rel_dist], dim=-1).to(x_norm.dtype)

        q = self.q_proj(x_norm).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        k = self.k_proj(neigh_x).view(
            x.shape[0],
            x.shape[1],
            neighbor_idx.shape[-1],
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(neigh_x).view(
            x.shape[0],
            x.shape[1],
            neighbor_idx.shape[-1],
            self.num_heads,
            self.head_dim,
        )
        logits = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.rel_bias(rel_feat)
        attn = torch.softmax(logits, dim=2)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        out = out.reshape(x.shape[0], x.shape[1], self.dim)
        x = residual + self.dropout(self.out_proj(out))
        return x + self.ffn(self.norm2(x))


class SparsePointEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LocalGeometryAttentionLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, xyz, neighbor_idx)
        return x


class SparseDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.spatial = LocalGeometryAttentionLayer(dim, num_heads, mlp_ratio, dropout)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.cond_norm = nn.LayerNorm(dim)
        self.cond_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cond_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_points, num_steps, dim = x.shape
        x_bt = x.permute(0, 2, 1, 3).reshape(batch_size * num_steps, num_points, dim)
        xyz_bt = xyz[:, None].expand(-1, num_steps, -1, -1).reshape(
            batch_size * num_steps,
            num_points,
            3,
        )
        idx_bt = neighbor_idx[:, None].expand(-1, num_steps, -1, -1).reshape(
            batch_size * num_steps,
            num_points,
            neighbor_idx.shape[-1],
        )
        x_bt = self.spatial(x_bt, xyz_bt, idx_bt)
        x = x_bt.reshape(batch_size, num_steps, num_points, dim).permute(0, 2, 1, 3)

        x_bn = x.reshape(batch_size * num_points, num_steps, dim)
        temporal_in = self.temporal_norm(x_bn)
        temporal_out, _ = self.temporal_attn(temporal_in, temporal_in, temporal_in, need_weights=False)
        x_bn = x_bn + self.temporal_dropout(temporal_out)
        x = x_bn.reshape(batch_size, num_points, num_steps, dim)

        x_flat = x.reshape(batch_size, num_points * num_steps, dim)
        cond_in = self.cond_norm(x_flat)
        cond_out, _ = self.cond_attn(cond_in, cond_tokens, cond_tokens, need_weights=False)
        x_flat = x_flat + self.cond_dropout(cond_out)
        x_flat = x_flat + self.ffn(self.ffn_norm(x_flat))
        return x_flat.reshape(batch_size, num_points, num_steps, dim)


class SparseFlowDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        future_steps: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ):
        super().__init__()
        self.future_steps = int(future_steps)
        self.noisy_delta_proj = nn.Linear(3, dim)
        self.step_embed = nn.Embedding(self.future_steps, dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                SparseDecoderBlock(dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.velocity_head = nn.Linear(dim, 3)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        point_context: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
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
            x = block(x, xyz, neighbor_idx, cond_tokens)
        return self.velocity_head(self.out_norm(x))


class MotionExpertModel(nn.Module):
    def __init__(
        self,
        config: MotionExpertConfig,
        visual_input_dim: Optional[int] = None,
        text_input_dim: Optional[int] = None,
        max_conditioning_tokens: Optional[int] = None,
    ):
        super().__init__()
        if config.future_steps != 15:
            raise ValueError("Sparse MotionExpert is locked to future_steps=15.")
        if str(config.decoder_type).lower() != "sparse_factorized":
            raise ValueError("Sparse MotionExpert only supports decoder_type='sparse_factorized'.")

        self.config = config
        sparse_cfg = config.sparse_decoder
        self.token_dim = sparse_cfg.hidden_dim
        self.max_conditioning_tokens = max_conditioning_tokens
        self.knn_k = int(config.knn_k)

        self.point_scalar_proj = nn.Sequential(
            nn.Linear(5, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.visual_proj = TokenProjector(self.token_dim, visual_input_dim)
        self.text_proj = TokenProjector(self.token_dim, text_input_dim)
        self.point_norm = nn.LayerNorm(self.token_dim)
        self.point_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.global_visual_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.dropout = nn.Dropout(config.dropout)

        self.point_encoder = SparsePointEncoder(
            dim=self.token_dim,
            num_layers=sparse_cfg.point_encoder_layers,
            num_heads=sparse_cfg.num_attention_heads,
            mlp_ratio=sparse_cfg.mlp_ratio,
            dropout=config.dropout,
        )
        self.flow_decoder = SparseFlowDecoder(
            dim=self.token_dim,
            future_steps=config.future_steps,
            num_layers=sparse_cfg.decoder_layers,
            num_heads=sparse_cfg.num_attention_heads,
            mlp_ratio=sparse_cfg.mlp_ratio,
            dropout=config.dropout,
        )

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
        self.register_buffer(
            "query_centered_xyz_center",
            torch.tensor(config.query_centered_xyz_center, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "query_centered_xyz_scale",
            torch.tensor(config.query_centered_xyz_scale, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "traj_delta_xyz_center",
            torch.tensor(config.traj_delta_xyz_center, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "traj_delta_xyz_scale",
            torch.tensor(config.traj_delta_xyz_scale, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "use_calvin_percentile_normalization",
            torch.tensor(bool(config.use_calvin_percentile_normalization), dtype=torch.bool),
            persistent=True,
        )

    def set_query_xyz_statistics(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(center, dtype=torch.float32, device=self.query_xyz_center.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.query_xyz_scale.device)
        self.query_xyz_center.copy_(center)
        self.query_xyz_scale.copy_(torch.clamp(scale, min=1e-6))
        self.use_calvin_percentile_normalization.copy_(
            torch.tensor(False, device=self.use_calvin_percentile_normalization.device)
        )
        self.config.use_calvin_percentile_normalization = False
        self.config.query_xyz_center = tuple(float(x) for x in self.query_xyz_center.tolist())
        self.config.query_xyz_scale = tuple(float(x) for x in self.query_xyz_scale.tolist())

    def set_delta_range(self, delta_range: torch.Tensor) -> None:
        delta_range = torch.as_tensor(
            delta_range,
            dtype=torch.float32,
            device=self.delta_range_xyz.device,
        )
        self.delta_range_xyz.copy_(torch.clamp(delta_range, min=1e-6))
        self.use_calvin_percentile_normalization.copy_(
            torch.tensor(False, device=self.use_calvin_percentile_normalization.device)
        )
        self.config.use_calvin_percentile_normalization = False
        self.config.delta_range_xyz = tuple(float(x) for x in self.delta_range_xyz.tolist())

    def set_calvin_normalization_statistics(
        self,
        query_center: torch.Tensor,
        query_scale: torch.Tensor,
        traj_center: torch.Tensor,
        traj_scale: torch.Tensor,
    ) -> None:
        query_center = torch.as_tensor(
            query_center,
            dtype=torch.float32,
            device=self.query_centered_xyz_center.device,
        )
        query_scale = torch.as_tensor(
            query_scale,
            dtype=torch.float32,
            device=self.query_centered_xyz_scale.device,
        )
        traj_center = torch.as_tensor(
            traj_center,
            dtype=torch.float32,
            device=self.traj_delta_xyz_center.device,
        )
        traj_scale = torch.as_tensor(
            traj_scale,
            dtype=torch.float32,
            device=self.traj_delta_xyz_scale.device,
        )
        self.query_centered_xyz_center.copy_(query_center)
        self.query_centered_xyz_scale.copy_(torch.clamp(query_scale, min=1e-6))
        self.traj_delta_xyz_center.copy_(traj_center)
        self.traj_delta_xyz_scale.copy_(torch.clamp(traj_scale, min=1e-6))
        self.use_calvin_percentile_normalization.copy_(
            torch.tensor(True, device=self.use_calvin_percentile_normalization.device)
        )
        self.config.use_calvin_percentile_normalization = True
        self.config.query_centered_xyz_center = tuple(float(x) for x in self.query_centered_xyz_center.tolist())
        self.config.query_centered_xyz_scale = tuple(float(x) for x in self.query_centered_xyz_scale.tolist())
        self.config.traj_delta_xyz_center = tuple(float(x) for x in self.traj_delta_xyz_center.tolist())
        self.config.traj_delta_xyz_scale = tuple(float(x) for x in self.traj_delta_xyz_scale.tolist())

    def normalize_query_xyz(self, query_world_xyz: torch.Tensor) -> torch.Tensor:
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        if bool(self.use_calvin_percentile_normalization.item()):
            xyz_norm = (
                query_world_xyz - self.query_centered_xyz_center
            ) / self.query_centered_xyz_scale
        else:
            xyz_norm = (query_world_xyz - self.query_xyz_center) / self.query_xyz_scale
        xyz_norm = torch.nan_to_num(xyz_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(xyz_norm, -NORMALIZED_CLAMP_ABS, NORMALIZED_CLAMP_ABS)

    def normalize_future_delta(self, future_delta_xyz: torch.Tensor) -> torch.Tensor:
        if bool(self.use_calvin_percentile_normalization.item()):
            future_delta_xyz_norm = (
                future_delta_xyz - self.traj_delta_xyz_center
            ) / self.traj_delta_xyz_scale
        else:
            future_delta_xyz_norm = future_delta_xyz / self.delta_range_xyz
        return torch.clamp(
            future_delta_xyz_norm,
            -NORMALIZED_CLAMP_ABS,
            NORMALIZED_CLAMP_ABS,
        )

    def unnormalize_future_delta(self, future_delta_xyz_norm: torch.Tensor) -> torch.Tensor:
        if bool(self.use_calvin_percentile_normalization.item()):
            return future_delta_xyz_norm * self.traj_delta_xyz_scale + self.traj_delta_xyz_center
        return future_delta_xyz_norm * self.delta_range_xyz

    def _infer_visual_grid(
        self,
        visual_tokens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if visual_tokens is None:
            return None
        batch_size, num_tokens, channels = visual_tokens.shape
        patch_size = 8 if str(self.config.backbone_type).lower() == "florence2" else 16
        h = math.ceil(int(self.config.input_height) / float(patch_size))
        w = math.ceil(int(self.config.input_width) / float(patch_size))
        if h * w == num_tokens:
            return visual_tokens.reshape(batch_size, h, w, channels)
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            return visual_tokens.reshape(batch_size, side, side, channels)
        return None

    def sample_visual_features(
        self,
        query_uv: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if visual_grid is None:
            visual_grid = self._infer_visual_grid(visual_tokens)
        if visual_grid is not None:
            if visual_grid.dim() == 5:
                visual_grid = visual_grid[:, 0]
            if visual_grid.dim() != 4:
                raise ValueError(f"visual_grid must have shape [B, H, W, C] or [B, ncam, H, W, C], got {tuple(visual_grid.shape)}.")
            grid_nchw = visual_grid.permute(0, 3, 1, 2).contiguous()
            sample_grid = query_uv.float().unsqueeze(2)
            sampled = F.grid_sample(
                grid_nchw.float(),
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2).to(visual_tokens.dtype)
        else:
            sampled = visual_tokens.mean(dim=1, keepdim=True).expand(-1, query_uv.shape[1], -1)
        global_visual = visual_tokens.mean(dim=1, keepdim=True)
        return sampled, global_visual

    def build_point_context(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        query_world_xyz_norm = self.normalize_query_xyz(query_world_xyz)

        sampled_visual, global_visual = self.sample_visual_features(
            query_uv=query_uv,
            visual_tokens=visual_tokens,
            visual_grid=visual_grid,
        )
        point_scalars = torch.cat(
            [query_world_xyz_norm, query_uv.float()],
            dim=-1,
        )
        point_tokens = self.point_scalar_proj(point_scalars) + self.visual_proj(sampled_visual)
        point_tokens = self.point_norm(self.dropout(point_tokens)) + self.point_type_embed
        neighbor_idx = build_knn_graph(query_world_xyz, self.knn_k)
        point_context = self.point_encoder(point_tokens, query_world_xyz, neighbor_idx)

        global_visual_tokens = self.visual_proj(global_visual) + self.global_visual_type_embed
        text_cond_tokens = self.text_proj(text_tokens) + self.text_type_embed
        cond_tokens = torch.cat([global_visual_tokens, text_cond_tokens], dim=1)
        return point_context, neighbor_idx, cond_tokens

    def build_conditioning_tokens(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        point_context, _, cond_tokens = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )
        return torch.cat([point_context, cond_tokens], dim=1)

    def forward_diffusion_training(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        target_delta_xyz_norm: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
    ):
        batch_size = target_delta_xyz_norm.shape[0]
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        point_context, neighbor_idx, cond_tokens = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            visual_grid=visual_grid,
            query_depth=query_depth,
        )
        noise = torch.randn_like(target_delta_xyz_norm)
        tau = torch.rand(
            batch_size,
            device=target_delta_xyz_norm.device,
            dtype=target_delta_xyz_norm.dtype,
        )
        tau_view = tau[:, None, None, None]
        x_tau = (1.0 - tau_view) * noise + tau_view * target_delta_xyz_norm
        target_velocity = target_delta_xyz_norm - noise
        pred_velocity = self.flow_decoder(
            x_tau=x_tau,
            tau=tau,
            point_context=point_context,
            xyz=query_world_xyz,
            neighbor_idx=neighbor_idx,
            cond_tokens=cond_tokens,
        )
        return {
            "pred_velocity": pred_velocity,
            "target_velocity": target_velocity,
            "noisy_delta_xyz_norm": x_tau,
            "tau": tau,
            "point_context": point_context,
            "motion_tokens": point_context,
            "conditioning_tokens": cond_tokens,
            "neighbor_idx": neighbor_idx,
        }

    def predict(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        point_context, neighbor_idx, cond_tokens = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            visual_grid=visual_grid,
            query_depth=query_depth,
        )
        batch_size, num_points, _ = query_world_xyz.shape
        x = torch.randn(
            batch_size,
            num_points,
            self.config.future_steps,
            3,
            device=query_world_xyz.device,
            dtype=point_context.dtype,
            generator=generator,
        )
        steps = max(1, int(num_inference_steps))
        dt = 1.0 / float(steps)
        for step_idx in range(steps):
            tau = torch.full(
                (batch_size,),
                step_idx / float(steps),
                device=query_world_xyz.device,
                dtype=x.dtype,
            )
            velocity = self.flow_decoder(
                x_tau=x,
                tau=tau,
                point_context=point_context,
                xyz=query_world_xyz,
                neighbor_idx=neighbor_idx,
                cond_tokens=cond_tokens,
            )
            x = x + velocity * dt
        x = torch.clamp(x, -NORMALIZED_CLAMP_ABS, NORMALIZED_CLAMP_ABS)
        return self.unnormalize_future_delta(x)
