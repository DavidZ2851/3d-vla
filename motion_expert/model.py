import math
import warnings
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import RmsNorm
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import MotionExpertConfig
from .geometry import center_query_world_xyz, validate_query_uv_grid


NORMALIZED_CLAMP_ABS = 2.0

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn_varlen_qkvpacked_func = None

_FLASH_ATTN_FALLBACK_WARNED = False


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


def _normalize_xyz_for_serialization(xyz: torch.Tensor) -> torch.Tensor:
    xyz = torch.nan_to_num(xyz.float(), nan=0.0, posinf=0.0, neginf=0.0)
    xyz_min = xyz.amin(dim=1, keepdim=True)
    xyz_max = xyz.amax(dim=1, keepdim=True)
    span = (xyz_max - xyz_min).clamp_min(1e-6)
    return (xyz - xyz_min) / span


def _normalized_xyz_to_grid(xyz_norm: torch.Tensor, depth: int) -> torch.Tensor:
    max_coord = (1 << int(depth)) - 1
    return torch.clamp((xyz_norm * max_coord).floor().long(), 0, max_coord)


def _morton3d_code(grid: torch.Tensor, depth: int, order: str) -> torch.Tensor:
    if order == "z":
        coord = grid
    elif order == "z-trans":
        coord = grid[..., [1, 0, 2]]
    else:
        raise ValueError(
            "Serialized decoder spatial attention currently supports "
            f"`z` and `z-trans` orders, got `{order}`."
        )

    x = coord[..., 0].long()
    y = coord[..., 1].long()
    z = coord[..., 2].long()
    code = torch.zeros_like(x)
    for bit in range(int(depth)):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)
    return code


def _prepare_flash_attn_dtype(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        return tensor
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return tensor.to(target_dtype) if tensor.dtype != target_dtype else tensor


def _warn_flash_attn_fallback() -> None:
    global _FLASH_ATTN_FALLBACK_WARNED
    if _FLASH_ATTN_FALLBACK_WARNED:
        return
    _FLASH_ATTN_FALLBACK_WARNED = True
    warnings.warn(
        "flash_attn_varlen_qkvpacked_func is not being used for serialized "
        "MotionExpert decoder spatial attention; falling back to PyTorch "
        "scaled_dot_product_attention. This is expected to use more memory "
        "and may be slower.",
        RuntimeWarning,
        stacklevel=3,
    )


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


class SerializedLocalAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        patch_size: int = 32,
        orders: Sequence[str] = ("z", "z-trans"),
        depth: int = 16,
        order_index: int = 0,
        shift_windows: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by {num_heads} heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("decoder_serialized_patch_size must be positive.")
        self.orders = tuple(str(order) for order in orders)
        if not self.orders:
            raise ValueError("decoder_serialized_orders must contain at least one order.")
        self.depth = int(depth)
        if self.depth <= 0 or self.depth > 20:
            raise ValueError("decoder_serialized_depth must be in [1, 20] for int64 Morton codes.")
        self.order_index = int(order_index)
        self.shift_windows = bool(shift_windows)
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.coord_proj = nn.Linear(3, dim, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def _ordered_qkv(
        self,
        x_norm: torch.Tensor,
        xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = x_norm.shape
        xyz_norm = _normalize_xyz_for_serialization(xyz)
        grid = _normalized_xyz_to_grid(xyz_norm, self.depth)
        order_name = self.orders[self.order_index % len(self.orders)]
        code = _morton3d_code(grid, self.depth, order_name)
        order = torch.argsort(code, dim=1, stable=True)
        inverse = torch.empty_like(order)
        inverse.scatter_(
            1,
            order,
            torch.arange(num_points, device=order.device).expand(batch_size, -1),
        )

        gather_idx = order.unsqueeze(-1).expand(-1, -1, self.dim)
        x_with_coord = x_norm + self.coord_proj(xyz_norm.to(dtype=x_norm.dtype))
        x_ordered = torch.gather(x_with_coord, dim=1, index=gather_idx)
        qkv = self.qkv(x_ordered).view(
            batch_size * num_points,
            3,
            self.num_heads,
            self.head_dim,
        )
        return qkv, inverse

    def _shift_offset(self, num_points: int) -> int:
        if not self.shift_windows or num_points <= self.patch_size or self.patch_size <= 1:
            return 0
        order_cycle = self.order_index // len(self.orders)
        if order_cycle % 2 == 0:
            return 0
        return min(num_points, self.patch_size // 2)

    def _window_lengths(self, num_points: int) -> Tuple[int, ...]:
        shift_offset = self._shift_offset(num_points)
        remaining = num_points
        lengths = []
        if shift_offset > 0:
            lengths.append(shift_offset)
            remaining -= shift_offset

        full_windows, remainder = divmod(remaining, self.patch_size)
        lengths.extend([self.patch_size] * full_windows)
        if remainder:
            lengths.append(remainder)
        if not lengths:
            lengths = [num_points]
        return tuple(lengths)

    def _cu_seqlens(self, batch_size: int, num_points: int, device: torch.device) -> torch.Tensor:
        lengths = self._window_lengths(num_points)
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device).repeat(batch_size)
        cu_seqlens = torch.cumsum(lengths_tensor, dim=0, dtype=torch.int32)
        return F.pad(cu_seqlens, (1, 0)).to(dtype=torch.int32)

    def _flash_attention(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        qkv_prepared = _prepare_flash_attn_dtype(qkv)
        out = flash_attn_varlen_qkvpacked_func(
            qkv_prepared,
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
        )
        return out.to(dtype=qkv.dtype)

    def _sdpa_attention(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        _warn_flash_attn_fallback()
        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q, k, v = qkv[start:end].unbind(dim=1)
            q = q.permute(1, 0, 2).unsqueeze(0)
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            outputs.append(out.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del neighbor_idx
        residual = x
        batch_size, num_points, _ = x.shape
        x_norm = self.norm1(x)
        qkv, inverse = self._ordered_qkv(x_norm, xyz)
        cu_seqlens = self._cu_seqlens(batch_size, num_points, x.device)

        if flash_attn_varlen_qkvpacked_func is not None and qkv.device.type == "cuda":
            out_ordered = self._flash_attention(qkv, cu_seqlens)
        else:
            out_ordered = self._sdpa_attention(qkv, cu_seqlens)

        out_ordered = out_ordered.reshape(batch_size, num_points, self.dim)
        out = torch.gather(
            out_ordered,
            dim=1,
            index=inverse.unsqueeze(-1).expand(-1, -1, self.dim),
        )
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
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        self.checkpoint_blocks = bool(checkpoint_blocks)
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
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(layer, x, xyz, neighbor_idx, use_reentrant=False)
            else:
                x = layer(x, xyz, neighbor_idx)
        return x


class SparseDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        spatial_attention: str = "knn",
        serialized_patch_size: int = 32,
        serialized_orders: Sequence[str] = ("z", "z-trans"),
        serialized_depth: int = 16,
        order_index: int = 0,
        serialized_shift_windows: bool = True,
    ):
        super().__init__()
        spatial_attention = str(spatial_attention).lower()
        if spatial_attention == "knn":
            self.spatial = LocalGeometryAttentionLayer(dim, num_heads, mlp_ratio, dropout)
        elif spatial_attention == "serialized_flash":
            self.spatial = SerializedLocalAttentionLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                patch_size=serialized_patch_size,
                orders=serialized_orders,
                depth=serialized_depth,
                order_index=order_index,
                shift_windows=serialized_shift_windows,
            )
        else:
            raise ValueError(
                "decoder_spatial_attention must be `knn` or `serialized_flash`, "
                f"got `{spatial_attention}`."
            )
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
        cond_key_padding_mask: torch.Tensor,
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
        cond_out, _ = self.cond_attn(
            cond_in,
            cond_tokens,
            cond_tokens,
            key_padding_mask=cond_key_padding_mask,
            need_weights=False,
        )
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
        checkpoint_blocks: bool = True,
        spatial_attention: str = "knn",
        serialized_patch_size: int = 32,
        serialized_orders: Sequence[str] = ("z", "z-trans"),
        serialized_depth: int = 16,
        serialized_shift_windows: bool = True,
    ):
        super().__init__()
        self.future_steps = int(future_steps)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.noisy_delta_proj = nn.Linear(3, dim)
        self.step_embed = nn.Embedding(self.future_steps, dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                SparseDecoderBlock(
                    dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    spatial_attention=spatial_attention,
                    serialized_patch_size=serialized_patch_size,
                    serialized_orders=serialized_orders,
                    serialized_depth=serialized_depth,
                    order_index=idx,
                    serialized_shift_windows=serialized_shift_windows,
                )
                for idx in range(num_layers)
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
        cond_key_padding_mask: torch.Tensor,
        return_hidden: bool = False,
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
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    xyz,
                    neighbor_idx,
                    cond_tokens,
                    cond_key_padding_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, xyz, neighbor_idx, cond_tokens, cond_key_padding_mask)
        hidden = self.out_norm(x)
        velocity = self.velocity_head(hidden)
        if return_hidden:
            return velocity, hidden
        return velocity


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
        self.visual_global_tokens = int(sparse_cfg.visual_global_tokens)
        if self.visual_global_tokens <= 0:
            raise ValueError("sparse_decoder.visual_global_tokens must be positive.")
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
            checkpoint_blocks=config.checkpoint_activations,
        )
        self.flow_decoder = SparseFlowDecoder(
            dim=self.token_dim,
            future_steps=config.future_steps,
            num_layers=sparse_cfg.decoder_layers,
            num_heads=sparse_cfg.num_attention_heads,
            mlp_ratio=sparse_cfg.mlp_ratio,
            dropout=config.dropout,
            checkpoint_blocks=config.checkpoint_activations,
            spatial_attention=sparse_cfg.decoder_spatial_attention,
            serialized_patch_size=sparse_cfg.decoder_serialized_patch_size,
            serialized_orders=sparse_cfg.decoder_serialized_orders,
            serialized_depth=sparse_cfg.decoder_serialized_depth,
            serialized_shift_windows=sparse_cfg.decoder_serialized_shift_windows,
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
        patch_size = 32 if str(self.config.backbone_type).lower() == "florence2" else 16
        h = math.ceil(int(self.config.input_height) / float(patch_size))
        w = math.ceil(int(self.config.input_width) / float(patch_size))
        if h * w == num_tokens:
            return visual_tokens.reshape(batch_size, h, w, channels)
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            return visual_tokens.reshape(batch_size, side, side, channels)
        return None

    @staticmethod
    def _visual_pool_hw(num_tokens: int) -> Tuple[int, int]:
        pool_h = max(1, int(math.floor(math.sqrt(num_tokens))))
        pool_w = int(math.ceil(num_tokens / float(pool_h)))
        return pool_h, pool_w

    def _pool_visual_conditioning_tokens(
        self,
        visual_tokens: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.visual_global_tokens == 1:
            return visual_tokens.mean(dim=1, keepdim=True)

        pool_h, pool_w = self._visual_pool_hw(self.visual_global_tokens)
        if visual_grid is not None and visual_grid.dim() in (4, 5):
            grid = visual_grid
            if grid.dim() == 5:
                batch_size, num_cameras, height, width, channels = grid.shape
                grid = grid.reshape(batch_size, num_cameras * height, width, channels)
            grid_nchw = grid.permute(0, 3, 1, 2).contiguous()
            pooled = F.adaptive_avg_pool2d(grid_nchw.float(), (pool_h, pool_w))
            pooled = pooled.flatten(2).transpose(1, 2)
            return pooled[:, : self.visual_global_tokens].to(dtype=visual_tokens.dtype)

        pooled = F.adaptive_avg_pool1d(
            visual_tokens.transpose(1, 2).float(),
            self.visual_global_tokens,
        )
        return pooled.transpose(1, 2).to(dtype=visual_tokens.dtype)

    def sample_visual_features(
        self,
        query_uv: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_uv = validate_query_uv_grid(query_uv)
        if visual_grid is None:
            visual_grid = self._infer_visual_grid(visual_tokens)
        global_visual = self._pool_visual_conditioning_tokens(visual_tokens, visual_grid)
        sampling_grid = visual_grid
        if sampling_grid is not None:
            if sampling_grid.dim() == 5:
                sampling_grid = sampling_grid[:, 0]
            if sampling_grid.dim() != 4:
                raise ValueError(f"visual_grid must have shape [B, H, W, C] or [B, ncam, H, W, C], got {tuple(sampling_grid.shape)}.")
            grid_nchw = sampling_grid.permute(0, 3, 1, 2).contiguous()
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
        return sampled, global_visual

    def _build_conditioning_key_padding_mask(
        self,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        global_token_count: int,
    ) -> torch.Tensor:
        batch_size, text_token_count, _ = text_tokens.shape
        global_mask = torch.zeros(
            batch_size,
            int(global_token_count),
            dtype=torch.bool,
            device=text_tokens.device,
        )
        if text_attention_mask is None:
            text_mask = torch.zeros(
                batch_size,
                text_token_count,
                dtype=torch.bool,
                device=text_tokens.device,
            )
        else:
            if text_attention_mask.shape != text_tokens.shape[:2]:
                raise ValueError(
                    "text_attention_mask must match text_tokens shape [B, L], "
                    f"got {tuple(text_attention_mask.shape)} for text_tokens "
                    f"{tuple(text_tokens.shape[:2])}."
                )
            text_mask = text_attention_mask.to(device=text_tokens.device) <= 0
        return torch.cat([global_mask, text_mask], dim=1)

    def build_point_context(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_uv = validate_query_uv_grid(query_uv)
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
        cond_key_padding_mask = self._build_conditioning_key_padding_mask(
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
            global_token_count=global_visual_tokens.shape[1],
        )
        return point_context, neighbor_idx, cond_tokens, cond_key_padding_mask

    def build_conditioning_tokens(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        point_context, _, cond_tokens, _ = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
        )
        return torch.cat([point_context, cond_tokens], dim=1)

    def forward_diffusion_training(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        target_delta_xyz_norm: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
    ):
        batch_size = target_delta_xyz_norm.shape[0]
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
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
            cond_key_padding_mask=cond_key_padding_mask,
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

    def extract_one_ode_step_latent(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
        solver_steps: int = 10,
        steps_to_run: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        del solver_steps
        if int(steps_to_run) != 1:
            raise ValueError(
                "Motion latent extraction currently supports exactly one ODE "
                f"solver step, got steps_to_run={steps_to_run}."
            )
        motion_pos = torch.nan_to_num(
            query_world_xyz.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
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
        tau = torch.zeros(
            batch_size,
            device=query_world_xyz.device,
            dtype=x.dtype,
        )
        _, motion_hidden = self.flow_decoder(
            x_tau=x,
            tau=tau,
            point_context=point_context,
            xyz=query_world_xyz,
            neighbor_idx=neighbor_idx,
            cond_tokens=cond_tokens,
            cond_key_padding_mask=cond_key_padding_mask,
            return_hidden=True,
        )
        return {
            "motion_hidden": motion_hidden,
            "motion_pos": motion_pos,
        }

    def predict(
        self,
        query_uv: torch.Tensor,
        query_world_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        visual_grid: Optional[torch.Tensor] = None,
        query_depth: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        query_world_xyz = center_query_world_xyz(query_world_xyz)
        point_context, neighbor_idx, cond_tokens, cond_key_padding_mask = self.build_point_context(
            query_uv=query_uv,
            query_world_xyz=query_world_xyz,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_attention_mask=text_attention_mask,
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
                cond_key_padding_mask=cond_key_padding_mask,
            )
            x = x + velocity * dt
        x = torch.clamp(x, -NORMALIZED_CLAMP_ABS, NORMALIZED_CLAMP_ABS)
        return self.unnormalize_future_delta(x)
