import os
import warnings
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    except ImportError:
        flash_attn_func = None
        flash_attn_varlen_func = None


_FLASH_ATTN_FALLBACK_WARNED = set()
_FLASH_ATTN_BACKEND_LOGGED = set()


def force_sdpa_attention() -> bool:
    return (
        os.environ.get("FLOWERVLA_FORCE_SDPA_ATTENTION", "0") == "1"
        or os.environ.get("MOTION_EXPERT_FORCE_SDPA_ATTENTION", "0") == "1"
    )


def disable_varlen_flash_attn() -> bool:
    return (
        os.environ.get("FLOWERVLA_DISABLE_VARLEN_FLASH_ATTN", "0") == "1"
        or os.environ.get("MOTION_EXPERT_DISABLE_VARLEN_FLASH_ATTN", "0") == "1"
    )


def deterministic_flash_attn() -> bool:
    return (
        os.environ.get("FLOWERVLA_FLASH_ATTN_DETERMINISTIC", "0") == "1"
        or os.environ.get("MOTION_EXPERT_FLASH_ATTN_DETERMINISTIC", "0") == "1"
    )


def flash_attn_available() -> bool:
    return flash_attn_func is not None and not force_sdpa_attention()


def sdpa_kernel_context():
    if not force_sdpa_attention() or not torch.cuda.is_available():
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    )


def prepare_flash_attn_dtype(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cuda":
        return tensor
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return tensor.to(target_dtype) if tensor.dtype != target_dtype else tensor


def warn_flash_attn_fallback(name: str) -> None:
    if name in _FLASH_ATTN_FALLBACK_WARNED:
        return
    _FLASH_ATTN_FALLBACK_WARNED.add(name)
    warnings.warn(
        f"{name} is falling back to PyTorch scaled_dot_product_attention because "
        "flash-attn is unavailable or the requested mask/layout is unsupported.",
        RuntimeWarning,
        stacklevel=3,
    )


def _should_log_backend(name: str) -> bool:
    if name in _FLASH_ATTN_BACKEND_LOGGED:
        return False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return False
    _FLASH_ATTN_BACKEND_LOGGED.add(name)
    return True


def log_flash_attn_backend(name: str, backend: str, reason: Optional[str] = None) -> None:
    if not _should_log_backend(name):
        return
    suffix = f" ({reason})" if reason else ""
    print(f"[flash-attn] {name}: {backend}{suffix}")


def _lengths_to_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


def scaled_dot_product_attention_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    attn_mask = None
    if key_padding_mask is not None:
        attn_mask = torch.zeros(
            key_padding_mask.shape[0],
            1,
            1,
            key_padding_mask.shape[1],
            dtype=q.dtype,
            device=q.device,
        )
        attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    with sdpa_kernel_context():
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
    return out.transpose(1, 2)


def flash_attention_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_padding_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
    fallback_name: str = "attention",
) -> torch.Tensor:
    if force_sdpa_attention() or flash_attn_func is None or q.device.type != "cuda":
        if force_sdpa_attention():
            reason = "forced by env"
        elif flash_attn_func is None:
            reason = "flash_attn not installed"
        else:
            reason = f"device={q.device.type}"
        log_flash_attn_backend(fallback_name, "sdpa-fallback", reason)
        warn_flash_attn_fallback(fallback_name)
        return scaled_dot_product_attention_bshd(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    original_dtype = q.dtype
    q_prepared = prepare_flash_attn_dtype(q)
    k_prepared = prepare_flash_attn_dtype(k)
    v_prepared = prepare_flash_attn_dtype(v)

    if key_padding_mask is None or not bool(key_padding_mask.any().item()):
        log_flash_attn_backend(fallback_name, "flash_attn_func")
        out = flash_attn_func(
            q_prepared,
            k_prepared,
            v_prepared,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        return out.to(dtype=original_dtype)

    if disable_varlen_flash_attn() or flash_attn_varlen_func is None:
        reason = (
            "varlen flash disabled by env"
            if disable_varlen_flash_attn()
            else "flash_attn_varlen_func unavailable"
        )
        log_flash_attn_backend(fallback_name, "sdpa-fallback", reason)
        warn_flash_attn_fallback(fallback_name)
        return scaled_dot_product_attention_bshd(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    batch_size, query_len, num_heads, head_dim = q_prepared.shape
    _, key_len, _, _ = k_prepared.shape
    valid_key_mask = ~key_padding_mask.to(dtype=torch.bool)
    key_lengths = valid_key_mask.sum(dim=1, dtype=torch.int32)
    if torch.any(key_lengths <= 0):
        log_flash_attn_backend(fallback_name, "sdpa-fallback", "empty key sequence")
        warn_flash_attn_fallback(fallback_name)
        return scaled_dot_product_attention_bshd(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    query_lengths = torch.full(
        (batch_size,),
        query_len,
        dtype=torch.int32,
        device=q_prepared.device,
    )
    flat_q = q_prepared.reshape(batch_size * query_len, num_heads, head_dim)
    flat_k = k_prepared[valid_key_mask]
    flat_v = v_prepared[valid_key_mask]
    log_flash_attn_backend(fallback_name, "flash_attn_varlen_func")
    out = flash_attn_varlen_func(
        flat_q,
        flat_k,
        flat_v,
        _lengths_to_cu_seqlens(query_lengths),
        _lengths_to_cu_seqlens(key_lengths),
        max_seqlen_q=query_len,
        max_seqlen_k=int(key_lengths.max().item()),
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
        deterministic=deterministic_flash_attn(),
    )
    return out.reshape(batch_size, query_len, num_heads, head_dim).to(dtype=original_dtype)


class FlashMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, use_flash_attn: bool = True, flash_attn_name: str = "attention", **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flash_attn = bool(use_flash_attn)
        self.flash_attn_name = str(flash_attn_name)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if (
            not self.use_flash_attn
            or need_weights
            or attn_mask is not None
            or not self._qkv_same_embed_dim
            or self.bias_k is not None
            or self.bias_v is not None
            or self.add_zero_attn
            or query.dim() != 3
            or key.dim() != 3
            or value.dim() != 3
        ):
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if self.batch_first:
            query_bf, key_bf, value_bf = query, key, value
        else:
            query_bf, key_bf, value_bf = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        q_proj, k_proj, v_proj = F._in_projection_packed(
            query_bf,
            key_bf,
            value_bf,
            self.in_proj_weight,
            self.in_proj_bias,
        )
        batch_size, query_len, embed_dim = q_proj.shape
        key_len = k_proj.shape[1]
        head_dim = embed_dim // self.num_heads
        q_proj = q_proj.reshape(batch_size, query_len, self.num_heads, head_dim)
        k_proj = k_proj.reshape(batch_size, key_len, self.num_heads, head_dim)
        v_proj = v_proj.reshape(batch_size, key_len, self.num_heads, head_dim)

        key_padding_mask_bool = None
        if key_padding_mask is not None:
            key_padding_mask_bool = key_padding_mask.to(device=query_bf.device, dtype=torch.bool)

        out = flash_attention_bshd(
            q_proj,
            k_proj,
            v_proj,
            key_padding_mask=key_padding_mask_bool,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            fallback_name=self.flash_attn_name,
        )
        out = out.reshape(batch_size, query_len, embed_dim)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None
