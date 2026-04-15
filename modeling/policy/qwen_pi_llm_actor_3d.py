"""
Pi0 / Pi0.5 dual-expert 3D policy with Qwen3-VL backbone.

Architecture:
  - Prefix expert  : full Qwen3-VL decoder stack (frozen + optional LoRA).
                     Processes scene tokens (RGB+PCD), instruction tokens,
                     and proprioception tokens in hidden_size space.
  - Action expert  : parallel lightweight stack in action_expert_dim space.
                     One ActionExpertLayer per Qwen layer, identical head
                     geometry so Q/K/V can be concatenated for shared attention.
  - 4D RoPE        : (t, x, y, z) positions on every token – temporal step
                     index plus metric 3D world coordinates.
  - Attention mask : prefix tokens are fully bidirectional; action tokens attend
                     to the full sequence (prefix + action) but prefix cannot
                     attend to actions.
  - KV caching     : at inference the prefix K/V are computed once and reused
                     across all denoising steps.
"""

from contextlib import nullcontext

import einops
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import RmsNorm
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

from ..encoder.multimodal.encoder3d_qwen import (
    _enable_lora_params,
    _is_peft_wrapped,
    _set_trainable_lora_only,
    _unwrap_peft_base,
    density_based_sampler,
)
from ..encoder.multimodal.multiscale import QwenMultiScaleFusion
from ..encoder.vision.qwen3vl import Qwen3VLBackbone, Qwen3VLNormalize
from ..utils.position_encodings import RotaryPositionEncoding4D, SinusoidalPosEmb
from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


# ---------------------------------------------------------------------------
# Merger stub (used to free PatchMerger / deepstack-merger memory)
# ---------------------------------------------------------------------------

class _MergerStub(nn.Module):
    """Zero-parameter drop-in replacement for Qwen's PatchMerger.

    This policy captures pre-merge ViT features via a forward-pre-hook that
    fires *before* the merger runs.  The merger's own output is discarded
    (get_image_features() return is never used).  Replacing the merger with
    this stub:
      - Keeps the ViT forward() from crashing (stub is callable).
      - Triggers the pre-hook as before (hook is re-registered on the stub).
      - Eliminates all merger parameter memory and optimizer states.
    """

    def forward(self, x):
        # Return value is thrown away by _encode_scene_tokens; shape does not matter.
        return x


# ---------------------------------------------------------------------------
# Action expert layer
# ---------------------------------------------------------------------------

class ActionExpertLayer(nn.Module):
    """
    Single transformer layer for the action expert in pi0-style dual-expert
    shared attention.  Head geometry (num_heads, num_kv_heads, head_dim) must
    exactly match the paired Qwen layer so that Q/K/V tensors from both experts
    can be concatenated along the sequence axis before the single softmax.
    """

    def __init__(
        self,
        action_expert_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.action_expert_dim = action_expert_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # QKV projections: action_expert_dim → attention head geometry
        self.q_proj = nn.Linear(action_expert_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(action_expert_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(action_expert_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, action_expert_dim, bias=False)

        # Per-head QK norm (matching Qwen3's q_norm / k_norm)
        self.q_norm = RmsNorm(head_dim)
        self.k_norm = RmsNorm(head_dim)

        # Layer norms
        self.input_layernorm = RmsNorm(action_expert_dim)
        self.post_attention_layernorm = RmsNorm(action_expert_dim)

        # SwiGLU FFN (same structure as Qwen)
        mlp_dim = action_expert_dim * mlp_ratio
        self.gate_proj = nn.Linear(action_expert_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(action_expert_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, action_expert_dim, bias=False)

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class QwenPiLLMActor3D(BaseDenoiseActor):
    """
    3D Qwen-pi policy with pi0-style dual-expert shared attention.

    Keeps the core 3DFA design:
      - lift visual features to 3D scene tokens,
      - scene / proprio / action share a common 3D frame,
      - actions represented as trajectory tokens,
      - 4D RoPE (time, x, y, z) on every token.
    """

    TOKEN_TYPE_CHAT = 0
    TOKEN_TYPE_SCENE = 1
    TOKEN_TYPE_INSTRUCTION = 2
    TOKEN_TYPE_PROPRIO = 3

    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,   # legacy, unused
        fps_subsampling_factor=5,
        embedding_dim=60,              # legacy, unused
        num_attn_heads=8,              # legacy, unused
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,      # legacy, unused
        relative=False,
        rotation_format="quat_xyzw",
        denoise_timesteps=100,
        denoise_model="ddpm",
        lv2_batch_size=1,
        vision_lora_rank=0,
        vision_lora_alpha=16,
        vision_lora_dropout=0.0,
        vision_lora_num_layers=0,
        vision_lora_target_modules="attn.qkv,attn.proj",
        llm_lora_rank=0,
        llm_lora_alpha=16,
        llm_lora_dropout=0.0,
        llm_lora_num_layers=0,
        llm_lora_target_modules="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        gradient_checkpointing=False,
        use_multiscale=False,
        pi_variant="pi0",
        action_expert_dim=1024,
        action_expert_mlp_ratio=4,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size,
        )
        if backbone != "qwen3vl":
            raise ValueError(
                f"{type(self).__name__} only supports backbone='qwen3vl', got {backbone!r}"
            )
        if pi_variant not in {"pi0", "pi05"}:
            raise ValueError(f"Unknown pi variant: {pi_variant!r}")

        # Null out the legacy encoder/head path from BaseDenoiseActor
        self.encoder = None
        self.traj_encoder = None
        self.prediction_head = None

        self._denoise_model_name = denoise_model
        self._pi_variant = pi_variant
        self._nhist = nhist
        self._nhand = nhand
        self.scene_subsampling_factor = max(1, int(fps_subsampling_factor))
        self.use_multiscale = bool(use_multiscale)
        self.finetune_backbone = finetune_backbone
        self.finetune_text_encoder = finetune_text_encoder
        self.num_finetune_vision_layers = num_finetune_vision_layers
        self._use_llm_lora = bool(llm_lora_rank > 0 and llm_lora_num_layers > 0)
        self._use_vision_lora = bool(vision_lora_rank > 0 and vision_lora_num_layers > 0)
        self._max_text_tokens = 1024
        self._max_action_tokens = 256
        self._max_proprio_tokens = max(8, nhist * nhand * 4)

        # ---- Backbone ----
        self.normalize = Qwen3VLNormalize()
        self.backbone = Qwen3VLBackbone(
            vision_lora_rank=vision_lora_rank,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            vision_lora_num_layers=vision_lora_num_layers,
            vision_lora_target_modules=vision_lora_target_modules,
            llm_lora_rank=llm_lora_rank,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
            llm_lora_target_modules=llm_lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
            use_multiscale=use_multiscale,
        )
        self._configure_backbone_requires_grad()
        self._replace_mergers_with_stubs()

        self.language_model = _unwrap_peft_base(self.backbone.vlm.language_model)
        self._force_eager_attention()

        # ---- Dimension constants ----
        self.hidden_size = self.backbone.llm_hidden_size
        self.action_expert_dim = action_expert_dim
        self.rot_dim = 3 if rotation_format in ("euler", "axisangle") else 6
        self.pose_dim = 3 + self.rot_dim
        self.proprio_dim = self._infer_proprio_dim(rotation_format)

        # Infer head geometry from the Qwen attention (must match action expert)
        _sa = self.language_model.layers[0].self_attn
        self.head_dim = _sa.head_dim
        self.num_heads = _sa.q_proj.out_features // self.head_dim
        self.num_kv_heads = _sa.k_proj.out_features // self.head_dim
        self.num_key_value_groups = _sa.num_key_value_groups
        self.attn_scaling = _sa.scaling
        self._attn_dropout = getattr(_sa, "attention_dropout", 0.0)

        # ---- Vision / scene encoding (prefix side, Qwen hidden_size) ----
        if self.use_multiscale:
            self.scene_fuser = QwenMultiScaleFusion(
                self.backbone.vision_hidden_size,
                self.hidden_size,
                self.backbone.multiscale_layers,
            )
        else:
            self.scene_norm = RmsNorm(self.backbone.vision_hidden_size)
            self.scene_proj = nn.Linear(
                self.backbone.vision_hidden_size, self.hidden_size, bias=False
            )

        # ---- Prefix token projections (hidden_size space) ----
        self.proprio_norm = RmsNorm(self.proprio_dim)
        self.proprio_proj = nn.Linear(self.proprio_dim, self.hidden_size, bias=False)
        self.prefix_token_type_embed = nn.Embedding(4, self.hidden_size)  # types 0–3
        self.text_index_embed = nn.Embedding(self._max_text_tokens, self.hidden_size)
        self.proprio_index_embed = nn.Embedding(
            self._max_proprio_tokens, self.hidden_size
        )

        # ---- Action token projections (action_expert_dim space) ----
        self.action_norm = RmsNorm(self.pose_dim)
        self.action_proj = nn.Linear(self.pose_dim, self.action_expert_dim, bias=False)
        self.action_index_embed = nn.Embedding(
            self._max_action_tokens, self.action_expert_dim
        )
        # Learnable per-type bias for action tokens
        self.action_type_embed = nn.Parameter(
            torch.zeros(self.action_expert_dim), requires_grad=True
        )

        # ---- 4D RoPE (shared encoder for prefix and action positions) ----
        self.relative_pe_layer = RotaryPositionEncoding4D(self.head_dim)

        # ---- Time conditioning (pi0: per-step MLP; pi0.5: per-layer AdaRMS) ----
        # time_cond_dim = action_expert_dim to match action token width
        self.time_cond_dim = action_expert_dim
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.time_cond_dim),
            nn.Linear(self.time_cond_dim, self.time_cond_dim),
            nn.SiLU(),
            nn.Linear(self.time_cond_dim, self.time_cond_dim),
        )
        self.proprio_time_proj = nn.Sequential(
            nn.Linear(
                self.proprio_dim * self._nhist * self._nhand,
                self.time_cond_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.time_cond_dim, self.time_cond_dim),
        )
        # pi0: cat(action_tokens [D], time_cond [D]) → MLP → D
        self.action_time_mlp_in = nn.Linear(
            2 * self.time_cond_dim, self.time_cond_dim
        )
        self.action_time_mlp_out = nn.Linear(self.time_cond_dim, self.time_cond_dim)

        # ---- Action expert layers ----
        n_layers = len(self.language_model.layers)
        self.action_expert_layers = nn.ModuleList([
            ActionExpertLayer(
                action_expert_dim=action_expert_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                mlp_ratio=action_expert_mlp_ratio,
            )
            for _ in range(n_layers)
        ])

        # ---- Pi0.5: per-layer AdaRMS modulation (shift, scale, gate) ----
        if self._pi_variant == "pi05":
            self.layer_modulators = nn.ModuleList([
                nn.Linear(self.time_cond_dim, 3 * self.action_expert_dim)
                for _ in range(n_layers)
            ])
            for mod in self.layer_modulators:
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)
        else:
            self.layer_modulators = None

        # ---- Action output heads (action_expert_dim space) ----
        self.final_action_norm = RmsNorm(self.action_expert_dim)
        self.pos_head = nn.Linear(self.action_expert_dim, 3, bias=False)
        self.rot_head = nn.Linear(self.action_expert_dim, self.rot_dim, bias=False)
        self.gripper_head = nn.Linear(self.action_expert_dim, 1, bias=True)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_proprio_dim(rotation_format):
        if rotation_format in ("euler", "axisangle"):
            return 7
        if rotation_format == "rotmatrix":
            return 13
        return 8

    def _force_eager_attention(self):
        self.language_model.config._attn_implementation = "eager"
        if hasattr(self.backbone.vlm, "config") and hasattr(
            self.backbone.vlm.config, "text_config"
        ):
            self.backbone.vlm.config.text_config._attn_implementation = "eager"
        for layer in self.language_model.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def _backbone_visual_dtype(self):
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
        for p in visual_inner.parameters():
            return p.dtype
        return torch.float32

    def _llm_dtype(self):
        return self.language_model.embed_tokens.weight.dtype

    def _action_expert_dtype(self):
        return self.action_expert_layers[0].q_proj.weight.dtype

    def _backbone_grad_context(self):
        needs_grad = (
            self.finetune_backbone
            or self.finetune_text_encoder
            or self.num_finetune_vision_layers > 0
            or self._use_vision_lora
            or self._use_llm_lora
        )
        return nullcontext() if needs_grad else torch.no_grad()

    @staticmethod
    def _is_merger_param(name: str) -> bool:
        """True for parameters that belong to the PatchMerger or deepstack mergers.

        We always exclude these from finetuning because this policy uses
        pre_merge_features (the hook captures the tensor *before* the merger
        runs), so merger weights receive zero gradient regardless.
        """
        return "merger" in name

    def _configure_backbone_requires_grad(self):
        vlm = self.backbone.vlm
        for p in vlm.parameters():
            p.requires_grad = False

        if self.finetune_backbone:
            for p in vlm.parameters():
                p.requires_grad = True
            return

        # ---- ViT selective finetuning ----
        # Merger / deepstack-merger weights are intentionally EXCLUDED: this
        # policy captures pre_merge_features via a forward-pre-hook, so those
        # weights sit outside the computation graph and accrue no gradient.
        visual = vlm.visual
        visual_inner = _unwrap_peft_base(visual)
        if self.num_finetune_vision_layers > 0:
            n_blocks = len(visual_inner.blocks)
            if self.num_finetune_vision_layers >= n_blocks:
                # Finetune everything in the ViT EXCEPT the mergers
                for name, p in visual_inner.named_parameters():
                    if not self._is_merger_param(name):
                        p.requires_grad = True
            else:
                # Finetune only the last N transformer blocks
                start = max(0, n_blocks - self.num_finetune_vision_layers)
                for block in visual_inner.blocks[start:]:
                    for p in block.parameters():
                        p.requires_grad = True
                # Mergers intentionally excluded (see docstring above)

        if self._use_vision_lora:
            if self.num_finetune_vision_layers == 0:
                _set_trainable_lora_only(visual)
            else:
                _enable_lora_params(visual)

        # ---- LLM LoRA or full finetuning ----
        lm = vlm.language_model
        if self.finetune_text_encoder:
            for p in lm.parameters():
                p.requires_grad = True
        elif _is_peft_wrapped(lm):
            _set_trainable_lora_only(lm)

    def _replace_mergers_with_stubs(self):
        """Replace PatchMerger and deepstack mergers with zero-param stubs.

        Context:
          - _encode_scene_tokens calls get_image_features() only to trigger a
            forward-pre-hook on the merger that captures pre-merge ViT features.
            The return value of get_image_features() is always discarded.
          - Therefore the merger and deepstack-merger weights are never part of
            any gradient computation and waste memory + optimizer state.

        We replace them with _MergerStub (no parameters) and re-register the
        pre-merge hook on the new stub so the capture still works.
        """
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)

        # ---- Main PatchMerger ----
        stub = _MergerStub()
        visual_inner.merger = stub
        # Re-register the backbone's pre-merge hook on the stub so it fires.
        stub.register_forward_pre_hook(self.backbone._pre_merge_hook)

        # ---- Deepstack mergers (used for multiscale deep visual fusion) ----
        # When use_multiscale=False these produce tokens that are never consumed
        # by our policy.  Replace with stubs to free their memory.
        ds_list = getattr(visual_inner, "deepstack_merger_list", None)
        if ds_list is not None and not self.use_multiscale:
            for i in range(len(ds_list)):
                ds_list[i] = _MergerStub()

    def _make_prefix_token_type(self, token_type, length, batch_size, device):
        if length == 0:
            return torch.zeros(
                batch_size,
                0,
                self.hidden_size,
                device=device,
                dtype=self.prefix_token_type_embed.weight.dtype,
            )
        ids = torch.full(
            (batch_size, length), token_type, device=device, dtype=torch.long
        )
        return self.prefix_token_type_embed(ids)

    def _text_order_embed(self, batch_size, length, start_idx, device, dtype):
        if length == 0:
            return torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=dtype)
        positions = torch.arange(
            start_idx, start_idx + length, device=device, dtype=torch.long
        ).clamp(max=self._max_text_tokens - 1)
        return (
            self.text_index_embed(positions)[None, :, :].to(dtype).expand(batch_size, -1, -1)
        )

    def _masked_mean(self, value, valid_mask):
        mask = valid_mask.expand_as(value).to(dtype=value.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (value * mask).sum() / denom

    def _get_template_ids(self, ids, cache_name, device):
        if len(ids) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return self.backbone._get_cached_template_ids(ids, cache_name, device)

    # ------------------------------------------------------------------
    # Time conditioning
    # ------------------------------------------------------------------

    def _encode_time_condition(self, timestep, fixed_inputs):
        time_feats = self.time_embed(timestep.float())
        proprio_input = fixed_inputs["proprio_time_input"].to(
            dtype=self.proprio_time_proj[0].weight.dtype
        )
        proprio_feats = self.proprio_time_proj(proprio_input)
        return time_feats.to(proprio_feats.dtype) + proprio_feats

    # ------------------------------------------------------------------
    # Scene / vision encoding
    # ------------------------------------------------------------------

    def _subsample_scene_tokens(self, scene_tokens, scene_xyz, scene_valid):
        if (
            self.scene_subsampling_factor <= 1
            or scene_tokens.shape[1] < self.scene_subsampling_factor
        ):
            return scene_tokens, scene_xyz, scene_valid
        sample_feats = scene_tokens.float() * scene_valid.unsqueeze(-1).float()
        keep_inds = density_based_sampler(sample_feats, self.scene_subsampling_factor)
        keep_feat = keep_inds.unsqueeze(-1).expand(-1, -1, scene_tokens.shape[-1])
        keep_xyz = keep_inds.unsqueeze(-1).expand(-1, -1, 3)
        scene_tokens = torch.gather(scene_tokens, 1, keep_feat)
        scene_xyz = torch.gather(scene_xyz, 1, keep_xyz)
        scene_valid = torch.gather(scene_valid, 1, keep_inds)
        return scene_tokens, scene_xyz, scene_valid

    def _encode_scene_tokens(self, rgb3d, pcd):
        """Returns (scene_tokens [B,N,H], scene_txyz [B,N,4], scene_valid [B,N])."""
        rgb3d_normed = self.normalize(rgb3d)
        visual_dtype = self._backbone_visual_dtype()
        if rgb3d_normed.dtype != visual_dtype:
            rgb3d_normed = rgb3d_normed.to(dtype=visual_dtype)

        B, ncam, C, H, W = rgb3d_normed.shape
        hp = H // self.backbone.patch_size
        wp = W // self.backbone.patch_size
        pixel_values = self.backbone.images_to_pixel_values(
            rgb3d_normed.reshape(B * ncam, C, H, W)
        ).to(dtype=visual_dtype)
        grid_thw = torch.tensor(
            [1, hp, wp], dtype=torch.long, device=rgb3d.device
        ).view(1, 3).expand(B * ncam, -1)

        self.backbone._pre_merge_features = None
        if self.use_multiscale:
            self.backbone._multiscale_features = {}
        with self._backbone_grad_context():
            self.backbone.vlm.visual(pixel_values, grid_thw=grid_thw)

        pre_merge_flat = self.backbone._pre_merge_features
        pre_merge = self.backbone._reshape_pre_merge_tokens(
            pre_merge_flat, B, ncam, hp, wp
        )
        h16, w16 = pre_merge.shape[2], pre_merge.shape[3]

        if self.use_multiscale:
            flat_ms = dict(self.backbone._multiscale_features)
            if self.backbone.multiscale_layers:
                flat_ms.setdefault(self.backbone.multiscale_layers[-1], pre_merge_flat)
            fuser_dtype = self.scene_fuser.projections[0].weight.dtype
            feature_maps = {
                li: self.backbone._reshape_pre_merge_tokens(
                    flat_ms[li], B, ncam, hp, wp
                ).to(dtype=fuser_dtype)
                for li in self.backbone.multiscale_layers
            }
            scene_tokens = self.scene_fuser(feature_maps)
        else:
            scene_tokens = pre_merge.reshape(B, ncam * h16 * w16, -1)
            scene_tokens = scene_tokens.to(dtype=self.scene_proj.weight.dtype)
            scene_tokens = self.scene_proj(self.scene_norm(scene_tokens))
        scene_tokens = scene_tokens.to(dtype=self._llm_dtype())

        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (h16, w16),
            mode="bilinear",
        )
        scene_xyz = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=ncam
        )
        scene_valid = torch.isfinite(scene_xyz).all(dim=-1)
        scene_valid = scene_valid & (scene_xyz.abs().sum(dim=-1) > 1e-6)
        scene_xyz = torch.nan_to_num(scene_xyz)
        scene_tokens, scene_xyz, scene_valid = self._subsample_scene_tokens(
            scene_tokens, scene_xyz, scene_valid
        )

        scene_tokens = scene_tokens + self._make_prefix_token_type(
            self.TOKEN_TYPE_SCENE, scene_tokens.shape[1], B, scene_tokens.device
        ).to(scene_tokens.dtype)

        # 4D TXYZ: t = 0 (single observation frame), xyz from PCD
        t_zero = scene_xyz.new_zeros(B, scene_xyz.shape[1], 1)
        scene_txyz = torch.cat([t_zero, scene_xyz.float()], dim=-1)  # [B, N, 4]

        return scene_tokens, scene_txyz, scene_valid

    # ------------------------------------------------------------------
    # Text / instruction encoding
    # ------------------------------------------------------------------

    def _embed_text_ids(self, token_ids):
        if token_ids.numel() == 0:
            return torch.zeros(
                token_ids.shape[0], 0, self.hidden_size,
                device=token_ids.device,
                dtype=self.language_model.embed_tokens.weight.dtype,
            )
        return self.language_model.embed_tokens(token_ids)

    def _encode_text_sections(self, instruction, instr_mask, anchor_xyz):
        """Returns prefix/suffix tokens + 4D positions.

        anchor_xyz: [B, 1, 3] world-frame anchor for language tokens.
        """
        if not torch.is_tensor(instruction):
            raise TypeError(
                f"{type(self).__name__} expects pre-tokenized instruction ids."
            )
        device = instruction.device
        batch_size, text_len = instruction.shape
        if instr_mask is None:
            instr_mask = torch.ones(batch_size, text_len, dtype=torch.bool, device=device)
        else:
            instr_mask = instr_mask.to(device=device, dtype=torch.bool)

        prefix_ids = self._get_template_ids(
            self.backbone._chat_prefix_ids, "pi_llm_prefix", device
        )
        mid_ids = self._get_template_ids(
            self.backbone._chat_mid_ids, "pi_llm_mid", device
        )
        suffix_ids = self._get_template_ids(
            self.backbone._chat_suffix_ids, "pi_llm_suffix", device
        )

        prefix_len = prefix_ids.numel()
        mid_len = mid_ids.numel()
        suffix_len = suffix_ids.numel()

        llm_dtype = self.language_model.embed_tokens.weight.dtype

        if prefix_len > 0:
            prefix_tok_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
            prefix_tokens = self._embed_text_ids(prefix_tok_ids)
            prefix_valid = torch.ones(batch_size, prefix_len, dtype=torch.bool, device=device)
        else:
            prefix_tokens = torch.zeros(
                batch_size, 0, self.hidden_size, device=device, dtype=llm_dtype
            )
            prefix_valid = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

        suffix_parts, suffix_valid_parts = [], []
        if mid_len > 0:
            suffix_parts.append(mid_ids.unsqueeze(0).expand(batch_size, -1))
            suffix_valid_parts.append(
                torch.ones(batch_size, mid_len, dtype=torch.bool, device=device)
            )
        suffix_parts.append(instruction.to(device=device))
        suffix_valid_parts.append(instr_mask)
        if suffix_len > 0:
            suffix_parts.append(suffix_ids.unsqueeze(0).expand(batch_size, -1))
            suffix_valid_parts.append(
                torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
            )

        suffix_token_ids = torch.cat(suffix_parts, dim=1)
        suffix_tokens = self._embed_text_ids(suffix_token_ids)
        suffix_valid = torch.cat(suffix_valid_parts, dim=1)

        # Token-type + order embeddings
        prefix_tokens = prefix_tokens + self._make_prefix_token_type(
            self.TOKEN_TYPE_CHAT, prefix_len, batch_size, device
        ).to(llm_dtype)
        prefix_tokens = prefix_tokens + self._text_order_embed(
            batch_size, prefix_len, 0, device, llm_dtype
        )
        suffix_tokens = suffix_tokens + self._make_prefix_token_type(
            self.TOKEN_TYPE_INSTRUCTION, suffix_tokens.shape[1], batch_size, device
        ).to(llm_dtype)
        suffix_tokens = suffix_tokens + self._text_order_embed(
            batch_size, suffix_tokens.shape[1], prefix_len, device, llm_dtype
        )

        # 4D TXYZ: t=0 for all text tokens, xyz = gripper anchor
        def _text_txyz(n_tok):
            t = anchor_xyz.new_zeros(batch_size, n_tok, 1)
            xyz = anchor_xyz.expand(-1, n_tok, -1)
            return torch.cat([t, xyz], dim=-1)  # [B, n_tok, 4]

        prefix_txyz = _text_txyz(prefix_len) if prefix_len > 0 else anchor_xyz.new_zeros(batch_size, 0, 4)
        suffix_txyz = _text_txyz(suffix_tokens.shape[1])

        return (
            prefix_tokens, prefix_txyz, prefix_valid,
            suffix_tokens, suffix_txyz, suffix_valid,
        )

    # ------------------------------------------------------------------
    # Proprioception encoding
    # ------------------------------------------------------------------

    def _encode_proprio_tokens(self, proprio):
        """Returns (tokens [B,nhist*nhand,H], txyz [B,nhist*nhand,4], valid [B,nhist*nhand])."""
        B, nhist, nhand, _ = proprio.shape
        prop_flat = proprio.flatten(1, 2)  # [B, nhist*nhand, pdim]
        prop_proj = prop_flat.to(dtype=self.proprio_proj.weight.dtype)
        proprio_tokens = self.proprio_proj(self.proprio_norm(prop_proj))
        proprio_tokens = proprio_tokens.to(dtype=self._llm_dtype())
        prop_count = proprio_tokens.shape[1]

        prop_index = torch.arange(
            prop_count, device=proprio.device, dtype=torch.long
        ).clamp(max=self._max_proprio_tokens - 1)
        proprio_tokens = proprio_tokens + self.proprio_index_embed(prop_index)[
            None, :, :
        ].to(proprio_tokens.dtype)
        proprio_tokens = proprio_tokens + self._make_prefix_token_type(
            self.TOKEN_TYPE_PROPRIO, prop_count, B, proprio.device
        ).to(proprio_tokens.dtype)

        # 4D TXYZ: t = history step index (oldest = 0), xyz from proprioception
        t_prop = torch.arange(nhist, device=proprio.device, dtype=torch.float32)
        t_prop = t_prop.unsqueeze(1).expand(nhist, nhand).reshape(nhist * nhand)
        t_prop = t_prop[None, :, None].expand(B, -1, -1)  # [B, nhist*nhand, 1]
        proprio_xyz = torch.nan_to_num(prop_flat[..., :3].float())  # [B, nhist*nhand, 3]
        proprio_txyz = torch.cat([t_prop, proprio_xyz], dim=-1)  # [B, nhist*nhand, 4]

        proprio_valid = torch.ones(B, prop_count, dtype=torch.bool, device=proprio.device)
        return proprio_tokens, proprio_txyz, proprio_valid

    # ------------------------------------------------------------------
    # Prefix assembly
    # ------------------------------------------------------------------

    def encode_inputs(
        self,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        B = proprio.shape[0]
        # Anchor language tokens to the last-frame mean gripper position
        text_anchor_xyz = torch.nan_to_num(
            proprio[:, -1, :, :3].mean(dim=1, keepdim=True).float()
        )  # [B, 1, 3]

        scene_tokens, scene_txyz, scene_valid = self._encode_scene_tokens(rgb3d, pcd)
        (
            text_prefix_tokens, text_prefix_txyz, text_prefix_valid,
            text_suffix_tokens, text_suffix_txyz, text_suffix_valid,
        ) = self._encode_text_sections(instruction, instr_mask, text_anchor_xyz)
        proprio_tokens, proprio_txyz, proprio_valid = self._encode_proprio_tokens(proprio)

        prefix_tokens = torch.cat(
            [text_prefix_tokens, scene_tokens, text_suffix_tokens, proprio_tokens], dim=1
        )
        prefix_txyz = torch.cat(
            [text_prefix_txyz, scene_txyz, text_suffix_txyz, proprio_txyz], dim=1
        )
        prefix_valid = torch.cat(
            [text_prefix_valid, scene_valid, text_suffix_valid, proprio_valid], dim=1
        )

        return {
            "query_trajectory": proprio[:, -1:],
            "proprio_time_input": proprio.flatten(1, 3),
            "prefix_tokens": prefix_tokens,
            "prefix_txyz": prefix_txyz,
            "prefix_valid": prefix_valid,
        }

    # ------------------------------------------------------------------
    # Action token construction
    # ------------------------------------------------------------------

    def _build_action_tokens(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        B, traj_len, nhand, _ = trajectory.shape
        flat_traj = trajectory.flatten(1, 2)  # [B, T*nhand, pose_dim]
        flat_traj = flat_traj.to(dtype=self.action_proj.weight.dtype)

        action_tokens = self.action_proj(self.action_norm(flat_traj))
        action_count = action_tokens.shape[1]

        action_index = torch.arange(
            action_count, device=trajectory.device, dtype=torch.long
        ).clamp(max=self._max_action_tokens - 1)
        action_tokens = action_tokens + self.action_index_embed(action_index)[
            None, :, :
        ].to(action_tokens.dtype)
        action_tokens = action_tokens + self.action_type_embed.to(action_tokens.dtype)

        # Time conditioning
        time_cond = self._encode_time_condition(timestep, fixed_inputs)

        if self._pi_variant == "pi0":
            # Mix timestep into action tokens via MLP (before the shared stack)
            time_expand = time_cond[:, None, :].expand(B, action_count, -1)
            fuse_input = torch.cat(
                [
                    action_tokens.to(self.action_time_mlp_in.weight.dtype),
                    time_expand.to(self.action_time_mlp_in.weight.dtype),
                ],
                dim=-1,
            )
            fused = self.action_time_mlp_in(fuse_input)
            fused = F.silu(fused)
            action_tokens = self.action_time_mlp_out(fused).to(
                self._action_expert_dtype()
            )
        else:
            action_tokens = action_tokens.to(self._action_expert_dtype())

        # 4D TXYZ: t = trajectory step index, xyz from (unnormalized) trajectory
        action_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:
            action_xyz = (
                fixed_inputs["query_trajectory"][..., :3]
                + torch.cumsum(action_xyz, dim=1)
            )
        action_xyz_flat = torch.nan_to_num(action_xyz.flatten(1, 2).float())  # [B, T*nhand, 3]

        t_act = torch.arange(traj_len, device=trajectory.device, dtype=torch.float32)
        t_act = t_act.unsqueeze(1).expand(traj_len, nhand).reshape(traj_len * nhand)
        t_act = t_act[None, :, None].expand(B, -1, -1)  # [B, T*nhand, 1]
        action_txyz = torch.cat([t_act, action_xyz_flat], dim=-1)  # [B, T*nhand, 4]

        if valid_mask is None:
            action_valid = torch.ones(
                B, action_count, dtype=torch.bool, device=trajectory.device
            )
        else:
            action_valid = valid_mask.squeeze(-1).flatten(1, 2).bool()

        return action_tokens, action_txyz, action_valid, time_cond

    # ------------------------------------------------------------------
    # Attention mask builders
    # ------------------------------------------------------------------

    def _build_attention_mask(self, prefix_len, full_valid, dtype):
        """Full training mask: prefix↔prefix bidirectional, action→all."""
        B, total_len = full_valid.shape
        device = full_valid.device
        allow = torch.zeros(B, total_len, total_len, device=device, dtype=torch.bool)
        if prefix_len > 0:
            allow[:, :prefix_len, :prefix_len] = True
        allow[:, prefix_len:, :] = True
        allow = allow & full_valid[:, None, :]
        attn_mask = torch.full(
            (B, 1, total_len, total_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        attn_mask.masked_fill_(allow.unsqueeze(1), 0.0)
        return attn_mask

    def _build_prefix_only_mask(self, prefix_valid, dtype):
        """Prefix-only bidirectional mask for KV-cache computation."""
        B, prefix_len = prefix_valid.shape
        device = prefix_valid.device
        # Valid query sees all valid keys (bidirectional among prefix)
        allow = prefix_valid[:, None, :].expand(-1, prefix_len, -1)
        attn_mask = torch.full(
            (B, 1, prefix_len, prefix_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        attn_mask.masked_fill_(allow.unsqueeze(1), 0.0)
        return attn_mask

    def _build_action_only_mask(self, prefix_valid, action_valid, dtype):
        """Action-only mask for cached inference: action attends to all valid keys."""
        B = prefix_valid.shape[0]
        prefix_len = prefix_valid.shape[1]
        action_len = action_valid.shape[1]
        device = prefix_valid.device
        # All valid prefix + all valid action are visible to action queries
        full_key_valid = torch.cat([prefix_valid, action_valid], dim=1)  # [B, total]
        attn_mask = torch.full(
            (B, 1, action_len, prefix_len + action_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        attn_mask.masked_fill_(full_key_valid[:, None, None, :].expand(-1, 1, action_len, -1), 0.0)
        return attn_mask

    # ------------------------------------------------------------------
    # Dual-expert attention helpers
    # ------------------------------------------------------------------

    def _apply_rope_to_qk(self, q, k, txyz):
        """Apply 4D RoPE to Q and K (both in [B, heads, N, head_dim] after transpose).

        txyz: [B, N, 4] – the 4D positions for these N tokens.
        Returns q, k with RoPE applied.
        """
        rope = self.relative_pe_layer(txyz.float())  # [B, N, head_dim, 2]
        cos = rope[..., 0].unsqueeze(1).to(q.dtype)  # [B, 1, N, head_dim]
        sin = rope[..., 1].unsqueeze(1).to(q.dtype)
        q = RotaryPositionEncoding4D.embed_rotary(q, cos, sin)
        k = RotaryPositionEncoding4D.embed_rotary(k, cos, sin)
        return q, k

    def _dual_expert_attention(
        self,
        qwen_attn,
        action_layer,
        prefix_normed,   # [B, prefix_len, hidden_size]
        action_normed,   # [B, action_len, action_expert_dim]
        prefix_txyz,     # [B, prefix_len, 4]
        action_txyz,     # [B, action_len, 4]
        attn_mask,       # [B, 1, prefix+action, prefix+action]
    ):
        B, prefix_len, _ = prefix_normed.shape
        action_len = action_normed.shape[1]

        # ---- Prefix QKV (Qwen projections) ----
        pq = qwen_attn.q_proj(prefix_normed).view(B, prefix_len, self.num_heads, self.head_dim)
        pq = qwen_attn.q_norm(pq).transpose(1, 2)   # [B, num_heads, prefix_len, head_dim]
        pk = qwen_attn.k_proj(prefix_normed).view(B, prefix_len, self.num_kv_heads, self.head_dim)
        pk = qwen_attn.k_norm(pk).transpose(1, 2)   # [B, num_kv_heads, prefix_len, head_dim]
        pv = qwen_attn.v_proj(prefix_normed).view(B, prefix_len, self.num_kv_heads, self.head_dim)
        pv = pv.transpose(1, 2)

        pq, pk = self._apply_rope_to_qk(pq, pk, prefix_txyz)

        # ---- Action QKV (action expert projections) ----
        aq = action_layer.q_proj(action_normed).view(B, action_len, self.num_heads, self.head_dim)
        aq = action_layer.q_norm(aq).transpose(1, 2)  # [B, num_heads, action_len, head_dim]
        ak = action_layer.k_proj(action_normed).view(B, action_len, self.num_kv_heads, self.head_dim)
        ak = action_layer.k_norm(ak).transpose(1, 2)  # [B, num_kv_heads, action_len, head_dim]
        av = action_layer.v_proj(action_normed).view(B, action_len, self.num_kv_heads, self.head_dim)
        av = av.transpose(1, 2)

        aq, ak = self._apply_rope_to_qk(aq, ak, action_txyz)

        # ---- Concatenate Q/K/V along the sequence dimension ----
        q = torch.cat([pq, aq], dim=2)  # [B, num_heads,    total, head_dim]
        k = torch.cat([pk, ak], dim=2)  # [B, num_kv_heads, total, head_dim]
        v = torch.cat([pv, av], dim=2)

        # GQA expand
        k = repeat_kv(k, self.num_key_value_groups)  # [B, num_heads, total, head_dim]
        v = repeat_kv(v, self.num_key_value_groups)

        # ---- Scaled dot-product attention ----
        # Use SDPA so PyTorch can dispatch to memory-efficient or flash attention,
        # avoiding the O(n²) fp32 attention-weight tensor that causes OOM at B=8.
        # attn_mask is a float additive bias [B, 1, total, total] (0 / -inf).
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask.to(q.dtype),
            dropout_p=self._attn_dropout if self.training else 0.0,
        )  # [B, num_heads, total, head_dim]

        # ---- Split and project ----
        inner_dim = self.num_heads * self.head_dim
        prefix_out = (
            attn_out[:, :, :prefix_len, :].transpose(1, 2).reshape(B, prefix_len, inner_dim)
        )
        action_out = (
            attn_out[:, :, prefix_len:, :].transpose(1, 2).reshape(B, action_len, inner_dim)
        )
        prefix_out = qwen_attn.o_proj(prefix_out)
        action_out = action_layer.o_proj(action_out)
        return prefix_out, action_out

    def _run_dual_expert_layer(
        self,
        qwen_layer,
        action_layer,
        prefix_states,   # [B, prefix_len, hidden_size]
        action_states,   # [B, action_len, action_expert_dim]
        prefix_txyz,     # [B, prefix_len, 4]
        action_txyz,     # [B, action_len, 4]
        attn_mask,       # [B, 1, total, total]
        time_cond,       # [B, action_expert_dim]
        layer_idx,
    ):
        aed = self._action_expert_dtype()

        # ---- Pre-attention norms ----
        prefix_normed = qwen_layer.input_layernorm(prefix_states)

        if self._pi_variant == "pi05":
            shift, scale, gate = self.layer_modulators[layer_idx](
                time_cond.to(self.layer_modulators[layer_idx].weight.dtype)
            ).chunk(3, dim=-1)
            shift = shift.to(aed)
            scale = scale.to(aed)
            gate = gate.to(aed)
            action_normed = action_layer.input_layernorm(action_states)
            action_normed = action_normed * (1 + scale[:, None, :]) + shift[:, None, :]
        else:
            action_normed = action_layer.input_layernorm(action_states)

        # Cast action_normed to match action_layer projections
        action_normed = action_normed.to(aed)
        # Cast prefix_normed to match Qwen projections
        prefix_normed = prefix_normed.to(self._llm_dtype())

        # ---- Shared attention (separate QKV, joint softmax, separate o_proj) ----
        prefix_attn_out, action_attn_out = self._dual_expert_attention(
            qwen_layer.self_attn, action_layer,
            prefix_normed, action_normed,
            prefix_txyz, action_txyz,
            attn_mask,
        )

        # ---- Post-attention residuals ----
        prefix_states = prefix_states + prefix_attn_out.to(prefix_states.dtype)
        if self._pi_variant == "pi05":
            action_states = action_states + gate[:, None, :] * action_attn_out.to(aed)
        else:
            action_states = action_states + action_attn_out.to(aed)

        # ---- FFNs (each expert's own) ----
        prefix_ffn_in = qwen_layer.post_attention_layernorm(prefix_states)
        prefix_states = prefix_states + qwen_layer.mlp(prefix_ffn_in)

        if self._pi_variant == "pi05":
            action_ffn_in = action_layer.post_attention_layernorm(action_states)
            action_ffn_in = action_ffn_in * (1 + scale[:, None, :]) + shift[:, None, :]
            action_states = action_states + gate[:, None, :] * action_layer.mlp(action_ffn_in)
        else:
            action_ffn_in = action_layer.post_attention_layernorm(action_states)
            action_states = action_states + action_layer.mlp(action_ffn_in)

        return prefix_states, action_states

    # ------------------------------------------------------------------
    # Full dual-expert stack (training)
    # ------------------------------------------------------------------

    def _run_dual_expert_stack(
        self,
        prefix_states,   # [B, prefix_len, hidden_size]
        action_states,   # [B, action_len, action_expert_dim]
        prefix_txyz,     # [B, prefix_len, 4]
        action_txyz,     # [B, action_len, 4]
        prefix_valid,    # [B, prefix_len]
        action_valid,    # [B, action_len]
        time_cond,       # [B, action_expert_dim]
    ):
        B, prefix_len = prefix_states.shape[:2]
        aed = self._action_expert_dtype()

        # Zero out invalid tokens at the start only (mask handles the rest)
        prefix_states = prefix_states * prefix_valid.unsqueeze(-1).to(prefix_states.dtype)
        action_states = action_states * action_valid.unsqueeze(-1).to(action_states.dtype)

        # Build single attention mask for the full sequence
        full_valid = torch.cat([prefix_valid, action_valid], dim=1)
        attn_mask = self._build_attention_mask(prefix_len, full_valid, prefix_states.dtype)

        for layer_idx, (qwen_layer, action_layer) in enumerate(
            zip(self.language_model.layers, self.action_expert_layers)
        ):
            prefix_states, action_states = self._run_dual_expert_layer(
                qwen_layer, action_layer,
                prefix_states, action_states,
                prefix_txyz, action_txyz,
                attn_mask,
                time_cond, layer_idx,
            )

        # Final norm on action output
        action_states = self.final_action_norm(action_states.to(self.final_action_norm.weight.dtype))
        action_states = action_states * action_valid.unsqueeze(-1).to(action_states.dtype)
        return action_states

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        B, traj_len, nhand, _ = trajectory.shape
        action_tokens, action_txyz, action_valid, time_cond = self._build_action_tokens(
            trajectory, timestep, fixed_inputs, valid_mask=valid_mask
        )

        prefix_tokens = fixed_inputs["prefix_tokens"]
        prefix_txyz = fixed_inputs["prefix_txyz"]
        prefix_valid = fixed_inputs["prefix_valid"]

        action_hidden = self._run_dual_expert_stack(
            prefix_tokens, action_tokens,
            prefix_txyz, action_txyz,
            prefix_valid, action_valid,
            time_cond,
        )

        pred_pos = self.pos_head(action_hidden)
        pred_rot = self.rot_head(action_hidden)
        pred_gripper = self.gripper_head(action_hidden)
        pred = torch.cat([pred_pos, pred_rot, pred_gripper], dim=-1)
        pred = pred.unflatten(1, (traj_len, nhand))
        return [pred]

    # ------------------------------------------------------------------
    # KV-cache helpers (inference only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_prefix_kv_cache(self, prefix_tokens, prefix_txyz, prefix_valid):
        """Run the prefix through the Qwen stack once, caching K/V at each layer.

        Returns list of (pk, pv) tuples, one per Qwen layer, where pk/pv are
        in [B, num_kv_heads, prefix_len, head_dim] with RoPE already applied to K.
        """
        B, prefix_len, _ = prefix_tokens.shape
        llm_dt = self._llm_dtype()

        attn_mask = self._build_prefix_only_mask(prefix_valid, llm_dt)

        prefix_states = prefix_tokens * prefix_valid.unsqueeze(-1).to(prefix_tokens.dtype)
        kv_cache = []

        for layer in self.language_model.layers:
            residual = prefix_states
            normed = layer.input_layernorm(prefix_states).to(llm_dt)

            pq = layer.self_attn.q_proj(normed).view(B, prefix_len, self.num_heads, self.head_dim)
            pq = layer.self_attn.q_norm(pq).transpose(1, 2)
            pk = layer.self_attn.k_proj(normed).view(B, prefix_len, self.num_kv_heads, self.head_dim)
            pk = layer.self_attn.k_norm(pk).transpose(1, 2)
            pv = layer.self_attn.v_proj(normed).view(B, prefix_len, self.num_kv_heads, self.head_dim)
            pv = pv.transpose(1, 2)

            pq, pk = self._apply_rope_to_qk(pq, pk, prefix_txyz.to(llm_dt))

            # Save K/V (post-RoPE K, raw V) for reuse at each denoising step
            kv_cache.append((pk, pv))

            # Self-attention to evolve prefix hidden states
            pk_exp = repeat_kv(pk, self.num_key_value_groups)
            pv_exp = repeat_kv(pv, self.num_key_value_groups)
            attn_out = F.scaled_dot_product_attention(
                pq, pk_exp, pv_exp,
                attn_mask=attn_mask.to(pq.dtype),
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, prefix_len, self.num_heads * self.head_dim)
            attn_out = layer.self_attn.o_proj(attn_out)

            prefix_states = residual + attn_out
            prefix_states = prefix_states + layer.mlp(layer.post_attention_layernorm(prefix_states))
            prefix_states = prefix_states * prefix_valid.unsqueeze(-1).to(prefix_states.dtype)

        return kv_cache

    def _policy_forward_with_cache(
        self, trajectory, timestep, fixed_inputs, prefix_kv_cache, valid_mask=None
    ):
        """Action-only forward using cached prefix K/V (efficient inference)."""
        B, traj_len, nhand, _ = trajectory.shape
        action_tokens, action_txyz, action_valid, time_cond = self._build_action_tokens(
            trajectory, timestep, fixed_inputs, valid_mask=valid_mask
        )

        prefix_valid = fixed_inputs["prefix_valid"]
        action_len = action_tokens.shape[1]
        aed = self._action_expert_dtype()

        # Attention mask: action queries attend to all valid prefix + valid action
        action_attn_mask = self._build_action_only_mask(
            prefix_valid, action_valid, action_tokens.dtype
        )

        action_states = action_tokens * action_valid.unsqueeze(-1).to(action_tokens.dtype)

        for layer_idx, (action_layer, (pk_cached, pv_cached)) in enumerate(
            zip(self.action_expert_layers, prefix_kv_cache)
        ):
            if self._pi_variant == "pi05":
                shift, scale, gate = self.layer_modulators[layer_idx](
                    time_cond.to(self.layer_modulators[layer_idx].weight.dtype)
                ).chunk(3, dim=-1)
                shift = shift.to(aed)
                scale = scale.to(aed)
                gate = gate.to(aed)
                action_normed = action_layer.input_layernorm(action_states)
                action_normed = action_normed * (1 + scale[:, None, :]) + shift[:, None, :]
            else:
                action_normed = action_layer.input_layernorm(action_states).to(aed)

            # Action QKV
            aq = action_layer.q_proj(action_normed).view(B, action_len, self.num_heads, self.head_dim)
            aq = action_layer.q_norm(aq).transpose(1, 2)
            ak = action_layer.k_proj(action_normed).view(B, action_len, self.num_kv_heads, self.head_dim)
            ak = action_layer.k_norm(ak).transpose(1, 2)
            av = action_layer.v_proj(action_normed).view(B, action_len, self.num_kv_heads, self.head_dim)
            av = av.transpose(1, 2)

            aq, ak = self._apply_rope_to_qk(aq, ak, action_txyz.to(aed))

            # Concatenate cached prefix K/V with action K/V
            pk_c = pk_cached.to(aed)
            pv_c = pv_cached.to(aed)
            k = torch.cat([pk_c, ak], dim=2)   # [B, num_kv_heads, prefix+action, head_dim]
            v = torch.cat([pv_c, av], dim=2)
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            # Attention
            action_attn_out = F.scaled_dot_product_attention(
                aq, k, v,
                attn_mask=action_attn_mask.to(aq.dtype),
            )
            action_attn_out = (
                action_attn_out.transpose(1, 2)
                .reshape(B, action_len, self.num_heads * self.head_dim)
            )
            action_attn_out = action_layer.o_proj(action_attn_out)

            # Residual + FFN
            if self._pi_variant == "pi05":
                action_states = action_states + gate[:, None, :] * action_attn_out.to(aed)
                action_ffn_in = action_layer.post_attention_layernorm(action_states)
                action_ffn_in = action_ffn_in * (1 + scale[:, None, :]) + shift[:, None, :]
                action_states = action_states + gate[:, None, :] * action_layer.mlp(action_ffn_in)
            else:
                action_states = action_states + action_attn_out.to(aed)
                action_states = action_states + action_layer.mlp(
                    action_layer.post_attention_layernorm(action_states)
                )

        action_states = self.final_action_norm(action_states.to(self.final_action_norm.weight.dtype))
        action_states = action_states * action_valid.unsqueeze(-1).to(action_states.dtype)

        pred_pos = self.pos_head(action_states)
        pred_rot = self.rot_head(action_states)
        pred_gripper = self.gripper_head(action_states)
        pred = torch.cat([pred_pos, pred_rot, pred_gripper], dim=-1)
        pred = pred.unflatten(1, (traj_len, nhand))
        return [pred]

    # ------------------------------------------------------------------
    # Inference (denoising loop)
    # ------------------------------------------------------------------

    def conditional_sample(self, trajectory, device, fixed_inputs, valid_mask=None):
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        # Build prefix KV cache once (prefix is constant across denoising steps)
        prefix_kv_cache = self._compute_prefix_kv_cache(
            fixed_inputs["prefix_tokens"],
            fixed_inputs["prefix_txyz"],
            fixed_inputs["prefix_valid"],
        )

        timesteps = self.position_scheduler.timesteps
        out = None
        for t_ind, t in enumerate(timesteps):
            t_batch = t.expand(len(trajectory)).to(device)
            if self._denoise_model_name in {"ddpm", "ddim"}:
                t_batch = t_batch.long()
            out = self._policy_forward_with_cache(
                trajectory, t_batch, fixed_inputs, prefix_kv_cache,
                valid_mask=valid_mask,
            )[-1]
            pos = self.position_scheduler.step(
                out[..., :3], t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1], t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), dim=-1)
        return torch.cat((trajectory, out[..., -1:]), dim=-1)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )

        gt_openess = gt_trajectory[..., -1:]
        gt_pose = gt_trajectory[..., :-1]
        gt_pose = self.normalize_pos(gt_pose)
        _, traj_len, nhand, _ = gt_pose.shape
        gt_pose = self.convert_rot(gt_pose.flatten(1, 2)).unflatten(1, (traj_len, nhand))
        valid_mask = (~trajectory_mask)[..., None].float()

        total_loss = 0.0
        for _ in range(self._lv2_batch_size):
            noise = torch.randn_like(gt_pose)
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )
            pos = self.position_scheduler.add_noise(
                gt_pose[..., :3], noise[..., :3], timesteps
            )
            rot = self.rotation_scheduler.add_noise(
                gt_pose[..., 3:], noise[..., 3:], timesteps
            )
            noisy_trajectory = torch.cat((pos, rot), dim=-1)

            pred = self.policy_forward_pass(
                noisy_trajectory, timesteps, fixed_inputs, valid_mask=valid_mask
            )[-1]

            denoise_target = self.position_scheduler.prepare_target(noise, gt_pose)
            pos_loss = self._masked_mean(
                (pred[..., :3] - denoise_target[..., :3]).pow(2), valid_mask
            )
            rot_loss = self._masked_mean(
                (pred[..., 3:-1] - denoise_target[..., 3:]).pow(2), valid_mask
            )
            openness_loss = self._masked_mean(
                F.binary_cross_entropy_with_logits(
                    pred[..., -1:], gt_openess, reduction="none"
                ),
                valid_mask,
            )
            total_loss = total_loss + (30.0 * pos_loss + 10.0 * rot_loss + openness_loss)

        return total_loss / self._lv2_batch_size


# ---------------------------------------------------------------------------
# Public variants
# ---------------------------------------------------------------------------

class QwenPi0LLMActor3D(QwenPiLLMActor3D):
    def __init__(self, *args, **kwargs):
        kwargs["pi_variant"] = "pi0"
        super().__init__(*args, **kwargs)


class QwenPi05LLMActor3D(QwenPiLLMActor3D):
    def __init__(self, *args, **kwargs):
        kwargs["pi_variant"] = "pi05"
        super().__init__(*args, **kwargs)
