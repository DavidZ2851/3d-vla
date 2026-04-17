from contextlib import nullcontext

import einops
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

from ...utils.position_encodings import RotaryPositionEncoding4D
from .encoder3d_qwen import Qwen3VLEncoder3D, _unwrap_peft_base


class Qwen3VLEncoderQwen3D(Qwen3VLEncoder3D):
    """
    Qwen3D-style Calvin encoder built on top of the existing Qwen3-VL backbone.

    Compared with ``Qwen3VLEncoder3D(use_post_merge=True)``, this encoder does
    not feed the merged scene tokens directly to the downstream policy head.
    Instead it:
      1. extracts native post-merger stride-32 visual tokens,
      2. lifts them into world-frame 3D with the downsampled PCD,
      3. injects those tokens into the Qwen prefix in place of the visual
         placeholder span,
      4. contextualizes the full prefix with bidirectional attention and custom
         4D ``(t, x, y, z)`` RoPE,
      5. projects the resulting scene/text tokens back to ``embedding_dim``.
    """

    def __init__(self, *args, use_multiscale=False, **kwargs):
        if use_multiscale:
            raise ValueError(
                "Qwen3VLEncoderQwen3D only supports use_multiscale=False."
            )
        super().__init__(
            *args,
            use_multiscale=False,
            use_post_merge=True,
            use_post_llm_visual=False,
            **kwargs,
        )

        self.language_model = _unwrap_peft_base(self.backbone.vlm.language_model)
        attn = self.language_model.layers[0].self_attn
        self.qwen_head_dim = attn.head_dim
        self.qwen_num_heads = attn.q_proj.out_features // attn.head_dim
        self.qwen_num_kv_heads = attn.k_proj.out_features // attn.head_dim
        self.qwen_num_key_value_groups = attn.num_key_value_groups
        self.qwen_attn_dropout = getattr(attn, "attention_dropout", 0.0)
        self.qwen_rope = RotaryPositionEncoding4D(self.qwen_head_dim)

        # The Qwen3D path reuses the visual tower up to the main post-merger
        # tokens, then runs its own text contextualizer. MiBoT's auxiliary
        # deepstack mergers are therefore never consumed downstream here, so
        # they must stay frozen or DDP will report them as unused parameters.
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
        for merger in getattr(visual_inner, "deepstack_merger_list", []):
            for param in merger.parameters():
                param.requires_grad = False

    def _qwen_grad_context(self):
        needs_grad = (
            self.finetune_backbone
            or self.finetune_text_encoder
            or self.num_finetune_vision_layers > 0
            or self._use_vision_lora
            or self._use_llm_lora
        )
        return nullcontext() if needs_grad else torch.no_grad()

    def _resolve_instr_mask(self, instruction, instr_mask, instr_lens):
        if instr_mask is not None:
            return instr_mask.to(device=instruction.device, dtype=torch.bool)
        if instr_lens is None:
            return torch.ones_like(instruction, dtype=torch.bool, device=instruction.device)

        lens = torch.as_tensor(instr_lens, device=instruction.device, dtype=torch.long)
        token_ids = torch.arange(instruction.shape[1], device=instruction.device)
        return token_ids.unsqueeze(0) < lens.unsqueeze(1)

    def _extract_post_merge_scene_tokens(self, rgb3d):
        rgb3d_normed = self.normalize(rgb3d)
        visual_dtype = self._backbone_visual_dtype()
        if rgb3d_normed.dtype != visual_dtype:
            rgb3d_normed = rgb3d_normed.to(dtype=visual_dtype)

        batch_size, num_cameras, channels, height, width = rgb3d_normed.shape
        hp = height // self.backbone.patch_size
        wp = width // self.backbone.patch_size
        ms = self.backbone.spatial_merge_size
        h32, w32 = hp // ms, wp // ms

        pixel_values = self.backbone.images_to_pixel_values(
            rgb3d_normed.reshape(batch_size * num_cameras, channels, height, width)
        ).to(dtype=visual_dtype)
        grid_thw = torch.tensor(
            [1, hp, wp], dtype=torch.long, device=rgb3d.device
        ).view(1, 3).expand(batch_size * num_cameras, -1)

        self._post_merge_features = None
        with self._qwen_grad_context():
            self.backbone.vlm.visual(pixel_values, grid_thw=grid_thw)

        if self._post_merge_features is None:
            raise RuntimeError("Failed to capture Qwen post-merger visual features.")

        # MiBoT/Qwen3VL keeps merge-block order through patch flattening,
        # rotary indexing, and PatchMerger, so the captured stride-32 token
        # sequence stays aligned with our downsampled PCD flatten order.
        scene_tokens = self._post_merge_features.reshape(
            batch_size, num_cameras * h32 * w32, -1
        )
        return scene_tokens, num_cameras, h32, w32

    def _lift_scene_tokens(self, pcd, num_cameras, grid_h, grid_w):
        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (grid_h, grid_w),
            mode="bilinear",
        )
        scene_xyz = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=num_cameras
        ).float()
        scene_valid = torch.isfinite(scene_xyz).all(dim=-1)
        scene_valid = scene_valid & (scene_xyz.abs().sum(dim=-1) > 1e-6)
        scene_xyz = torch.nan_to_num(scene_xyz)
        return scene_xyz, scene_valid

    def _inject_scene_tokens(self, instruction, instr_mask, scene_tokens, scene_xyz, scene_valid, num_cameras):
        device = instruction.device
        merged_tok = scene_tokens.shape[1] // num_cameras
        input_ids, instr_start, text_len = self.backbone.build_input_ids(
            instruction, num_cameras, merged_tok, device
        )
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_token_mask = input_ids.eq(self.backbone.image_token_id)
        expected_tokens = int(image_token_mask[0].sum().item())
        if scene_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Scene token count does not match Qwen placeholder span: "
                f"{scene_tokens.shape[1]} vs {expected_tokens}."
            )

        inputs_embeds = inputs_embeds.masked_scatter(
            image_token_mask.unsqueeze(-1),
            scene_tokens.to(dtype=inputs_embeds.dtype).reshape(-1),
        )

        prefix_valid = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        prefix_valid[:, instr_start : instr_start + text_len] = instr_mask
        prefix_valid = prefix_valid.masked_scatter(
            image_token_mask,
            scene_valid.reshape(-1),
        )

        prefix_txyz = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], 4,
            dtype=torch.float32,
            device=device,
        )
        scene_txyz = torch.cat(
            [scene_xyz.new_zeros(scene_xyz.shape[0], scene_xyz.shape[1], 1), scene_xyz],
            dim=-1,
        )
        prefix_txyz = prefix_txyz.masked_scatter(
            image_token_mask.unsqueeze(-1).expand(-1, -1, 4),
            scene_txyz.reshape(-1),
        )

        for batch_idx in range(input_ids.shape[0]):
            text_like_mask = prefix_valid[batch_idx] & (~image_token_mask[batch_idx])
            token_count = int(text_like_mask.sum().item())
            if token_count == 0:
                continue
            prefix_txyz[batch_idx, text_like_mask, 0] = torch.arange(
                token_count, device=device, dtype=torch.float32
            )

        return input_ids, inputs_embeds, prefix_valid, prefix_txyz, instr_start, text_len

    @staticmethod
    def _build_prefix_attention_mask(prefix_valid, dtype):
        batch_size, prefix_len = prefix_valid.shape
        device = prefix_valid.device
        allow = prefix_valid[:, None, :].expand(-1, prefix_len, -1)
        attn_mask = torch.full(
            (batch_size, 1, prefix_len, prefix_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        attn_mask.masked_fill_(allow.unsqueeze(1), 0.0)
        return attn_mask

    def _apply_qwen_rope(self, q, k, prefix_txyz):
        rope = self.qwen_rope(prefix_txyz.float())
        cos = rope[..., 0].unsqueeze(1).to(q.dtype)
        sin = rope[..., 1].unsqueeze(1).to(q.dtype)
        q = RotaryPositionEncoding4D.embed_rotary(q, cos, sin)
        k = RotaryPositionEncoding4D.embed_rotary(k, cos, sin)
        return q, k

    def _contextualize_prefix(self, inputs_embeds, prefix_valid, prefix_txyz):
        hidden_states = inputs_embeds * prefix_valid.unsqueeze(-1).to(inputs_embeds.dtype)
        attn_mask = self._build_prefix_attention_mask(prefix_valid, hidden_states.dtype)

        for layer in self.language_model.layers:
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            q = layer.self_attn.q_proj(normed).view(
                hidden_states.shape[0], hidden_states.shape[1], self.qwen_num_heads, self.qwen_head_dim
            )
            q = layer.self_attn.q_norm(q).transpose(1, 2)
            k = layer.self_attn.k_proj(normed).view(
                hidden_states.shape[0], hidden_states.shape[1], self.qwen_num_kv_heads, self.qwen_head_dim
            )
            k = layer.self_attn.k_norm(k).transpose(1, 2)
            v = layer.self_attn.v_proj(normed).view(
                hidden_states.shape[0], hidden_states.shape[1], self.qwen_num_kv_heads, self.qwen_head_dim
            ).transpose(1, 2)

            q, k = self._apply_qwen_rope(q, k, prefix_txyz)
            k = repeat_kv(k, self.qwen_num_key_value_groups)
            v = repeat_kv(v, self.qwen_num_key_value_groups)

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.qwen_attn_dropout if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).reshape(
                hidden_states.shape[0], hidden_states.shape[1], self.qwen_num_heads * self.qwen_head_dim
            )
            hidden_states = residual + layer.self_attn.o_proj(attn_out)
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
            hidden_states = hidden_states * prefix_valid.unsqueeze(-1).to(hidden_states.dtype)

        hidden_states = self.language_model.norm(hidden_states)
        return hidden_states * prefix_valid.unsqueeze(-1).to(hidden_states.dtype)

    def forward(
        self,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        if not torch.is_tensor(instruction):
            raise TypeError(
                f"{type(self).__name__} expects pre-tokenized instruction ids."
            )

        instr_mask = self._resolve_instr_mask(instruction, instr_mask, instr_lens)
        scene_tokens, num_cameras, grid_h, grid_w = self._extract_post_merge_scene_tokens(rgb3d)
        scene_xyz, scene_valid = self._lift_scene_tokens(pcd, num_cameras, grid_h, grid_w)
        (
            input_ids,
            inputs_embeds,
            prefix_valid,
            prefix_txyz,
            instr_start,
            text_len,
        ) = self._inject_scene_tokens(
            instruction,
            instr_mask,
            scene_tokens,
            scene_xyz,
            scene_valid,
            num_cameras,
        )

        hidden_states = self._contextualize_prefix(inputs_embeds, prefix_valid, prefix_txyz)

        image_token_mask = input_ids.eq(self.backbone.image_token_id)
        image_positions = image_token_mask[0].nonzero(as_tuple=False).squeeze(-1)
        scene_hidden = hidden_states[:, image_positions, :]
        rgb3d_feats = self.vis_proj(
            self.vis_norm(scene_hidden.to(dtype=self.vis_proj.weight.dtype))
        )

        instr_hidden = hidden_states[:, instr_start : instr_start + text_len, :]
        instr_hidden = instr_hidden * instr_mask.to(
            device=instr_hidden.device, dtype=instr_hidden.dtype
        ).unsqueeze(-1)
        instr_hidden = self.backbone.token_dropout(instr_hidden)
        instr_feats = self.cond_linear(
            self.cond_norm(instr_hidden.to(dtype=self.cond_linear.weight.dtype))
        )

        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, scene_xyz)
        fps_feats, fps_pos = self.run_fps(rgb3d_feats, scene_xyz)

        return (
            rgb3d_feats, scene_xyz,
            None, None,
            instr_feats, instr_pos,
            proprio_feats,
            fps_feats, fps_pos,
        )
