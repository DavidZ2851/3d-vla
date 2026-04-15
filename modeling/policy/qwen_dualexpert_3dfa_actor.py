"""
Hybrid 3DFA + Pi0 Dual-Expert Policy with Qwen3-VL backbone.

Architecture summary
--------------------
Encoder (runs once per observation):
  - Qwen3-VL ViT + **PatchMerger** → merged visual embeddings injected into
    the Qwen LLM prefix.  PCD bilinearly downsampled to stride-32 grid.
  - Qwen LLM runs with the original MiBot/Qwen causal flash-attn2 path.
    Per-layer hidden states are captured via registered forward hooks.
  - **scene_feats**: post-LLM visual features (final hidden states at image-token
    positions), projected to embedding_dim.
  - **instr_feats**: post-LLM text features (final hidden states at instruction-
    token positions), projected to embedding_dim.
  - Proprioception encoded via cross-attention to scene (same as 3DFA-VLA).

Decoder (runs at every denoising timestep):
  - Action tokens: traj_encoder(noisy_trajectory) + hand_embed + sinusoidal step pos.
  - Time conditioning: SinusoidalEmb(t) + MLP(proprio_feats) → AdaLN signal.
  - Main stage (num_shared_attn_layers sampled Qwen depths):
      1) action tokens cross-attend to final instruction features once
      2) for each selected Qwen layer i, action cross-attends to that layer's refined
         RGB + text tokens with 3D RoPE; AdaLN uses timestep + proprio encoded
         against that layer's refined RGB scene tokens
  - Output heads:
      pos/rot branches cross-attend to the final RGB + text tokens; AdaLN uses
      timestep + proprio encoded against the final RGB scene tokens.
      openess from position branch features.

Key properties vs. existing policies
--------------------------------------
  vs. QwenPi{0,05}LLMActor3D (pi0/pi0.5):
    + Uses post-merge (stride 32) instead of pre-merge (stride 16).
    + Uses 3D RoPE instead of 4D TXYZ RoPE.
    + Uses AdaLN instead of AdaRMS/token-fusion for time conditioning.
    + Uses 3DFA-VLA output heads (DoubleCross+Self) instead of linear heads.
    + Keeps original flash-attn2 causal attention (no custom mask).
    - Action does NOT jointly attend to prefix in a single softmax; instead,
      Qwen LLM runs independently and action expert cross-attends to sampled layers.
  vs. QwenFlowerActor3D (3DFA-VLA):
    + Action expert sees progressively refined Qwen LLM context at sampled layer
      depths (pi0 benefit), instead of a static final-layer snapshot.
    + scene_feats are post-LLM features (not raw ViT output).
    + Keeps original flash-attn2 causal attention (no custom mask).
    + Restores Flower-style Stage 1 trajectory-to-language cross-attention.
"""

from contextlib import nullcontext

import einops
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import RmsNorm

from ..encoder.multimodal.encoder3d_qwen import (
    _enable_lora_params,
    _is_peft_wrapped,
    _set_trainable_lora_only,
    _unwrap_peft_base,
    density_based_sampler,
)
from ..encoder.vision.qwen3vl import Qwen3VLBackbone, Qwen3VLNormalize
from ..utils.layers import DoubleCrossSelfAttentionModule, AttentionModule
from ..utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def _build_backloaded_layer_schedule(
    num_qwen_layers,
    num_shared_attn_layers,
    bias_power=0.8,
):
    """
    Select a strictly increasing subset of Qwen layers for the action expert.

    Coverage stays close to uniform across the full decoder depth, while a
    mild concave bias (< 1.0) makes the spacing slightly denser near the last
    part of the stack where the language model is most task-specific.
    """
    num_qwen_layers = int(num_qwen_layers)
    num_shared_attn_layers = int(num_shared_attn_layers)

    if num_qwen_layers <= 0:
        raise ValueError(f"num_qwen_layers must be positive, got {num_qwen_layers}.")
    if num_shared_attn_layers <= 0:
        raise ValueError(
            f"num_shared_attn_layers must be positive, got {num_shared_attn_layers}."
        )
    if num_shared_attn_layers >= num_qwen_layers:
        return tuple(range(num_qwen_layers))
    if num_shared_attn_layers == 1:
        return (num_qwen_layers - 1,)

    layer_ids = []
    last_qwen_idx = num_qwen_layers - 1
    for shared_idx in range(num_shared_attn_layers):
        frac = shared_idx / float(num_shared_attn_layers - 1)
        target_qwen_idx = int(round((frac ** bias_power) * last_qwen_idx))

        min_allowed = 0 if shared_idx == 0 else layer_ids[-1] + 1
        remaining = num_shared_attn_layers - shared_idx - 1
        max_allowed = last_qwen_idx - remaining
        layer_ids.append(min(max(target_qwen_idx, min_allowed), max_allowed))

    layer_ids[0] = 0
    layer_ids[-1] = last_qwen_idx
    return tuple(layer_ids)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class QwenDualExpert3DFAActor(BaseDenoiseActor):
    """
    Hybrid 3DFA + Pi0 policy.  See module docstring for full description.
    """

    def __init__(
        self,
        # backbone
        backbone="qwen3vl",
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        fps_subsampling_factor=5,
        # embedding / attention
        embedding_dim=120,
        num_attn_heads=8,
        # history / bimanual
        nhist=3,
        nhand=1,
        # legacy args kept for config compatibility
        output_level="res3",
        upsample=False,
        num_vis_instr_attn_layers=2,
        # decoder depth over sampled Qwen layers
        num_shared_attn_layers=4,
        # trajectory
        relative=False,
        rotation_format="quat_xyzw",
        # denoising
        denoise_timesteps=100,
        denoise_model="ddpm",
        lv2_batch_size=1,
        # LoRA
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

        # Null out the base class defaults (we set our own below)
        self.encoder = None
        self.traj_encoder = None
        self.prediction_head = None

        self._nhist = nhist
        self._nhand = nhand
        self.scene_subsampling_factor = max(1, int(fps_subsampling_factor))
        self.finetune_backbone = finetune_backbone
        self.finetune_text_encoder = finetune_text_encoder
        self.num_finetune_vision_layers = num_finetune_vision_layers
        self._use_llm_lora = bool(llm_lora_rank > 0 and llm_lora_num_layers > 0)
        self._use_vision_lora = bool(vision_lora_rank > 0 and vision_lora_num_layers > 0)

        # ---------- Backbone (with merger, NOT stubbed) ----------
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
            use_multiscale=False,  # no multiscale in this policy
        )
        self._configure_backbone_requires_grad()

        # Keep both the possibly PEFT-wrapped LM (for forward) and the base LM
        # (for direct layer / helper access).
        self.language_model = self.backbone.vlm.language_model
        self.language_model_base = _unwrap_peft_base(self.language_model)
        n_qwen_layers = len(self.language_model_base.layers)
        self.shared_attn_qwen_layer_ids = _build_backloaded_layer_schedule(
            n_qwen_layers,
            num_shared_attn_layers,
        )
        self._shared_attn_qwen_layer_to_slot = {
            qwen_layer_idx: slot_idx
            for slot_idx, qwen_layer_idx in enumerate(self.shared_attn_qwen_layer_ids)
        }
        n_shared_layers = len(self.shared_attn_qwen_layer_ids)

        # ---------- Dimensions ----------
        self.hidden_size = self.backbone.llm_hidden_size   # Qwen LLM / merger out dim
        self.rot_dim = 3 if rotation_format in ("euler", "axisangle") else 6
        self.pose_dim = 3 + self.rot_dim
        self._embedding_dim = embedding_dim
        self._nhist_nhand = nhist * nhand

        # ---------- Per-selected-layer RGB / text bridges (llm_hidden_size → embedding_dim) ----------
        self.vis_norm_layers = nn.ModuleList(
            [RmsNorm(self.hidden_size) for _ in range(n_shared_layers)]
        )
        self.vis_proj_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, embedding_dim, bias=False) for _ in range(n_shared_layers)]
        )

        self.cond_norm_layers = nn.ModuleList(
            [RmsNorm(self.hidden_size) for _ in range(n_shared_layers)]
        )
        self.cond_linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, embedding_dim, bias=False) for _ in range(n_shared_layers)]
        )

        # ---------- 3D RoPE (shared for action tokens and prefix positions) ----------
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # ---------- Proprioception encoder (identical to Qwen3VLEncoder3D) ----------
        self.curr_gripper_embed = nn.Embedding(nhist * nhand, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            n_heads=num_attn_heads,
            rotary_pe=True,
            use_adaln=False,
            pre_norm=False,
        )

        # ---------- Action token encoding ----------
        self.traj_encoder = nn.Linear(
            6 if rotation_format in ("euler", "axisangle") else 9,
            embedding_dim,
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)   # step-index positional
        self.hand_embed = nn.Embedding(2, embedding_dim) if nhand > 1 else None

        # ---------- Language cross-attention (same role as 3DFA-VLA) ----------
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False,
        )

        # ---------- Time conditioning (3DFA-VLA style: AdaLN signal) ----------
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist * nhand, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ---------- Action expert layers (one per selected Qwen LLM layer) ----------
        # Each is DoubleCrossSelfAttentionModule with num_layers=1:
        #   cross-attn(action → refined_context_i) + self-attn(action → action) + FFN
        # All with 3D RoPE and AdaLN time conditioning.
        self.action_expert_layers = nn.ModuleList([
            DoubleCrossSelfAttentionModule(
                num_layers=1,
                d_model=embedding_dim,
                dim_fw=4 * embedding_dim,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe_0=True,   # cross-attn uses 3D RoPE
                rotary_pe_1=True,   # self-attn uses 3D RoPE
                use_adaln=True,     # AdaLN from time conditioning
            )
            for _ in range(n_shared_layers)
        ])

        # ---------- Output heads (identical to 3DFA-VLA TransformerHead) ----------
        # Position branch
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=True,
            rotary_pe_1=True,
            use_adaln=True,
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),
        )

        # Rotation branch
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=True,
            rotary_pe_1=True,
            use_adaln=True,
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.rot_dim),
        )

        # Openess branch (from position features)
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # ---------- Register permanent hooks ----------
        self._layer_hidden_states = {
            qwen_layer_idx: None for qwen_layer_idx in self.shared_attn_qwen_layer_ids
        }
        self._register_permanent_hooks()

    # ------------------------------------------------------------------
    # Backbone grad control (same pattern as other Qwen policies)
    # ------------------------------------------------------------------

    def _configure_backbone_requires_grad(self):
        vlm = self.backbone.vlm
        for p in vlm.parameters():
            p.requires_grad = False

        if self.finetune_backbone:
            for p in vlm.parameters():
                p.requires_grad = True
            return

        visual = vlm.visual
        visual_inner = _unwrap_peft_base(visual)

        if self.num_finetune_vision_layers > 0:
            n_blocks = len(visual_inner.blocks)
            if self.num_finetune_vision_layers >= n_blocks:
                for p in visual_inner.parameters():
                    p.requires_grad = True
            else:
                start = max(0, n_blocks - self.num_finetune_vision_layers)
                for block in visual_inner.blocks[start:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in visual_inner.merger.parameters():
                    p.requires_grad = True

        if self._use_vision_lora:
            if self.num_finetune_vision_layers == 0:
                _set_trainable_lora_only(visual)
            else:
                _enable_lora_params(visual)

        lm = vlm.language_model
        if self.finetune_text_encoder:
            for p in lm.parameters():
                p.requires_grad = True
        elif _is_peft_wrapped(lm):
            _set_trainable_lora_only(lm)

    def _backbone_visual_dtype(self):
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
        for p in visual_inner.parameters():
            return p.dtype
        return torch.float32

    def _llm_dtype(self):
        return self.language_model_base.embed_tokens.weight.dtype

    def _backbone_grad_context(self):
        needs_grad = (
            self.finetune_backbone
            or self.finetune_text_encoder
            or self.num_finetune_vision_layers > 0
            or self._use_vision_lora
            or self._use_llm_lora
        )
        return nullcontext() if needs_grad else torch.no_grad()

    # ------------------------------------------------------------------
    # Permanent hooks (registered once in __init__)
    # ------------------------------------------------------------------

    def _register_permanent_hooks(self):
        """Register forward hooks to capture only the selected LLM hidden states."""
        for qwen_layer_idx in self.shared_attn_qwen_layer_ids:
            layer = self.language_model_base.layers[qwen_layer_idx]

            def _make_layer_hook(idx):
                def hook(module, input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self._layer_hidden_states[idx] = hs

                return hook

            layer.register_forward_hook(_make_layer_hook(qwen_layer_idx))

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _encode_proprio(self, proprio, scene_feats, scene_xyz):
        """Proprioception encoding via cross-attention to scene (same as 3DFA-VLA).

        Args:
            proprio:     (B, nhist, nhand, D)
            scene_feats: (B, N_scene, embedding_dim)
            scene_xyz:   (B, N_scene, 3)

        Returns:
            proprio_feats: (B, nhist*nhand, embedding_dim)
        """
        B = proprio.shape[0]
        prop_flat = proprio.flatten(1, 2)  # (B, nhist*nhand, D)
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).expand(B, -1, -1)
        proprio_pos = self.relative_pe_layer(torch.nan_to_num(prop_flat[..., :3].float()))
        scene_pos = self.relative_pe_layer(scene_xyz.float())
        proprio_feats = self.gripper_context_head(
            proprio_feats, scene_feats,
            seq1_pos=proprio_pos, seq2_pos=scene_pos,
        )[-1]
        return proprio_feats

    def _run_fps(self, features, pos):
        """Density-based FPS subsampling of scene tokens."""
        if self.scene_subsampling_factor <= 1:
            return features, pos
        bs, npts, ch = features.shape
        inds = density_based_sampler(features, self.scene_subsampling_factor)
        exp = inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_feats = torch.gather(features, 1, exp)
        if pos is not None:
            exp3 = inds.unsqueeze(-1).expand(-1, -1, 3)
            sampled_pos = torch.gather(pos, 1, exp3)
        else:
            sampled_pos = None
        return sampled_feats, sampled_pos

    def _build_valid_mask_hist(self, valid_mask, nhand, nhist):
        """Match FlowerActor3D's masking of fake-hand proprio history."""
        if valid_mask is None:
            return None
        hand_valid = valid_mask[:, 0, :, 0]
        valid_mask_hist = hand_valid[:, None, :].expand(-1, nhist, -1)
        return valid_mask_hist.reshape(valid_mask.shape[0], nhist * nhand).unsqueeze(-1).float()

    def _encode_denoising_timestep(self, timestep, proprio_feats, valid_mask_hist=None):
        """Compute FlowerActor3D-style timestep + proprio AdaLN conditioning."""
        time_feats = self.time_emb(timestep.float())
        if valid_mask_hist is not None:
            proprio_feats = proprio_feats * valid_mask_hist
        proprio_flat = proprio_feats.flatten(1)
        gripper_feats = self.curr_gripper_emb(
            proprio_flat.to(self.curr_gripper_emb[0].weight.dtype)
        )
        return time_feats.to(gripper_feats.dtype) + gripper_feats

    def _project_scene_tokens(self, hidden_states, shared_layer_slot):
        vis_norm = self.vis_norm_layers[shared_layer_slot]
        vis_proj = self.vis_proj_layers[shared_layer_slot]
        return vis_proj(vis_norm(hidden_states.to(vis_proj.weight.dtype)))

    def _project_text_tokens(self, hidden_states, shared_layer_slot, text_mask=None):
        if text_mask is not None:
            hidden_states = hidden_states * text_mask.unsqueeze(-1)
        cond_norm = self.cond_norm_layers[shared_layer_slot]
        cond_linear = self.cond_linear_layers[shared_layer_slot]
        return cond_linear(cond_norm(hidden_states.to(cond_linear.weight.dtype)))

    @staticmethod
    def _build_context_key_padding_mask(scene_feats, instr_key_padding_mask):
        if instr_key_padding_mask is None:
            return None
        scene_mask = torch.zeros(
            scene_feats.shape[:2],
            dtype=torch.bool,
            device=scene_feats.device,
        )
        return torch.cat([scene_mask, instr_key_padding_mask], dim=1)

    # ------------------------------------------------------------------
    # encode_inputs (runs once per observation)
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
        """Encode scene, instruction, and proprioception.

        Runs Qwen3-VL ViT + merger + LLM once (original causal flash-attn2 path).
        Captures per-layer LLM hidden states via registered hooks.

        scene_feats and instr_feats are both extracted from the **final LLM
        hidden states** (post-LLM).

        Returns:
            dict with keys: query_trajectory, scene_feats, scene_xyz,
                instr_feats, instr_xyz, proprio_feats, per_layer_scene_states,
                per_layer_instr_states, per_layer_proprio_feats.
        """
        B, ncam, C, H, W = rgb3d.shape
        dev = rgb3d.device

        # Spatial dimensions
        hp = H // self.backbone.patch_size
        wp = W // self.backbone.patch_size
        ms = self.backbone.spatial_merge_size
        h32, w32 = hp // ms, wp // ms
        merged_tok = h32 * w32

        # Normalize + prepare pixel_values
        rgb3d_normed = self.normalize(rgb3d)
        visual_dtype = self._backbone_visual_dtype()
        if rgb3d_normed.dtype != visual_dtype:
            rgb3d_normed = rgb3d_normed.to(visual_dtype)
        pixel_values = self.backbone.images_to_pixel_values(
            rgb3d_normed.reshape(B * ncam, C, H, W)
        ).to(visual_dtype)
        grid_thw = torch.tensor(
            [1, hp, wp], dtype=torch.long, device=dev
        ).view(1, 3).expand(B * ncam, -1)

        # Build input_ids and layout metadata
        input_ids, instr_start, text_len = self.backbone.build_input_ids(
            instruction, ncam, merged_tok, dev
        )
        seq_len = input_ids.shape[1]

        # 2D attention mask (1=real, 0=pad) — used only for position_ids computation
        if instr_mask is not None:
            full_attn_mask_2d = torch.ones(
                B, seq_len, dtype=instr_mask.dtype, device=dev,
            )
            full_attn_mask_2d[:, instr_start : instr_start + text_len] = instr_mask.to(dev)
        else:
            full_attn_mask_2d = None

        # M-RoPE position IDs (native Qwen format, needs 2D mask)
        position_ids, rope_deltas = self.backbone._get_native_position_ids_cached(
            input_ids, grid_thw, full_attn_mask_2d, instr_mask,
            ncam, hp, wp, text_valid_lens=instr_lens,
        )
        self.backbone.vlm.rope_deltas = rope_deltas

        # Build multimodal input embeddings manually.
        llm_dtype = self._llm_dtype()
        inputs_embeds = self.backbone.vlm.get_input_embeddings()(input_ids)

        image_embeds, deepstack_image_embeds = self.backbone.vlm.get_image_features(
            pixel_values, grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.backbone.vlm.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds

        if full_attn_mask_2d is None:
            base_attn_mask = torch.ones(B, seq_len, device=dev, dtype=torch.long)
        else:
            base_attn_mask = full_attn_mask_2d.to(dtype=torch.long)
        full_attn_mask_2d = base_attn_mask

        # Reset hook captures before forward
        for qwen_layer_idx in self._layer_hidden_states:
            self._layer_hidden_states[qwen_layer_idx] = None

        # Capture post-deepstack states for early layers.
        n_layers = len(self.language_model_base.layers)
        post_deepstack_states = {
            qwen_layer_idx: None for qwen_layer_idx in self.shared_attn_qwen_layer_ids
        }
        deepstack_counter = {"idx": 0}
        orig_deepstack_process = self.language_model_base._deepstack_process

        def _wrapped_deepstack_process(hidden_states, visual_pos_masks_, visual_embeds_):
            out = orig_deepstack_process(hidden_states, visual_pos_masks_, visual_embeds_)
            idx = deepstack_counter["idx"]
            if idx in post_deepstack_states:
                post_deepstack_states[idx] = out
            deepstack_counter["idx"] = idx + 1
            return out

        self.language_model_base._deepstack_process = _wrapped_deepstack_process
        try:
            with self._backbone_grad_context():
                outputs = self.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attn_mask_2d,
                    position_ids=position_ids,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
        finally:
            self.language_model_base._deepstack_process = orig_deepstack_process

        # ---------- PCD at stride-32 grid ----------
        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (h32, w32),
            mode="bilinear",
        )
        scene_xyz = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=ncam
        )
        scene_xyz = torch.nan_to_num(scene_xyz.float())  # (B, N_scene, 3)

        # ---------- Post-LLM visual features ----------
        # Extract final hidden states at image-token positions.  These are richer
        # than pre-LLM post-merge features because they passed through the full LLM.
        last_hidden = outputs.last_hidden_state  # (B, seq_len, llm_hidden_size)
        final_shared_layer_slot = self._shared_attn_qwen_layer_to_slot[n_layers - 1]
        image_token_mask = (input_ids == self.backbone.image_token_id)  # (B, seq_len)
        # All samples share the same chat-template layout → same image positions
        img_positions = image_token_mask[0].nonzero(as_tuple=False).squeeze(-1)
        post_llm_visual = last_hidden[:, img_positions, :]  # (B, N_scene, hidden_size)
        scene_feats = self._project_scene_tokens(post_llm_visual, final_shared_layer_slot)
        # (B, N_scene, embedding_dim)

        # ---------- Post-LLM instruction features ----------
        text_hidden = last_hidden[:, instr_start : instr_start + text_len, :]
        tm = None
        instr_key_padding_mask = None
        if instr_mask is not None:
            tm = instr_mask.to(device=text_hidden.device, dtype=text_hidden.dtype)
            instr_key_padding_mask = ~instr_mask.to(device=text_hidden.device, dtype=torch.bool)
        instr_feats = self._project_text_tokens(text_hidden, final_shared_layer_slot, tm)
        # (B, L, embedding_dim)

        # Instruction 3D position = gripper anchor (per-token same anchor)
        text_anchor = torch.nan_to_num(
            proprio[:, -1, :, :3].mean(dim=1, keepdim=True).float()
        )  # (B, 1, 3)
        instr_xyz = text_anchor.expand(-1, text_len, -1)  # (B, L, 3)

        # ---------- Per-layer scene states ----------
        # Layer hooks capture raw decoder outputs; for early layers where deepstack
        # is applied, replace with post-deepstack states captured above.
        per_layer_scene_states = []
        per_layer_instr_states = []
        per_layer_proprio_feats = []
        for shared_layer_slot, qwen_layer_idx in enumerate(self.shared_attn_qwen_layer_ids):
            hs = self._layer_hidden_states[qwen_layer_idx]
            if post_deepstack_states[qwen_layer_idx] is not None:
                hs = post_deepstack_states[qwen_layer_idx]
            if hs is None:
                raise RuntimeError(
                    f"Missing Qwen hidden states for selected layer {qwen_layer_idx}."
                )
            layer_visual = hs[:, img_positions, :]
            layer_scene_feats = self._project_scene_tokens(layer_visual, shared_layer_slot)
            layer_text_hidden = hs[:, instr_start : instr_start + text_len, :]
            layer_instr_feats = self._project_text_tokens(
                layer_text_hidden,
                shared_layer_slot,
                tm,
            )
            per_layer_scene_states.append(layer_scene_feats)
            per_layer_instr_states.append(layer_instr_feats)
            per_layer_proprio_feats.append(
                self._encode_proprio(proprio, layer_scene_feats, scene_xyz)
            )

        # ---------- Proprioception encoding ----------
        proprio_feats = self._encode_proprio(proprio, scene_feats, scene_xyz)

        return {
            "query_trajectory": proprio[:, -1:],           # (B, 1, nhand, D)
            "proprio_feats": proprio_feats,                 # (B, nhist*nhand, emb)
            "scene_feats": scene_feats,                     # (B, N_scene, emb)
            "scene_xyz": scene_xyz,                         # (B, N_scene, 3)
            "instr_feats": instr_feats,                     # (B, L, emb)
            "instr_xyz": instr_xyz,                         # (B, L, 3)
            "instr_key_padding_mask": instr_key_padding_mask,      # (B, L) bool, True=pad
            "per_layer_scene_states": per_layer_scene_states,    # list[(B, N_scene, emb)]
            "per_layer_instr_states": per_layer_instr_states,    # list[(B, L, emb)]
            "per_layer_proprio_feats": per_layer_proprio_feats,  # list[(B, nhist*nhand, emb)]
        }

    # ------------------------------------------------------------------
    # policy_forward_pass (runs at every denoising step)
    # ------------------------------------------------------------------

    def _predict_pos(
        self,
        traj_feats, context_feats,
        rel_traj_pos, rel_context_pos,
        time_embs, traj_len,
        token_mask_float=None, token_mask_bool=None,
        context_key_padding_mask=None,
    ):
        position_features = self.position_self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_context_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs,
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)
        position = self.position_predictor(position_features)
        return position, position_features

    def _predict_rot(
        self,
        traj_feats, context_feats,
        rel_traj_pos, rel_context_pos,
        time_embs, traj_len,
        token_mask_float=None, token_mask_bool=None,
        context_key_padding_mask=None,
    ):
        rotation_features = self.rotation_self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_context_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs,
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        """Denoising forward pass.

        Args:
            trajectory:   (B, T, nhand, pose_dim) normalized noisy trajectory
            timestep:     (B,) denoising timestep
            fixed_inputs: dict from encode_inputs
            valid_mask:   (B, T, nhand, 1) float, 1=real, 0=padded hand

        Returns:
            list of length 1 with (B, T, nhand, pose_dim+1) predictions
        """
        B, traj_len, nhand, _ = trajectory.shape

        # ---------- Action tokens ----------
        traj_feats = self.traj_encoder(trajectory)  # (B, T, nhand, emb)
        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]

        # Sinusoidal step-index positional embedding (temporal ordering)
        # Shape: (traj_len, emb) → (1, traj_len, 1, emb) → (B, traj_len, nhand, emb)
        traj_time_pos = self.traj_time_emb(
            torch.arange(traj_len, device=trajectory.device, dtype=torch.float32)
        ).unsqueeze(0).unsqueeze(2).expand(B, -1, nhand, -1)
        traj_time_pos_flat = einops.rearrange(traj_time_pos, "b l h c -> b (l h) c")
        traj_feats = einops.rearrange(traj_feats, "b l h c -> b (l h) c")

        # Unnormalized xyz for 3D RoPE
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:
            anchor = fixed_inputs["query_trajectory"][..., :3]
            traj_xyz = anchor + torch.cumsum(traj_xyz, dim=1)
        traj_xyz = einops.rearrange(
            torch.nan_to_num(traj_xyz.float()), "b l h c -> b (l h) c"
        )

        # Token mask (for bimanual padding)
        if valid_mask is not None:
            token_mask_float = valid_mask.squeeze(-1).reshape(B, -1).float()
            token_mask_bool = token_mask_float == 0
        else:
            token_mask_float = None
            token_mask_bool = None
        valid_mask_hist = self._build_valid_mask_hist(valid_mask, nhand, self._nhist)

        # ---------- 3D RoPE: action tokens ----------
        rel_traj_pos = self.relative_pe_layer(traj_xyz)  # (B, T*nhand, emb, 2)

        # ---------- Language cross-attention ----------
        instr_feats = fixed_inputs["instr_feats"]
        instr_key_padding_mask = fixed_inputs["instr_key_padding_mask"]
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_token_mask=token_mask_float,
            seq2_key_padding_mask=instr_key_padding_mask,
            seq1_sem_pos=traj_time_pos_flat,
            seq2_sem_pos=None,
        )[-1]
        traj_feats = traj_feats + traj_time_pos_flat

        # ---------- Main stage: action expert at each Qwen layer ----------
        scene_xyz = fixed_inputs["scene_xyz"].float()
        instr_feats = fixed_inputs["instr_feats"]
        instr_xyz = fixed_inputs["instr_xyz"].float()
        per_layer_scene_states = fixed_inputs["per_layer_scene_states"]
        per_layer_instr_states = fixed_inputs["per_layer_instr_states"]
        per_layer_proprio_feats = fixed_inputs["per_layer_proprio_feats"]
        for expert_layer, scene_feats_i, instr_feats_i, proprio_feats_i in zip(
            self.action_expert_layers,
            per_layer_scene_states,
            per_layer_instr_states,
            per_layer_proprio_feats,
        ):
            shared_time_embs = self._encode_denoising_timestep(
                timestep, proprio_feats_i, valid_mask_hist
            )
            context_feats = torch.cat([scene_feats_i, instr_feats_i], dim=1)
            context_key_padding_mask = self._build_context_key_padding_mask(
                scene_feats_i, instr_key_padding_mask
            )
            context_xyz = torch.cat([scene_xyz, instr_xyz], dim=1)
            rel_context_pos = self.relative_pe_layer(context_xyz)
            traj_feats = expert_layer(
                seq=traj_feats,
                seq0=context_feats,
                seq_token_mask=token_mask_float,
                seq_token_mask_bool=token_mask_bool,
                seq0_key_padding_mask=context_key_padding_mask,
                seq_pos_0=rel_traj_pos,
                seq0_pos=rel_context_pos,
                seq_pos_1=rel_traj_pos,
                ada_sgnl=shared_time_embs,
            )[-1]

        # ---------- Output heads: final RGB + text context ----------
        scene_feats = fixed_inputs["scene_feats"]
        context_feats = torch.cat([scene_feats, instr_feats], dim=1)
        context_key_padding_mask = self._build_context_key_padding_mask(
            scene_feats, instr_key_padding_mask
        )
        context_xyz = torch.cat([scene_xyz, instr_xyz], dim=1)
        rel_context_pos = self.relative_pe_layer(context_xyz)
        final_time_embs = self._encode_denoising_timestep(
            timestep, fixed_inputs["proprio_feats"], valid_mask_hist
        )

        # Rotation head
        rotation = self._predict_rot(
            traj_feats, context_feats,
            rel_traj_pos, rel_context_pos,
            final_time_embs, traj_feats.shape[1],
            token_mask_float, token_mask_bool,
            context_key_padding_mask,
        )

        # Position head
        position, position_features = self._predict_pos(
            traj_feats, context_feats,
            rel_traj_pos, rel_context_pos,
            final_time_embs, traj_feats.shape[1],
            token_mask_float, token_mask_bool,
            context_key_padding_mask,
        )

        # Openess from position branch
        openess = self.openess_predictor(position_features)

        pred = torch.cat([position, rotation, openess], dim=-1)
        pred = pred.unflatten(1, (traj_len, nhand))
        return [pred]
