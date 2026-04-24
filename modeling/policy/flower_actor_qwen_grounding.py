"""
Grounding variants of QwenFlowerActor3D. Two parallel classes:

  - QwenFlowerActor3DGroundingCond  (model_type: qwen_flower_actor_3d_grounding_cond)
    Takes an optional 3D contact point as an additional context token in the
    action decoder. The point is projected to embedding_dim, placed at its own
    3D RoPE position, and concatenated into the scene+language context before
    the shared DoubleCrossSelf attention.  When grounding_3d is None the model
    behaves identically to qwen_flower_actor_3d.

  - QwenFlowerActor3DGroundingPred  (model_type: qwen_flower_actor_3d_grounding_pred)
    Predicts the 3D contact point as an auxiliary training objective from the
    SHARED decoder features (mean-pooled traj_feats after self_attn), so the
    L1 grounding loss backpropagates through all shared attention layers used
    by every action head.  When grounding_3d is None the auxiliary loss is
    skipped.  Inference is unchanged.

Both variants leave existing code untouched (new file + new model_type entries).

Additional variant:
  - QwenFlowerActor3DGroundingCondLLMVisual
    (model_type: qwen_flower_actor_3d_grounding_cond_llm_visual)
    Same as QwenFlowerActor3DGroundingCond, but scene features come from
    final Qwen LLM visual token states (stride 32) rather than pre-merge
    ViT features (stride 16).

  - QwenFlowerActor3DGroundingCondQwen3D
    (model_type: qwen_flower_actor_3d_grounding_cond_qwen3d)
    Keeps the same grounding-conditioned Flower decoder, but replaces the
    encoder with a Qwen3D-style contextualizer:
      post-merger stride-32 scene tokens -> 3D lift with PCD -> manual Qwen
      prefix attention with bidirectional masking + 4D TXYZ RoPE.
"""

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from .flower_actor import TransformerHead
from ..encoder.multimodal.encoder3d_qwen import Qwen3VLEncoder3D
try:
    from ..encoder.multimodal.encoder3d_qwen3d import Qwen3VLEncoderQwen3D
    _QWEN3D_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    Qwen3VLEncoderQwen3D = None
    _QWEN3D_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Variant 1 – Grounding Conditioning
# ---------------------------------------------------------------------------

class GroundingCondTransformerHead(TransformerHead):
    """
    Extends TransformerHead to accept an optional grounding token.

    When grounding_feats (B, 1, F) and grounding_pos (B, 1, 3) are provided,
    the grounding token is concatenated into the scene+language context before
    shared attention, with its world-frame xyz used as the 3D RoPE position.
    When both are None the forward is identical to the parent class.
    """

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None,
                grounding_feats=None, grounding_pos=None,
                grounding_key_padding_mask=None):
        """
        Extra args vs TransformerHead.forward:
            grounding_feats: (B, 1, F) or None
            grounding_pos:   (B, 1, 3) world-frame xyz, or None
        """
        B, traj_len, nhand, _ = trajectory.shape

        # --- trajectory embeddings ---
        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        if valid_mask is not None:
            token_mask = valid_mask.squeeze(-1).reshape(B, -1)
            token_mask_float = token_mask.float()
            token_mask_bool = token_mask == 0
        else:
            token_mask_float = None
            token_mask_bool = None

        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(B, 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_token_mask=token_mask_float,
            seq1_sem_pos=traj_time_pos,
            seq2_sem_pos=None,
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        valid_mask_hist = None
        if valid_mask is not None:
            hand_valid = valid_mask[:, 0, :, 0]       # (B, nhand)
            valid_mask_hist = hand_valid[:, None, :]  # (B, 1, nhand)
            nhist = proprio_feats.shape[1] // nhand
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)
            valid_mask_hist = (
                valid_mask_hist.reshape(B, nhist * nhand).unsqueeze(-1).float()
            )
        time_embs = self.encode_denoising_timestep(timesteps, proprio_feats, valid_mask_hist)

        # --- build context, optionally injecting grounding token ---
        context_feats = torch.cat((rgb3d_feats, instr_feats), dim=1)
        context_pos = torch.cat((rgb3d_pos, instr_pos), dim=1)
        context_key_padding_mask = None
        if grounding_feats is not None and grounding_pos is not None:
            context_feats = torch.cat((context_feats, grounding_feats), dim=1)
            context_pos = torch.cat((context_pos, grounding_pos), dim=1)
            if grounding_key_padding_mask is not None:
                prefix_mask = torch.zeros(
                    context_feats.shape[0],
                    context_feats.shape[1] - grounding_key_padding_mask.shape[1],
                    dtype=torch.bool,
                    device=context_feats.device,
                )
                context_key_padding_mask = torch.cat(
                    (prefix_mask, grounding_key_padding_mask.to(dtype=torch.bool)),
                    dim=1,
                )

        # --- 3D RoPE ---
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(context_pos)

        # --- shared DoubleCrossSelf attention ---
        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs,
        )[-1]

        # --- output heads ---
        rotation = self.predict_rot(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
        )
        position, position_features = self.predict_pos(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
        )
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), dim=-1)
                 .unflatten(1, (traj_len, nhand))
        ]


class QwenFlowerActor3DGroundingCond(BaseDenoiseActor):
    """
    Grounding-conditioned policy. Identical to qwen_flower_actor_3d except:
      - Adds a grounding_proj MLP to embed the 3D contact point.
      - Injects the grounding token into the action decoder's context when
        grounding_3d is provided.
      - grounding_3d is threaded through forward -> encode_inputs via a
        transient instance attribute; no changes to compute_loss or
        compute_trajectory are needed.
    """

    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,
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
        llm_lora_target_modules=(
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
        ),
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

        self.encoder = Qwen3VLEncoder3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
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

        self.prediction_head = GroundingCondTransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6,
        )
        if nhand == 1:
            self.prediction_head.hand_embed = None

        # Projects 3D contact point (B, 3) -> (B, 1, embedding_dim)
        self.grounding_proj = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Transient storage used to thread grounding_3d into encode_inputs
        # without changing the compute_loss / compute_trajectory signatures.
        self._pending_grounding_3d = None
        self._pending_grounding_keep_mask = None

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False,
        instr_mask=None,
        instr_lens=None,
        grounding_3d=None,
        grounding_keep_mask=None,
        **kwargs,
    ):
        """Identical to BaseDenoiseActor.forward but accepts grounding_3d."""
        self._pending_grounding_3d = grounding_3d
        self._pending_grounding_keep_mask = grounding_keep_mask
        try:
            return super().forward(
                gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
                instruction, proprio,
                run_inference=run_inference,
                instr_mask=instr_mask,
                instr_lens=instr_lens,
            )
        finally:
            self._pending_grounding_3d = None
            self._pending_grounding_keep_mask = None

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
        """Calls the base encoder, then appends (grounding_feats, grounding_pos)."""
        base_result = super().encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )
        grounding_3d = self._pending_grounding_3d
        grounding_keep_mask = self._pending_grounding_keep_mask
        if grounding_3d is not None:
            # grounding_feats: (B, 1, F)  grounding_pos: (B, 1, 3)
            grounding_feats = self.grounding_proj(grounding_3d).unsqueeze(1)
            grounding_pos = grounding_3d.unsqueeze(1)
            grounding_key_padding_mask = None
            if grounding_keep_mask is not None:
                grounding_key_padding_mask = (~grounding_keep_mask.bool()).view(-1, 1)
        else:
            grounding_feats = None
            grounding_pos = None
            grounding_key_padding_mask = None
        return base_result + (grounding_feats, grounding_pos, grounding_key_padding_mask)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            grounding_feats, grounding_pos, grounding_key_padding_mask,
        ) = fixed_inputs

        trajectory_feats = self.traj_encoder(trajectory)
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(traj_xyz, dim=1)
            )

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos,
            valid_mask=valid_mask,
            grounding_feats=grounding_feats,
            grounding_pos=grounding_pos,
            grounding_key_padding_mask=grounding_key_padding_mask,
        )


class QwenFlowerActor3DGroundingCondLLMVisual(QwenFlowerActor3DGroundingCond):
    """
    Grounding-conditioned 3DFA policy using post-LLM visual tokens.

    This variant keeps the same decoder and grounding-token conditioning as
    ``QwenFlowerActor3DGroundingCond``. The only change is in the encoder:
    scene RGB features are taken from the final Qwen LLM hidden states at
    image-token positions (patch-size 32), following the native Qwen3VL
    multimodal forward path before splitting visual/text features.
    """

    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,
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
        llm_lora_target_modules=(
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
        ),
        gradient_checkpointing=False,
        use_multiscale=False,
    ):
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
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

        # Override encoder only: use final LLM visual tokens as scene features.
        self.encoder = Qwen3VLEncoder3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
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
            use_post_llm_visual=True,
        )


class QwenFlowerActor3DGroundingCondQwen3D(QwenFlowerActor3DGroundingCond):
    """
    Grounding-conditioned 3DFA policy using Qwen3D-style scene contextualization.

    This variant preserves the existing Flower grounding head and contact-point
    conditioning token. Only the encoder changes: scene tokens are taken from
    native Qwen stride-32 post-merger features, lifted to 3D with the PCD, and
    then jointly contextualized with the chat/instruction prefix using full
    bidirectional Qwen attention plus custom 4D TXYZ RoPE.
    """

    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,
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
        llm_lora_target_modules=(
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
        ),
        gradient_checkpointing=False,
        use_multiscale=False,
    ):
        if Qwen3VLEncoderQwen3D is None:
            raise ImportError(
                "qwen_flower_actor_3d_grounding_cond_qwen3d requires "
                "`transformers.models.qwen3_vl`. Please update/install a "
                "compatible transformers build to use this model type."
            ) from _QWEN3D_IMPORT_ERROR
        if use_multiscale:
            raise ValueError(
                "QwenFlowerActor3DGroundingCondQwen3D only supports use_multiscale=False."
            )
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
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

        self.encoder = Qwen3VLEncoderQwen3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
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


# ---------------------------------------------------------------------------
# Variant 2 – Grounding Prediction (auxiliary loss)
# ---------------------------------------------------------------------------

class GroundingPredTransformerHead(TransformerHead):
    """
    Extends TransformerHead with an auxiliary contact-point predictor.

    After the shared self_attn block, the trajectory features are mean-pooled
    to (B, F) and passed through contact_point_predictor to predict (B, 3).
    The prediction is stored as self._last_contact_pred so the actor can
    retrieve it after the forward pass without changing the return type.

    Return type is unchanged: list of (B, T, H, action_dim) tensors.
    """

    def __init__(
        self,
        embedding_dim=60,
        num_attn_heads=8,
        num_shared_attn_layers=4,
        nhist=3,
        rotary_pe=True,
        rot_dim=6,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            nhist=nhist,
            rotary_pe=rotary_pe,
            rot_dim=rot_dim,
        )
        self.contact_point_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),
        )
        self._last_contact_pred = None  # set each forward pass

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None,
                **kwargs):
        B, traj_len, nhand, _ = trajectory.shape

        # --- trajectory embeddings ---
        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        if valid_mask is not None:
            token_mask = valid_mask.squeeze(-1).reshape(B, -1)
            token_mask_float = token_mask.float()
            token_mask_bool = token_mask == 0
        else:
            token_mask_float = None
            token_mask_bool = None

        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(B, 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_token_mask=token_mask_float,
            seq1_sem_pos=traj_time_pos,
            seq2_sem_pos=None,
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        valid_mask_hist = None
        if valid_mask is not None:
            hand_valid = valid_mask[:, 0, :, 0]
            valid_mask_hist = hand_valid[:, None, :]
            nhist = proprio_feats.shape[1] // nhand
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)
            valid_mask_hist = (
                valid_mask_hist.reshape(B, nhist * nhand).unsqueeze(-1).float()
            )
        time_embs = self.encode_denoising_timestep(timesteps, proprio_feats, valid_mask_hist)

        # --- context = scene + language ---
        context_feats = torch.cat((rgb3d_feats, instr_feats), dim=1)
        context_pos = torch.cat((rgb3d_pos, instr_pos), dim=1)

        # --- 3D RoPE ---
        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(context_pos)

        # --- shared DoubleCrossSelf attention ---
        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs,
        )[-1]

        # --- auxiliary contact prediction (stored for actor to retrieve) ---
        # Mean-pool trajectory tokens; gradients flow back through self_attn
        # and all upstream components used by every action head.
        pooled = traj_feats.mean(dim=1)              # (B, F)
        self._last_contact_pred = self.contact_point_predictor(pooled)  # (B, 3)

        # --- output heads (unchanged) ---
        rotation = self.predict_rot(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
        )
        position, position_features = self.predict_pos(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
        )
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), dim=-1)
                 .unflatten(1, (traj_len, nhand))
        ]


class QwenFlowerActor3DGroundingPred(BaseDenoiseActor):
    """
    Policy with 3D contact-point prediction as an auxiliary training objective.

    The contact point is predicted from the mean-pooled trajectory features
    AFTER the shared decoder attention (GroundingPredTransformerHead), so the
    L1 grounding loss directly improves the shared representations that all
    action heads consume.

    When grounding_3d is None the auxiliary loss is skipped and the model
    trains exactly like qwen_flower_actor_3d.  Inference is unchanged.
    """

    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,
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
        llm_lora_target_modules=(
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
        ),
        gradient_checkpointing=False,
        use_multiscale=False,
        grounding_loss_weight=10.0,
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

        self.encoder = Qwen3VLEncoder3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
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

        self.prediction_head = GroundingPredTransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6,
        )
        if nhand == 1:
            self.prediction_head.hand_embed = None

        self.grounding_loss_weight = grounding_loss_weight

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False,
        instr_mask=None,
        instr_lens=None,
        grounding_3d=None,
        **kwargs,
    ):
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio,
                instr_mask=instr_mask,
                instr_lens=instr_lens,
            )
        return self.compute_loss(
            gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
            instruction, proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
            grounding_3d=grounding_3d,
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
        ) = fixed_inputs

        trajectory_feats = self.traj_encoder(trajectory)
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(traj_xyz, dim=1)
            )

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos,
            valid_mask=valid_mask,
        )

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
        grounding_3d=None,
    ):
        """
        Computes action loss (same as base) plus an optional auxiliary L1 loss
        on the predicted 3D contact point.

        The contact prediction is taken from the head after the last lv2
        forward pass (predictions are deterministic given fixed_inputs, so
        averaging across lv2 iterations would be redundant).
        """
        # --- action loss (identical to BaseDenoiseActor.compute_loss) ---
        action_loss = super().compute_loss(
            gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
            instruction, proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )

        # --- auxiliary grounding loss ---
        if grounding_3d is not None:
            contact_pred = self.prediction_head._last_contact_pred
            if contact_pred is not None:
                grounding_loss = F.l1_loss(contact_pred, grounding_3d)
                return action_loss + self.grounding_loss_weight * grounding_loss

        return action_loss
