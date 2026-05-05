from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch import nn
import einops

from motion_expert.config import load_motion_expert_config
from motion_expert.data import build_regular_query_uv
from motion_expert.model import MotionExpertModel

from ..encoder.multimodal.encoder3d import Encoder as Encoder3D
from ..encoder.multimodal.encoder2d import Encoder as Encoder2D
from ..utils.flash_attention import FlashMultiheadAttention
from ..utils.multihead_custom_attention import MultiheadCustomAttention
from ..utils.layers import DoubleCrossSelfAttentionModule, AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding3D

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


def _largest_divisor_at_most(value, limit):
    limit = max(1, min(int(value), int(limit)))
    while value % limit != 0:
        limit -= 1
    return limit


class CrossAttention3D(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, dropout=0.1, flash_attn_name="CrossAttention3D"):
        super().__init__()
        self.q_norm = nn.LayerNorm(embedding_dim)
        self.kv_norm = nn.LayerNorm(embedding_dim)
        self.attn = MultiheadCustomAttention(
            embedding_dim,
            num_attn_heads,
            dropout=dropout,
            use_flash_attn=True,
            flash_attn_name=flash_attn_name,
        )
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

    def forward(self, query_tokens, memory_tokens, query_pos, memory_pos):
        query_pos = torch.nan_to_num(query_pos.float(), nan=0.0, posinf=0.0, neginf=0.0)
        memory_pos = torch.nan_to_num(memory_pos.float(), nan=0.0, posinf=0.0, neginf=0.0)
        query_rope = self.relative_pe_layer(query_pos)
        memory_rope = self.relative_pe_layer(memory_pos)
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(memory_tokens).to(q.dtype)
        out = self.attn(
            query=q.transpose(0, 1),
            key=kv.transpose(0, 1),
            value=kv.transpose(0, 1),
            rotary_pe=(query_rope, memory_rope),
        )[0]
        return out.transpose(0, 1)


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_attn_heads, dropout=0.1):
        super().__init__()
        pool_heads = _largest_divisor_at_most(hidden_dim, num_attn_heads)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = FlashMultiheadAttention(
            hidden_dim,
            pool_heads,
            dropout=dropout,
            batch_first=True,
            flash_attn_name="TemporalAttentionPool",
        )

    def forward(self, motion_hidden):
        batch_size, num_points, num_steps, hidden_dim = motion_hidden.shape
        tokens = motion_hidden.reshape(batch_size * num_points, num_steps, hidden_dim)
        tokens = self.norm(tokens)
        query = self.query.expand(batch_size * num_points, -1, -1).to(tokens.dtype)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return pooled.squeeze(1).reshape(batch_size, num_points, hidden_dim)


class FlowerActor3D(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 num_finetune_vision_layers=0,  # Qwen-only; accepted but ignored for Florence2
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
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
            lv2_batch_size=lv2_batch_size
        )
        # Vision-language encoder, runs only once
        self.encoder = Encoder3D(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6  # NOTE: could change to always 6d
        )


class FlowerActor3DMotionLatent(FlowerActor3D):
    def __init__(self,
                 # Encoder arguments
                 backbone="florence2",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 num_finetune_vision_layers=0,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Motion latent arguments
                 motion_expert_config=None,
                 motion_expert_checkpoint=None,
                 motion_solver_steps=10,
                 motion_steps_to_run=1,
                 motion_finetune=True,
                 motion_scene_fusion=True,
                 motion_action_fusion=True,
                 # Training arguments
                 lv2_batch_size=1):
        if backbone != "florence2":
            raise ValueError(
                f"{type(self).__name__} currently supports backbone='florence2', got {backbone!r}."
            )
        if motion_expert_config is None:
            raise ValueError("flower_actor_3d_motion_latent requires --motion_expert_config.")
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
            lv2_batch_size=lv2_batch_size
        )

        motion_config = load_motion_expert_config(str(motion_expert_config))
        if str(motion_config.backbone_type).lower() != "florence2":
            raise ValueError(
                "flower_actor_3d_motion_latent expects a Florence2 Motion Expert config, "
                f"got backbone_type={motion_config.backbone_type!r}."
            )
        motion_feature_dim = int(self.encoder.cond_linear.in_features)
        max_conditioning_tokens = (
            motion_config.num_query_tokens
            + int(motion_config.sparse_decoder.visual_global_tokens)
            + motion_config.text_max_length
        )
        self.motion_expert = MotionExpertModel(
            config=motion_config,
            visual_input_dim=motion_feature_dim,
            text_input_dim=motion_feature_dim,
            max_conditioning_tokens=max_conditioning_tokens,
        )
        self.motion_config = motion_config
        self.motion_solver_steps = int(motion_solver_steps)
        self.motion_steps_to_run = int(motion_steps_to_run)
        self.motion_finetune = bool(motion_finetune)
        self.motion_scene_fusion = bool(motion_scene_fusion)

        self.motion_temporal_pool = TemporalAttentionPool(
            motion_config.sparse_decoder.hidden_dim,
            num_attn_heads,
            dropout=motion_config.dropout,
        )
        self.motion_proj = nn.Linear(motion_config.sparse_decoder.hidden_dim, embedding_dim)
        if self.motion_scene_fusion:
            self.scene_motion_attn = CrossAttention3D(
                embedding_dim,
                num_attn_heads,
                flash_attn_name="scene-motion CrossAttention3D",
            )
            self.alpha_scene = nn.Parameter(torch.zeros(()))
        else:
            self.scene_motion_attn = None
            self.alpha_scene = None
        self.prediction_head = MotionLatentTransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * 2,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6,
            motion_action_fusion=motion_action_fusion,
        )

        if motion_expert_checkpoint is not None:
            self._load_motion_expert_checkpoint(str(motion_expert_checkpoint))
        if not self.motion_finetune:
            for param in self.motion_expert.parameters():
                param.requires_grad = False

    def _load_motion_expert_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        cleaned_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                cleaned_state[key[len("model."):]] = value
            elif key.startswith((
                "point_scalar_proj.",
                "visual_proj.",
                "text_proj.",
                "point_norm.",
                "point_type_embed",
                "global_visual_type_embed",
                "text_type_embed",
                "point_encoder.",
                "flow_decoder.",
                "query_xyz_center",
                "query_xyz_scale",
                "delta_range_xyz",
                "query_centered_xyz_center",
                "query_centered_xyz_scale",
                "traj_delta_xyz_center",
                "traj_delta_xyz_scale",
                "use_calvin_percentile_normalization",
            )):
                cleaned_state[key] = value
        if not cleaned_state:
            raise ValueError(
                f"No MotionExpertModel weights found in checkpoint {checkpoint_path!r}."
            )
        missing, unexpected = self.motion_expert.load_state_dict(cleaned_state, strict=False)
        if (missing or unexpected) and (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                "[motion_latent] Loaded Motion Expert with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )

    def _build_motion_queries(self, pcd):
        if pcd is None:
            raise ValueError("flower_actor_3d_motion_latent requires 3D point clouds.")
        front_pcd = pcd[:, 0].float()
        batch_size, _, height, width = front_pcd.shape
        query_uv = build_regular_query_uv(
            batch_size=batch_size,
            image_hw=(height, width),
            grid_size=self.motion_config.grid_size,
            normalized=True,
            device=front_pcd.device,
            dtype=front_pcd.dtype,
        )
        query_xyz = F.grid_sample(
            front_pcd,
            query_uv.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        query_xyz = torch.nan_to_num(query_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        return query_uv, query_xyz

    def _build_motion_tokens(self, query_uv, query_xyz, instr_mask=None):
        visual_tokens = getattr(self.encoder, "_last_motion_visual_tokens", None)
        visual_grid = getattr(self.encoder, "_last_motion_visual_grid", None)
        text_tokens = getattr(self.encoder, "_last_motion_text_tokens", None)
        if visual_tokens is None or visual_grid is None or text_tokens is None:
            raise RuntimeError(
                "Motion latent features were not cached by the Florence2 encoder. "
                "Ensure backbone='florence2' and encode_inputs ran before motion extraction."
            )
        grad_ctx = nullcontext() if self.motion_finetune else torch.no_grad()
        with grad_ctx:
            motion = self.motion_expert.extract_one_ode_step_latent(
                query_uv=query_uv,
                query_world_xyz=query_xyz,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_attention_mask=instr_mask,
                visual_grid=visual_grid,
                solver_steps=self.motion_solver_steps,
                steps_to_run=self.motion_steps_to_run,
            )
        pooled = self.motion_temporal_pool(motion["motion_hidden"])
        motion_tokens = self.motion_proj(pooled.to(self.motion_proj.weight.dtype))
        return motion_tokens, motion["motion_pos"]

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
        base_result = super().encode_inputs(
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )
        (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = base_result
        query_uv, query_xyz = self._build_motion_queries(pcd)
        motion_tokens, motion_pos = self._build_motion_tokens(
            query_uv,
            query_xyz,
            instr_mask=instr_mask,
        )
        if self.scene_motion_attn is not None:
            rgb3d_feats = rgb3d_feats + self.alpha_scene * self.scene_motion_attn(
                rgb3d_feats,
                motion_tokens,
                pcd_tokens,
                motion_pos,
            )
        return (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            motion_tokens, motion_pos,
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            motion_tokens, motion_pos,
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
            motion_tokens=motion_tokens,
            motion_pos=motion_pos,
        )


class FlowerActor3DGroundingCond(FlowerActor3D):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 num_finetune_vision_layers=0,  # Qwen-only; accepted but ignored for Florence2
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
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
            lv2_batch_size=lv2_batch_size
        )
        self.prediction_head = GroundingCondTransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6
        )
        self.grounding_proj = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
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
        self._pending_grounding_3d = grounding_3d
        self._pending_grounding_keep_mask = grounding_keep_mask
        try:
            return super().forward(
                gt_trajectory,
                trajectory_mask,
                rgb3d,
                rgb2d,
                pcd,
                instruction,
                proprio,
                run_inference=run_inference,
                instr_mask=instr_mask,
                instr_lens=instr_lens,
                **kwargs,
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
        base_result = super().encode_inputs(
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )

        grounding_3d = self._pending_grounding_3d
        grounding_keep_mask = self._pending_grounding_keep_mask
        if grounding_3d is not None:
            if grounding_3d.ndim == 3 and grounding_3d.shape[1] == 1:
                grounding_3d = grounding_3d.squeeze(1)
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
            grounding_feats, grounding_pos, grounding_key_padding_mask
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


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = DoubleCrossSelfAttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=rotary_pe,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=rotary_pe,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = DoubleCrossSelfAttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe_0=rotary_pe,
            rotary_pe_1=rotary_pe,
            use_adaln=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand=2, F)
            trajectory: (B, trajectory_length, nhand=2, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand(2), F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)
            valid_mask: (B, trajectory_length, nhand=2, 1)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        B, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        if valid_mask is not None:
            token_mask = valid_mask.squeeze(-1)  # (B, T, nhand=2, 1) → (B, T, nhand=2)
            token_mask = token_mask.reshape(token_mask.shape[0], -1)  # (B, T*2)
            token_mask_float = token_mask.float()  # float mask for multiplication, 1 for real, 0 for fake
            token_mask_bool = token_mask == 0  # bool mask for attention, False for real, True for fake
        else:
            token_mask_float = None
            token_mask_bool = None

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_token_mask=token_mask_float,  # NOTE: (B, T*2) 1 for real, 0 for fake
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        valid_mask_hist = None
        if valid_mask is not None:
            hand_valid = valid_mask[:, 0, :, 0]  # (B, 2)
            valid_mask_hist = hand_valid[:, None, :]  # (B,1,2)
            nhist = proprio_feats.shape[1]//2
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)  # (B,nhist,nhand=2)
            valid_mask_hist = valid_mask_hist.reshape(B, nhist*nhand).unsqueeze(-1).float()  # (B, nhist*nhand, 1)
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats, valid_mask_hist  # NOTE: mask fake hand feature inside proprio_feats
        )

        # Concatenate scene feature and language features
        rgb3d_feats = torch.cat((rgb3d_feats, instr_feats), 1)
        rgb3d_pos = torch.cat((rgb3d_pos, instr_pos), 1)

        # Positional embeddings
        rel_traj_pos = self.relative_pe_layer(traj_xyz)  # NOTE: fake hand doesn't influence this
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)

        # Self attention among gripper and sampled context
        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            seq_token_mask=token_mask_float,  # NOTE: mask shared cross and self attention for fake hand
            seq_token_mask_bool=token_mask_bool,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1], token_mask_float=token_mask_float, token_mask_bool=token_mask_bool  # NOTE: mask fake hand attention for rot head
        )

        # Position head
        position, position_features = self.predict_pos(
            traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1], token_mask_float=token_mask_float, token_mask_bool=token_mask_bool  # NOTE: mask fake hand attention for pos head
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

    def encode_denoising_timestep(self, timestep, proprio_feats, valid_mask_hist=None):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)
            - proprio_feats: (B, nhist*nhand, F)
            - valid_mask_hist: (B, nhist*nhand, 1)
        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        if valid_mask_hist is not None:
            proprio_feats = proprio_feats * valid_mask_hist
        proprio_feats = proprio_feats.flatten(1)  # (B, nhist*nhand, F) -> (B, nhist*nhand*F)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)  # (B, F)
        return time_feats + curr_gripper_feats

    def predict_pos(self, traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
                    time_embs, traj_len, token_mask_float=None, token_mask_bool=None,
                    seq0_key_padding_mask=None):
        position_features = self.position_self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=seq0_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
                    time_embs, traj_len, token_mask_float=None, token_mask_bool=None,
                    seq0_key_padding_mask=None):
        rotation_features = self.rotation_self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=seq0_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation


class MotionLatentTransformerHead(TransformerHead):
    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6,
                 motion_action_fusion=True):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            nhist=nhist,
            rotary_pe=rotary_pe,
            rot_dim=rot_dim,
        )
        self.motion_action_fusion = bool(motion_action_fusion)
        if self.motion_action_fusion:
            self.action_motion_attn = CrossAttention3D(
                embedding_dim,
                num_attn_heads,
                flash_attn_name="action-motion CrossAttention3D",
            )
            self.alpha_action = nn.Parameter(torch.zeros(num_shared_attn_layers))
        else:
            self.action_motion_attn = None
            self.alpha_action = None

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None,
                motion_tokens=None, motion_pos=None):
        B, traj_len, nhand, _ = trajectory.shape

        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        if valid_mask is not None:
            token_mask = valid_mask.squeeze(-1)
            token_mask = token_mask.reshape(token_mask.shape[0], -1)
            token_mask_float = token_mask.float()
            token_mask_bool = token_mask == 0
        else:
            token_mask_float = None
            token_mask_bool = None

        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_token_mask=token_mask_float,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        valid_mask_hist = None
        if valid_mask is not None:
            hand_valid = valid_mask[:, 0, :, 0]
            valid_mask_hist = hand_valid[:, None, :]
            nhist = proprio_feats.shape[1] // 2
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)
            valid_mask_hist = valid_mask_hist.reshape(B, nhist * nhand).unsqueeze(-1).float()
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats, valid_mask_hist
        )

        rgb3d_feats = torch.cat((rgb3d_feats, instr_feats), 1)
        rgb3d_pos = torch.cat((rgb3d_pos, instr_pos), 1)

        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(rgb3d_pos)

        self_kp_mask = token_mask_bool if token_mask_bool is not None else None
        for layer_idx in range(self.self_attn.num_layers):
            if token_mask_float is not None:
                traj_feats = traj_feats * token_mask_float.unsqueeze(-1)
            traj_feats = self.self_attn.attn_layers_0[layer_idx](
                traj_feats, rgb3d_feats,
                None,
                rel_traj_pos, rel_scene_pos,
                None, None,
                time_embs
            )
            if token_mask_float is not None:
                traj_feats = traj_feats * token_mask_float.unsqueeze(-1)
            if (
                self.motion_action_fusion
                and self.action_motion_attn is not None
                and motion_tokens is not None
                and motion_pos is not None
            ):
                traj_feats = traj_feats + self.alpha_action[layer_idx] * self.action_motion_attn(
                    traj_feats,
                    motion_tokens,
                    traj_xyz,
                    motion_pos,
                )
                if token_mask_float is not None:
                    traj_feats = traj_feats * token_mask_float.unsqueeze(-1)
            traj_feats = self.self_attn.attn_layers_1[layer_idx](
                traj_feats, traj_feats,
                self_kp_mask,
                rel_traj_pos, rel_traj_pos,
                None, None,
                time_embs
            )
            if token_mask_float is not None:
                traj_feats = traj_feats * token_mask_float.unsqueeze(-1)
            traj_feats = self.self_attn.ffw_layers[layer_idx](traj_feats, time_embs)
            if token_mask_float is not None:
                traj_feats = traj_feats * token_mask_float.unsqueeze(-1)

        rotation = self.predict_rot(
            traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float,
            token_mask_bool=token_mask_bool
        )

        position, position_features = self.predict_pos(
            traj_feats, rgb3d_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float,
            token_mask_bool=token_mask_bool
        )

        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]


class GroundingCondTransformerHead(TransformerHead):
    """TransformerHead variant that injects an optional grounding token."""

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None,
                grounding_feats=None, grounding_pos=None,
                grounding_key_padding_mask=None):
        B, traj_len, nhand, _ = trajectory.shape

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
            seq2_sem_pos=None
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

        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(context_pos)

        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]

        rotation = self.predict_rot(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float,
            token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
        )
        position, position_features = self.predict_pos(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float,
            token_mask_bool=token_mask_bool,
            seq0_key_padding_mask=context_key_padding_mask,
        )
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), dim=-1)
                 .unflatten(1, (traj_len, nhand))
        ]


class FlowerActor2D(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
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
            lv2_batch_size=lv2_batch_size
        )
        # Vision-language encoder, runs only once
        self.encoder = Encoder2D(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead2D(
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6
        )


class TransformerHead2D(TransformerHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=False,
                 rot_dim=6):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            nhist=nhist,
            rotary_pe=rotary_pe,
            rot_dim=rot_dim
        )

        del self.relative_pe_layer

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats
        )

        # Concatenate scene feature and language features
        rgb3d_feats = torch.cat((rgb3d_feats, instr_feats), 1)

        # Self attention among gripper and sampled context
        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            traj_feats, rgb3d_feats, None, None,
            time_embs, traj_feats.shape[1]
        )

        # Position head
        position, position_features = self.predict_pos(
            traj_feats, rgb3d_feats, None, None,
            time_embs, traj_feats.shape[1]
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]
