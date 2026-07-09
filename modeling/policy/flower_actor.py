from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch import nn
import einops

from motion_expert.config import load_motion_expert_config
from motion_expert.data import build_regular_query_uv
from motion_expert.model import MotionExpertModel
from motion_expert.model_2d import MotionExpert2DModel
from motion_expert.model_uvd import MotionExpertUVDModel
from motion_expert.zarr_dataset_uvd import (
    DEFAULT_QUERY_DEPTH_MEAN,
    DEFAULT_QUERY_DEPTH_STD,
)

from ..encoder.multimodal.encoder3d import Encoder as Encoder3D
from ..encoder.multimodal.encoder2d import Encoder as Encoder2D
from ..encoder.multimodal.encoder2dfa import Encoder as Encoder2DFA
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


class CrossAttentionNoRoPE(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, dropout=0.1, flash_attn_name="CrossAttentionNoRoPE"):
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

    def forward(self, query_tokens, memory_tokens, query_pos=None, memory_pos=None):
        del query_pos, memory_pos
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(memory_tokens).to(q.dtype)
        out = self.attn(
            query=q.transpose(0, 1),
            key=kv.transpose(0, 1),
            value=kv.transpose(0, 1),
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


class Motion2DLatentMixin:
    """Mixin for policies that fuse a 2D MotionExpert latent into scene tokens."""

    def _init_motion_2d_latent(
        self,
        motion_expert_config,
        motion_expert_checkpoint,
        motion_solver_steps,
        motion_steps_to_run,
        motion_finetune,
        motion_scene_fusion,
        embedding_dim,
        num_attn_heads,
    ):
        if motion_expert_config is None:
            raise ValueError(f"{type(self).__name__} requires --motion_expert_config.")

        motion_config = load_motion_expert_config(str(motion_expert_config))
        if str(motion_config.backbone_type).lower() != "florence2":
            raise ValueError(
                f"{type(self).__name__} expects a Florence2 2D Motion Expert config, "
                f"got backbone_type={motion_config.backbone_type!r}."
            )
        motion_feature_dim = int(self.encoder.cond_linear.in_features)
        max_conditioning_tokens = (
            int(motion_config.sparse_decoder.visual_global_tokens)
            + motion_config.text_max_length
        )
        self.motion_expert = MotionExpert2DModel(
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
            self.scene_motion_attn = CrossAttentionNoRoPE(
                embedding_dim,
                num_attn_heads,
                flash_attn_name="scene-motion CrossAttentionNoRoPE",
            )
            self.alpha_scene = nn.Parameter(torch.zeros(()))
        else:
            self.scene_motion_attn = None
            self.alpha_scene = None

        if motion_expert_checkpoint is not None:
            self._load_motion_2d_expert_checkpoint(str(motion_expert_checkpoint))
        if not self.motion_finetune:
            for param in self.motion_expert.parameters():
                param.requires_grad = False
            if hasattr(torch, "compile"):
                self.motion_expert.point_encoder = torch.compile(
                    self.motion_expert.point_encoder, fullgraph=False
                )
                self.motion_expert.flow_decoder = torch.compile(
                    self.motion_expert.flow_decoder, fullgraph=False
                )

    def _load_motion_2d_expert_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        motion_keys = set(self.motion_expert.state_dict().keys())
        accepted_prefixes = (
            "point_scalar_proj.",
            "visual_proj.",
            "text_proj.",
            "point_norm.",
            "point_type_embed",
            "global_visual_type_embed",
            "text_type_embed",
            "point_encoder.",
            "flow_decoder.",
            "traj_delta_uv_scale",
        )
        cleaned_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("motion_expert."):
                key = key[len("motion_expert."):]
            if key.startswith("model."):
                key = key[len("model."):]
            if key in motion_keys or key.startswith(accepted_prefixes):
                cleaned_state[key] = value
        if not cleaned_state:
            raise ValueError(
                f"No MotionExpert2DModel weights found in checkpoint {checkpoint_path!r}."
            )
        missing, unexpected = self.motion_expert.load_state_dict(cleaned_state, strict=False)
        if (missing or unexpected) and (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                "[motion_2d_latent] Loaded 2D Motion Expert with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )

    def _build_motion_2d_queries(self, rgb3d):
        if rgb3d is None:
            raise ValueError(f"{type(self).__name__} requires RGB observations.")
        front_rgb = rgb3d[:, 0].float()
        batch_size, _, height, width = front_rgb.shape
        cache_key = (batch_size, height, width, front_rgb.device, front_rgb.dtype)
        if getattr(self, "_query_uv_2d_cache_key", None) != cache_key:
            self._query_uv_2d_cache_key = cache_key
            self._query_uv_2d_cached = build_regular_query_uv(
                batch_size=batch_size,
                image_hw=(height, width),
                grid_size=self.motion_config.grid_size,
                normalized=True,
                device=front_rgb.device,
                dtype=front_rgb.dtype,
            )
        return self._query_uv_2d_cached

    def _build_motion_2d_tokens(self, query_uv, instr_mask=None):
        visual_tokens = getattr(self.encoder, "_last_motion_visual_tokens", None)
        visual_grid = getattr(self.encoder, "_last_motion_visual_grid", None)
        text_tokens = getattr(self.encoder, "_last_motion_text_tokens", None)
        if visual_tokens is None or visual_grid is None or text_tokens is None:
            raise RuntimeError(
                "2D motion latent features were not cached by the Florence2 encoder. "
                "Ensure backbone='florence2' and encode_inputs ran before motion extraction."
            )
        grad_ctx = nullcontext() if self.motion_finetune else torch.no_grad()
        with grad_ctx:
            motion = self.motion_expert.extract_one_ode_step_latent(
                query_uv=query_uv,
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

    def _scene_motion_2d_delta(self, rgb3d_feats, motion_tokens):
        return self.alpha_scene * self.scene_motion_attn(rgb3d_feats, motion_tokens)

    def _fuse_motion_2d_into_scene(self, base_result, rgb3d, instr_mask=None):
        (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = base_result
        query_uv = self._build_motion_2d_queries(rgb3d)
        motion_tokens, _motion_pos = self._build_motion_2d_tokens(
            query_uv,
            instr_mask=instr_mask,
        )
        if self.scene_motion_attn is not None:
            rgb3d_feats = rgb3d_feats + self._scene_motion_2d_delta(
                rgb3d_feats,
                motion_tokens,
            )
        return (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        )


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
                 proprio_nhand=None,
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
        proprio_nhist = nhist * (2 if proprio_nhand is None else int(proprio_nhand))
        # Vision-language encoder, runs only once
        self.encoder = Encoder3D(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Action decoder, runs at every denoising timestep
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
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
                 proprio_nhand=None,
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
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size
        )
        proprio_nhist = nhist * (2 if proprio_nhand is None else int(proprio_nhand))

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

        # Interpretability hooks (off by default; enabled by the motion-attention
        # probe). When `_capture_motion_attn` is True, encode_inputs stores per-patch
        # motion-update-norm maps for the real motion latent and cheap counterfactuals.
        self._capture_motion_attn = False
        self._motion_attn_capture = None
        self._motion_attn_last_query = None
        self._motion_attn_seed = 0

        # 3D cameras to sample motion queries from. Default (0,) = front-only,
        # i.e. one motion latent (unchanged behavior for CALVIN/LIBERO/RoboTwin2).
        # When set to several cameras (e.g. ALOHA front+back via the trainer), each
        # camera contributes its OWN motion latent -- shared expert weights, but
        # queries and visual conditioning taken from that camera -- and the
        # per-camera latents are concatenated into ONE world-frame motion bank that
        # all scene patches attend to through the single scene_motion_attn /
        # alpha_scene gate ("unified" multi-camera fusion, Design B). This adds no
        # parameters and leaves the single-camera path byte-for-byte unchanged.
        self._motion_query_cameras = (0,)

        self.prediction_head = MotionLatentTransformerHead(
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
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
            if hasattr(torch, "compile"):
                self.motion_expert.point_encoder = torch.compile(
                    self.motion_expert.point_encoder, fullgraph=False
                )
                self.motion_expert.flow_decoder = torch.compile(
                    self.motion_expert.flow_decoder, fullgraph=False
                )

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

    def _build_motion_queries(self, pcd, cam=0):
        if pcd is None:
            raise ValueError("flower_actor_3d_motion_latent requires 3D point clouds.")
        cam_pcd = pcd[:, cam].float()
        batch_size, _, height, width = cam_pcd.shape
        # The regular query-UV grid is camera-independent (normalized image grid),
        # so cache it on geometry only; query_xyz is grid-sampled per camera below.
        cache_key = (batch_size, height, width, cam_pcd.device, cam_pcd.dtype)
        if getattr(self, "_query_uv_cache_key", None) != cache_key:
            self._query_uv_cache_key = cache_key
            self._query_uv_cached = build_regular_query_uv(
                batch_size=batch_size,
                image_hw=(height, width),
                grid_size=self.motion_config.grid_size,
                normalized=True,
                device=cam_pcd.device,
                dtype=cam_pcd.dtype,
            )
        query_uv = self._query_uv_cached
        query_xyz = F.grid_sample(
            cam_pcd,
            query_uv.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        query_xyz = torch.nan_to_num(query_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        return query_uv, query_xyz

    def _build_motion_tokens(self, query_uv, query_xyz, instr_mask=None, cam=None):
        visual_tokens = getattr(self.encoder, "_last_motion_visual_tokens", None)
        visual_grid = getattr(self.encoder, "_last_motion_visual_grid", None)
        text_tokens = getattr(self.encoder, "_last_motion_text_tokens", None)
        if visual_tokens is None or visual_grid is None or text_tokens is None:
            raise RuntimeError(
                "Motion latent features were not cached by the Florence2 encoder. "
                "Ensure backbone='florence2' and encode_inputs ran before motion extraction."
            )
        # Per-camera conditioning for the unified multi-camera fusion: restrict the
        # motion expert's visual sampling AND global pooling to camera `cam`'s grid
        # so each camera's latent is conditioned on its own image. `cam=None` keeps
        # the original behavior (full multi-cam grid; expert samples from cam 0).
        if cam is not None:
            if visual_grid.dim() != 5:
                raise ValueError(
                    "Per-camera motion conditioning expects a cached "
                    "(B, ncam, H, W, C) visual grid; got "
                    f"{tuple(visual_grid.shape)}."
                )
            _, ncam, h_feat, w_feat, _ = visual_grid.shape
            if not 0 <= cam < ncam:
                raise ValueError(
                    f"motion query camera {cam} out of range (ncam={ncam})."
                )
            tokens_per_cam = h_feat * w_feat
            visual_grid = visual_grid[:, cam]  # (B, H, W, C) -> expert samples this cam
            visual_tokens = visual_tokens[
                :, cam * tokens_per_cam:(cam + 1) * tokens_per_cam
            ]
        # Use no_grad (not inference_mode) for the frozen motion expert: its outputs
        # feed the trainable motion-fusion layers (motion_temporal_pool / scene/action
        # attn), and inference_mode tensors cannot be saved for backward.
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
        motion_cams = tuple(getattr(self, "_motion_query_cameras", (0,)))
        if motion_cams == (0,):
            # Original single-camera (front) path: full multi-cam visual
            # conditioning, one motion latent. Unchanged for CALVIN/LIBERO/RoboTwin2.
            query_uv, query_xyz = self._build_motion_queries(pcd)
            motion_tokens, motion_pos = self._build_motion_tokens(
                query_uv,
                query_xyz,
                instr_mask=instr_mask,
            )
        else:
            # Unified multi-camera fusion (Design B): one latent per camera (shared
            # expert, per-camera queries + visual conditioning), concatenated into a
            # single world-frame motion bank. All scene patches then attend to the
            # union through the single scene_motion_attn / alpha_scene gate below.
            tok_list, pos_list = [], []
            for cam in motion_cams:
                q_uv, q_xyz = self._build_motion_queries(pcd, cam=cam)
                tok, pos = self._build_motion_tokens(
                    q_uv, q_xyz, instr_mask=instr_mask, cam=cam,
                )
                tok_list.append(tok)
                pos_list.append(pos)
            motion_tokens = torch.cat(tok_list, dim=1)
            motion_pos = torch.cat(pos_list, dim=1)
        if self.scene_motion_attn is not None:
            scene_motion_delta = self._scene_motion_delta(
                rgb3d_feats, motion_tokens, pcd_tokens, motion_pos
            )
            if self._capture_motion_attn:
                self._capture_scene_motion(
                    rgb3d_feats, motion_tokens, pcd_tokens, motion_pos,
                    scene_motion_delta,
                )
            rgb3d_feats = rgb3d_feats + scene_motion_delta
        return (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            motion_tokens, motion_pos,
        )

    # ------------------------------------------------------------------
    # Motion-attention interpretability helpers
    #
    # The scene fusion is `rgb3d_feats += alpha_scene * scene_motion_attn(
    #   x=rgb3d_feats (query), context=motion_tokens (key/value))`. Because patches
    # are the *query*, the softmax is normalized per-patch over motion tokens and is
    # NOT a per-patch importance map. The well-defined per-patch signal is instead
    # the magnitude of the residual update injected by the motion latent, which
    # already folds in value vectors and the learned alpha_scene gate.
    # ------------------------------------------------------------------
    def _scene_motion_delta(self, rgb3d_feats, motion_tokens, pcd_tokens, motion_pos):
        """alpha_scene * scene_motion_attn(...), i.e. the residual update the motion
        latent adds to each scene patch. Shape (B, N, C)."""
        return self.alpha_scene * self.scene_motion_attn(
            rgb3d_feats, motion_tokens, pcd_tokens, motion_pos
        )

    @torch.no_grad()
    def _capture_scene_motion(self, rgb3d_feats, motion_tokens, pcd_tokens,
                              motion_pos, scene_motion_delta):
        """Stash per-patch update-norm maps for the real motion latent and two cheap
        single-sample counterfactuals. The query (`rgb3d_feats`) is held fixed across
        all variants, so any difference in the maps is attributable to the *content*
        of the motion latent rather than to the visual backbone."""
        def patch_norm(delta):
            return delta.detach().float().norm(dim=-1)  # (B, N)

        capture = {
            "real": patch_norm(scene_motion_delta),
            "grid": getattr(self.encoder, "_last_rgb3d_grid", None),
            "alpha_scene": float(self.alpha_scene.detach().float().reshape(-1)[0].cpu()),
            "rgb3d_pos": pcd_tokens.detach().float().cpu(),
        }

        # (a) zeroed motion: floor — the gated residual should contribute ~nothing.
        capture["zero"] = patch_norm(
            self._scene_motion_delta(
                rgb3d_feats, torch.zeros_like(motion_tokens), pcd_tokens, motion_pos
            )
        )

        # (b) spatial shuffle: identical latent magnitudes but the plan->location
        # binding is scrambled (motion_pos kept fixed). Tests whether the *spatial
        # arrangement* of the predicted motion is what drives the modulation.
        num_motion = motion_tokens.shape[1]
        gen = torch.Generator(device=motion_tokens.device)
        gen.manual_seed(int(self._motion_attn_seed))
        perm = torch.randperm(num_motion, generator=gen, device=motion_tokens.device)
        capture["shuffle"] = patch_norm(
            self._scene_motion_delta(
                rgb3d_feats, motion_tokens[:, perm, :], pcd_tokens, motion_pos
            )
        )

        self._motion_attn_capture = capture
        # Keep the (fixed) real scene query so a caller can score decoy-instruction
        # motion tokens against the exact same scene features.
        self._motion_attn_last_query = (
            rgb3d_feats.detach(), pcd_tokens.detach(), motion_pos.detach(),
        )

    @torch.no_grad()
    def scene_motion_norm_for_tokens(self, motion_tokens):
        """Per-patch update norm for externally supplied motion tokens, scored
        against the most recently captured (real) scene query. Used to build the
        'different instruction' counterfactual: same scene, different plan."""
        if self._motion_attn_last_query is None:
            raise RuntimeError(
                "No captured scene query. Run a forward pass with "
                "`_capture_motion_attn = True` before calling this."
            )
        rgb3d_feats, pcd_tokens, motion_pos = self._motion_attn_last_query
        delta = self._scene_motion_delta(
            rgb3d_feats, motion_tokens.to(rgb3d_feats.dtype), pcd_tokens, motion_pos
        )
        return delta.detach().float().norm(dim=-1)

    @torch.no_grad()
    def encode_motion_tokens(self, rgb3d, rgb2d, pcd, instruction, proprio,
                             instr_mask=None, instr_lens=None):
        """Encode (rgb, pcd, instruction) and return only the motion latent tokens
        and their positions. Re-runs the Florence2 encoder so the motion caches
        reflect `instruction`; positions come from the fixed query grid on the front
        point cloud, so they match a real pass on the same observation. Used to feed
        a 'different instruction' (decoy) plan into the same scene. Note this
        overwrites the encoder's motion caches, so call it *after* reading a real
        capture for the step."""
        super().encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )
        query_uv, query_xyz = self._build_motion_queries(pcd)
        motion_tokens, motion_pos = self._build_motion_tokens(
            query_uv, query_xyz, instr_mask=instr_mask
        )
        return motion_tokens.detach().clone(), motion_pos.detach().clone()

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None,
                            return_hidden=False):
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
        # relative to absolute for decoder position queries; honors
        # self._relative_position_mode (RoboTwin2 uses "cumulative")
        traj_xyz = self._absolute_query_xyz(query_trajectory, traj_xyz)

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
            return_hidden=return_hidden,
        )


class FlowerActor3DMotionTraj(FlowerActor3DMotionLatent):
    """Ablation A: one-step denoising (noise -> clean) of the 3D MotionExpert, fused
    via the *explicit predicted trajectory*.

    Reuses the entire FlowerActor3DMotionLatent fusion stack (scene-motion +
    action-motion CrossAttention3D, MotionLatentTransformerHead). The only change is
    what fills each motion token: instead of the decoder hidden at the noise point,
    we run a single Euler flow step to the clean trajectory and embed the predicted
    per-step displacements.

    How the explicit trajectory enters the scene-motion cross-attention pattern:
    the fusion is RoPE-3D cross-attention where each motion token is anchored at a
    query point's current world xyz (``motion_pos``) and its *content* conditions the
    scene patches. We keep exactly one token per query point -- anchored at the same
    position -- but its content is now an encoding of that point's predicted future
    motion. The whole (future_steps x 3) trajectory is flattened and mapped by a small
    MLP to one ``embedding_dim`` token. Flattening preserves temporal order by
    construction (step i occupies fixed slots), so -- unlike a per-step embed + pool --
    the token is *not* a permutation-invariant function of the set of displacements;
    trajectory shape (ordering, direction changes) survives. Token count, positions,
    and every downstream attention thus match the latent variant one-for-one; only the
    per-token semantics differ (explicit motion vs. latent).

    NOTE on ablation fairness: the latent variant's token comes from a 512-d decoder
    hidden contextualized across points + conditioning, whereas this variant re-encodes
    from raw displacements through a shallow MLP. The capacity is deliberately modest
    to keep the signal "explicit"; a gap vs. the latent variant should be read with
    that asymmetry in mind.
    """

    # NOTE: the explicit signature is required -- the trainer uses
    # inspect.signature(model_cls.__init__) to decide which optional kwargs
    # (motion_expert_config, motion_finetune, ...) to inject, so *args/**kwargs would
    # silently drop them.
    def __init__(self,
                 backbone="florence2",
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
                 rotation_format='quat_xyzw',
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 motion_expert_config=None,
                 motion_expert_checkpoint=None,
                 motion_solver_steps=10,
                 motion_steps_to_run=1,
                 motion_finetune=True,
                 motion_scene_fusion=True,
                 motion_action_fusion=True,
                 proprio_nhand=None,
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
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            motion_action_fusion=motion_action_fusion,
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size,
        )
        hidden_dim = self.motion_config.sparse_decoder.hidden_dim
        future_steps = int(self.motion_config.future_steps)
        # Order-aware trajectory encoder: flatten the per-point future trajectory
        # (future_steps x 3, normalized expert frame) and map it to one motion token.
        # Flattening keeps temporal order, so no per-step temporal pooling is needed.
        self.motion_traj_encoder = nn.Sequential(
            nn.Linear(future_steps * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # The inherited per-step pool/projection are unused in this variant; drop them
        # so they don't sit as dead parameters (also avoids DDP unused-param overhead).
        del self.motion_temporal_pool
        del self.motion_proj

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
            motion = self.motion_expert.extract_one_step_denoise(
                query_uv=query_uv,
                query_world_xyz=query_xyz,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_attention_mask=instr_mask,
                visual_grid=visual_grid,
                return_clean_latent=False,
                return_clean_trajectory=False,
            )
        # (B, P, future_steps, 3) normalized clean displacements -> one token/point.
        future_delta = motion["future_delta_norm"]
        batch_size, num_points, num_steps, _ = future_delta.shape
        flat = future_delta.reshape(batch_size, num_points, num_steps * 3)
        weight_dtype = self.motion_traj_encoder[0].weight.dtype
        motion_tokens = self.motion_traj_encoder(flat.to(weight_dtype))
        return motion_tokens, motion["motion_pos"]


class FlowerActor3DMotionCleanLatent(FlowerActor3DMotionLatent):
    """Ablation B: one-step denoising (noise -> clean) of the 3D MotionExpert, fused
    via the *latent of the clean trajectory*.

    Identical to FlowerActor3DMotionLatent except for which latent is read. The parent
    reads the decoder hidden at the noise point (a single network eval at tau=0). Here
    we take one Euler flow step to the clean trajectory, then re-evaluate the decoder
    *at that clean trajectory* (tau=1) and pool its hidden. Everything else -- temporal
    pool, projection, scene/action fusion, head -- is inherited unchanged.
    """

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
            motion = self.motion_expert.extract_one_step_denoise(
                query_uv=query_uv,
                query_world_xyz=query_xyz,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_attention_mask=instr_mask,
                visual_grid=visual_grid,
                return_clean_latent=True,
                return_clean_trajectory=False,
            )
        pooled = self.motion_temporal_pool(motion["motion_hidden"])
        motion_tokens = self.motion_proj(pooled.to(self.motion_proj.weight.dtype))
        return motion_tokens, motion["motion_pos"]


class FlowerActor3DMotionPartialLatent(FlowerActor3DMotionLatent):
    """Ablation: partially-denoised motion latent.

    Identical to FlowerActor3DMotionLatent except for which latent is read. The
    parent reads the decoder hidden at the pure noise point (tau=0, no
    integration). Here we run ``motion_steps_to_run`` Euler steps of a
    ``motion_solver_steps``-step noise->clean schedule (default 1 of 10, dt=0.1)
    and read the hidden at the partially-denoised trajectory. Noise is random, so
    this isolates a single variable vs the baseline: *where on the denoising path
    the latent is read*. Everything downstream (temporal pool, projection,
    scene/action fusion, head) is inherited unchanged.
    """

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
            motion = self.motion_expert.extract_partial_denoise_latent(
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


class FlowerActor3DMotionEncoderLatent(FlowerActor3DMotionLatent):
    """Ablation: scene-motion fusion from the MotionExpert *encoder* latent.

    Every other motion-latent variant fuses a latent read from the flow *decoder*
    (the per-future-step hidden, after 0/partial/full denoising). This ablation
    instead uses the SparsePointEncoder output (``point_context``) directly -- the
    purely-spatial per-point latent produced before the decoder. Consequences:

      * A single forward through only the tokenizer + ``point_encoder``; the flow
        decoder is never run (no diffusion noise, no ODE integration).
      * The latent is ``(B, P, hidden_dim)`` with no future-step axis, so the
        inherited per-step ``motion_temporal_pool`` is not applicable and is dropped.
        Tokens are projected straight to ``embedding_dim`` via the inherited
        ``motion_proj``.

    Token count, positions (``motion_pos`` = query world xyz), and every downstream
    fusion (scene-motion CrossAttention3D, MotionLatentTransformerHead) match the
    baseline one-for-one; only the per-token *content* differs (encoder latent vs.
    decoder latent).
    """

    # NOTE: the explicit signature is required -- the trainer uses
    # inspect.signature(model_cls.__init__) to decide which optional kwargs
    # (motion_expert_config, motion_finetune, ...) to inject, so *args/**kwargs would
    # silently drop them.
    def __init__(self,
                 backbone="florence2",
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
                 rotation_format='quat_xyzw',
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 motion_expert_config=None,
                 motion_expert_checkpoint=None,
                 motion_solver_steps=10,
                 motion_steps_to_run=1,
                 motion_finetune=True,
                 motion_scene_fusion=True,
                 motion_action_fusion=True,
                 proprio_nhand=None,
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
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            motion_action_fusion=motion_action_fusion,
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size,
        )
        # The encoder latent has no future-step axis, so the per-step temporal pool
        # is unused. Drop it so it doesn't sit as a dead trainable parameter.
        del self.motion_temporal_pool
        # The flow decoder is never run in this ablation. Freeze it so its params are
        # excluded from the optimizer / DDP gradient sync (the encoder weights it was
        # loaded with are still kept for parity, just not trained).
        for param in self.motion_expert.flow_decoder.parameters():
            param.requires_grad = False

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
            motion = self.motion_expert.extract_encoder_latent(
                query_uv=query_uv,
                query_world_xyz=query_xyz,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_attention_mask=instr_mask,
                visual_grid=visual_grid,
            )
        # (B, P, hidden_dim) encoder latent -> one motion token per query point.
        latent = motion["motion_encoder_latent"]
        motion_tokens = self.motion_proj(latent.to(self.motion_proj.weight.dtype))
        return motion_tokens, motion["motion_pos"]


class FlowerActor3DMotionLatentUVD(FlowerActor3DMotionLatent):
    """CALVIN 3DFA-VLA policy that fuses a **UVD** MotionExpert latent.

    Identical in every respect to ``FlowerActor3DMotionLatent`` -- same Florence2
    scene encoder, same temporal pool / motion projection, same scene-motion and
    action-motion ``CrossAttention3D`` fusion, same ``MotionLatentTransformerHead``
    -- except the motion expert is a ``MotionExpertUVDModel`` that reasons in
    ``(u_grid, v_grid, depth_metres)`` space rather than world XYZ.

    The expert's geometry input ``query_uvd`` is built from the same regular query
    grid as the parent (so ``(u, v)`` are the normalized image grid used for visual
    sampling) plus the **camera-frame metric depth** of each query point, globally
    standardized exactly as in the UVD MotionExpert's training data. That depth is
    recovered from the world point cloud (which the policy already grid-samples into
    ``query_xyz``) via CALVIN's fixed static front-camera transform, so no extra
    inputs / plumbing are needed.

    Crucially, the motion tokens are still positioned at their **world-xyz**
    locations for the RoPE-3D scene/action fusion, so the fusion is byte-for-byte
    the parent's; only the expert's internal geometry is UVD. CALVIN static-camera
    only (the world<->camera transform is CALVIN-specific).
    """

    # CALVIN static front-camera transform, copied verbatim from
    # utils/depth2cloud/calvin.py. There, ``T_world_cam = inv(M)`` with
    # ``M = reshape(4,4).T`` of this flat list, so ``world->cam == M`` and we can
    # recover the camera-frame depth of a world point as ``-(M @ [xyz,1]).z``.
    _CALVIN_STATIC_WORLD2CAM_FLAT = (
        0.7232445478439331, -0.030260592699050903, 0.6899287700653076, 0.0,
        0.5141233801841736, 0.6906105875968933, -0.5086592435836792, 0.0,
        -0.46107974648475647, 0.7225935459136963, 0.5150379538536072, 0.0,
        0.21526622772216797, -0.26317155361175537, -4.399168968200684, 1.0,
    )

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
                 # UVD query-depth standardization (camera-frame metres). Defaults
                 # match compute_calvin_uvd_norm_stats.py / the expert's training.
                 query_depth_mean=None,
                 query_depth_std=None,
                 # Training arguments
                 proprio_nhand=None,
                 lv2_batch_size=1):
        if backbone != "florence2":
            raise ValueError(
                f"{type(self).__name__} currently supports backbone='florence2', got {backbone!r}."
            )
        if motion_expert_config is None:
            raise ValueError(
                "flower_actor_3d_motion_latent_uvd requires --motion_expert_config."
            )
        # Build the FlowerActor3D base (encoder + base head) directly, skipping the
        # parent's world-XYZ MotionExpertModel construction so we can swap in the
        # UVD expert without instantiating/loading the wrong model.
        FlowerActor3D.__init__(
            self,
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
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size,
        )
        proprio_nhist = nhist * (2 if proprio_nhand is None else int(proprio_nhand))

        motion_config = load_motion_expert_config(str(motion_expert_config))
        if str(motion_config.backbone_type).lower() != "florence2":
            raise ValueError(
                "flower_actor_3d_motion_latent_uvd expects a Florence2 UVD Motion "
                f"Expert config, got backbone_type={motion_config.backbone_type!r}."
            )
        motion_feature_dim = int(self.encoder.cond_linear.in_features)
        # UVD conditioning bank = pooled global-visual tokens + text tokens (no
        # per-query tokens, mirroring the 2D expert).
        max_conditioning_tokens = (
            int(motion_config.sparse_decoder.visual_global_tokens)
            + motion_config.text_max_length
        )
        self.motion_expert = MotionExpertUVDModel(
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

        # Query-depth global standardization (camera-frame metres).
        self.query_depth_mean = float(
            DEFAULT_QUERY_DEPTH_MEAN if query_depth_mean is None else query_depth_mean
        )
        self.query_depth_std = max(
            float(DEFAULT_QUERY_DEPTH_STD if query_depth_std is None else query_depth_std),
            1e-6,
        )
        # world->cam (4x4) for recovering the expert's camera-frame depth from the
        # world point cloud. Non-persistent buffer: follows .to()/.cuda() moves but
        # is never written to / read from checkpoints.
        world2cam = (
            torch.tensor(self._CALVIN_STATIC_WORLD2CAM_FLAT, dtype=torch.float32)
            .reshape(4, 4)
            .t()
            .contiguous()
        )
        self.register_buffer("world2cam", world2cam, persistent=False)

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

        # Interpretability hooks (off by default; see parent for semantics).
        self._capture_motion_attn = False
        self._motion_attn_capture = None
        self._motion_attn_last_query = None
        self._motion_attn_seed = 0
        # Single front camera for CALVIN (unchanged single-camera fusion).
        self._motion_query_cameras = (0,)

        self.prediction_head = MotionLatentTransformerHead(
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
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
            if hasattr(torch, "compile"):
                self.motion_expert.point_encoder = torch.compile(
                    self.motion_expert.point_encoder, fullgraph=False
                )
                self.motion_expert.flow_decoder = torch.compile(
                    self.motion_expert.flow_decoder, fullgraph=False
                )

    def _load_motion_expert_checkpoint(self, checkpoint_path):
        """Load MotionExpertUVDModel weights from a UVD training checkpoint.

        The checkpoint stores the expert under a ``model.`` prefix (plus a separate
        ``feature_extractor.`` Florence2 backbone the policy supplies itself). We
        keep only keys that belong to ``MotionExpertUVDModel`` (matching the live
        module's keys or the known weight/buffer prefixes, including the UVD-delta
        scale buffer ``traj_delta_uvd_scale``).
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        motion_keys = set(self.motion_expert.state_dict().keys())
        accepted_prefixes = (
            "point_scalar_proj.",
            "visual_proj.",
            "text_proj.",
            "point_norm.",
            "point_type_embed",
            "global_visual_type_embed",
            "text_type_embed",
            "point_encoder.",
            "flow_decoder.",
            "traj_delta_uvd_scale",
        )
        cleaned_state = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("motion_expert."):
                key = key[len("motion_expert."):]
            if key.startswith("model."):
                key = key[len("model."):]
            if key in motion_keys or key.startswith(accepted_prefixes):
                cleaned_state[key] = value
        if not cleaned_state:
            raise ValueError(
                f"No MotionExpertUVDModel weights found in checkpoint {checkpoint_path!r}."
            )
        missing, unexpected = self.motion_expert.load_state_dict(cleaned_state, strict=False)
        if (missing or unexpected) and (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                "[motion_latent_uvd] Loaded UVD Motion Expert with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )

    def _query_xyz_to_cam_depth(self, query_xyz):
        """Camera-frame metric depth (B, P) for world-frame query points (B, P, 3).

        Inverts CALVIN's static-camera unprojection: the front point cloud is
        ``world = T_world_cam[:3] @ [cam_xyz, 1]`` with ``cam_xyz.z = -depth``, so
        ``depth = -(world2cam @ [world, 1]).z``.
        """
        w2c = self.world2cam.to(query_xyz.dtype)
        cam_xyz = query_xyz @ w2c[:3, :3].t() + w2c[:3, 3]
        return -cam_xyz[..., 2]

    def _build_motion_tokens(self, query_uv, query_xyz, instr_mask=None, cam=None):
        visual_tokens = getattr(self.encoder, "_last_motion_visual_tokens", None)
        visual_grid = getattr(self.encoder, "_last_motion_visual_grid", None)
        text_tokens = getattr(self.encoder, "_last_motion_text_tokens", None)
        if visual_tokens is None or visual_grid is None or text_tokens is None:
            raise RuntimeError(
                "Motion latent features were not cached by the Florence2 encoder. "
                "Ensure backbone='florence2' and encode_inputs ran before motion extraction."
            )
        # Per-camera conditioning for the unified multi-camera path -- identical
        # slicing to the parent (no-op for CALVIN's single front camera, cam=None).
        if cam is not None:
            if visual_grid.dim() != 5:
                raise ValueError(
                    "Per-camera motion conditioning expects a cached "
                    "(B, ncam, H, W, C) visual grid; got "
                    f"{tuple(visual_grid.shape)}."
                )
            _, ncam, h_feat, w_feat, _ = visual_grid.shape
            if not 0 <= cam < ncam:
                raise ValueError(
                    f"motion query camera {cam} out of range (ncam={ncam})."
                )
            tokens_per_cam = h_feat * w_feat
            visual_grid = visual_grid[:, cam]
            visual_tokens = visual_tokens[
                :, cam * tokens_per_cam:(cam + 1) * tokens_per_cam
            ]
        # UVD geometry coordinate: [u_grid, v_grid, depth_standardized]. (u, v) are
        # the same normalized grid used for visual sampling; depth is the
        # camera-frame metric depth recovered from the world point cloud, globally
        # standardized exactly as in the UVD MotionExpert's training data.
        cam_depth = self._query_xyz_to_cam_depth(query_xyz.float())
        depth_norm = (cam_depth - self.query_depth_mean) / self.query_depth_std
        query_uvd = torch.cat(
            [query_uv.to(depth_norm.dtype), depth_norm.unsqueeze(-1)], dim=-1
        )
        query_uvd = torch.nan_to_num(query_uvd, nan=0.0, posinf=0.0, neginf=0.0)
        grad_ctx = nullcontext() if self.motion_finetune else torch.no_grad()
        with grad_ctx:
            motion = self.motion_expert.extract_one_ode_step_latent(
                query_uv=query_uv,
                query_uvd=query_uvd,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_attention_mask=instr_mask,
                visual_grid=visual_grid,
                solver_steps=self.motion_solver_steps,
                steps_to_run=self.motion_steps_to_run,
            )
        pooled = self.motion_temporal_pool(motion["motion_hidden"])
        motion_tokens = self.motion_proj(pooled.to(self.motion_proj.weight.dtype))
        # Place the motion tokens at their WORLD positions so the RoPE-3D
        # scene/action fusion stays in the scene's coordinate frame; the UVD
        # geometry is internal to the expert only.
        return motion_tokens, query_xyz.to(motion_tokens.dtype)


class FlowerActor3DMotion2D(Motion2DLatentMixin, FlowerActor3D):
    """3DFA-VLA policy that fuses a 2D MotionExpert latent into scene tokens.

    The policy keeps the regular 3DFA scene encoder / action decoder. The
    2D motion latent is injected only through scene<->motion cross-attention,
    and that cross-attention intentionally uses no 3D RoPE.
    """

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
                 # Training arguments
                 proprio_nhand=None,
                 lv2_batch_size=1):
        if backbone != "florence2":
            raise ValueError(
                f"{type(self).__name__} currently supports backbone='florence2', got {backbone!r}."
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
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size
        )
        self._init_motion_2d_latent(
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
        )

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
        return self._fuse_motion_2d_into_scene(
            base_result,
            rgb3d,
            instr_mask=instr_mask,
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
                 proprio_nhand=None,
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
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size
        )
        proprio_nhist = nhist * (2 if proprio_nhand is None else int(proprio_nhand))
        self.prediction_head = GroundingCondTransformerHead(
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
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
        # relative to absolute for decoder position queries; honors
        # self._relative_position_mode (RoboTwin2 uses "cumulative")
        traj_xyz = self._absolute_query_xyz(query_trajectory, traj_xyz)

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
        self.use_rgb2d_context = False

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
        )[None, :, None, :].expand(len(traj_feats), -1, nhand, -1)
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
            nhist = proprio_feats.shape[1] // nhand
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)  # (B,nhist,nhand=2)
            valid_mask_hist = valid_mask_hist.reshape(B, nhist*nhand).unsqueeze(-1).float()  # (B, nhist*nhand, 1)
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats, valid_mask_hist  # NOTE: mask fake hand feature inside proprio_feats
        )

        # Concatenate context features. The optional 2D camera path is enabled
        # by dataset-specific trainers so shared FlowerActor3D defaults stay
        # unchanged.
        context_feats = [rgb3d_feats]
        context_pos = [rgb3d_pos]
        if self.use_rgb2d_context and rgb2d_feats is not None and rgb2d_pos is not None:
            context_feats.append(rgb2d_feats)
            context_pos.append(rgb2d_pos)
        context_feats.append(instr_feats)
        context_pos.append(instr_pos)
        rgb3d_feats = torch.cat(context_feats, 1)
        rgb3d_pos = torch.cat(context_pos, 1)

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
                motion_tokens=None, motion_pos=None, return_hidden=False):
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
        )[None, :, None, :].expand(len(traj_feats), -1, nhand, -1)
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
            nhist = proprio_feats.shape[1] // nhand
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)
            valid_mask_hist = valid_mask_hist.reshape(B, nhist * nhand).unsqueeze(-1).float()
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats, valid_mask_hist
        )

        context_feats = [rgb3d_feats]
        context_pos = [rgb3d_pos]
        if self.use_rgb2d_context and rgb2d_feats is not None and rgb2d_pos is not None:
            context_feats.append(rgb2d_feats)
            context_pos.append(rgb2d_pos)
        context_feats.append(instr_feats)
        context_pos.append(instr_pos)
        rgb3d_feats = torch.cat(context_feats, 1)
        rgb3d_pos = torch.cat(context_pos, 1)

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

        outputs = [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]
        if return_hidden:
            # Per-waypoint EEF "thought": the trajectory tokens after the shared
            # scene/motion-fused self-attention, before the rotation/position
            # heads. Shape (B, traj_len, nhand, C). Consumed (detached) by the
            # joint-action decoder as chain-of-thought conditioning.
            h_eef = traj_feats.unflatten(1, (traj_len, nhand))
            return outputs, h_eef
        return outputs


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

        context_feats = [rgb3d_feats]
        context_pos = [rgb3d_pos]
        if self.use_rgb2d_context and rgb2d_feats is not None and rgb2d_pos is not None:
            context_feats.append(rgb2d_feats)
            context_pos.append(rgb2d_pos)
        context_feats.append(instr_feats)
        context_pos.append(instr_pos)
        context_feats = torch.cat(context_feats, dim=1)
        context_pos = torch.cat(context_pos, dim=1)
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


class FlowerActor2DFA(BaseDenoiseActor):
    """2D ablation of ``FlowerActor3D`` (a.k.a. ``2dfavla``).

    Identical architecture and parameter count to ``flower_actor_3d`` (without
    the motion latent), but the model never lifts pixels to 3D and uses no 3D
    rotary positional encoding anywhere:

      * ``Encoder2DFA`` discards the point cloud and tags scene tokens with
        camera-id + 2D sinusoidal position embeddings;
      * ``TransformerHead2D`` runs the same DoubleCrossSelfAttention decoder as
        the 3D head but with ``rotary_pe=False`` (the 3D RoPE layer is removed,
        which is parameter-free, so size is unchanged).
    """

    def __init__(self,
                 # Encoder arguments
                 backbone="florence2",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 num_finetune_vision_layers=0,  # accepted for CLI parity; unused
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
                 proprio_nhand=None,
                 # Number of 3D scene cameras to tag with camera-id embeddings
                 # (CALVIN/LIBERO: 2; RoboTwin2 head/left/right: 3).
                 num_camera_ids=2,
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
        proprio_nhist = nhist * (2 if proprio_nhand is None else int(proprio_nhand))
        # Vision-language encoder, runs only once. 2D variant: no 3D lifting.
        self.encoder = Encoder2DFA(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder,
            num_camera_ids=num_camera_ids
        )

        # Action decoder, runs at every denoising timestep. 2D variant: no RoPE.
        self.prediction_head = TransformerHead2D(
            embedding_dim=embedding_dim,
            nhist=proprio_nhist,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6
        )


class FlowerActor2DMotion2D(Motion2DLatentMixin, FlowerActor2DFA):
    """2DFA-VLA policy with a 2D MotionExpert scene-fusion latent."""

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
                 # Training arguments
                 proprio_nhand=None,
                 lv2_batch_size=1):
        if backbone != "florence2":
            raise ValueError(
                f"{type(self).__name__} currently supports backbone='florence2', got {backbone!r}."
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
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size
        )
        self._init_motion_2d_latent(
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
        )

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
        return self._fuse_motion_2d_into_scene(
            base_result,
            rgb3d,
            instr_mask=instr_mask,
        )


class FlowerActor2DCLIP(BaseDenoiseActor):
    """Legacy CLIP-only 2D actor (dormant).

    Kept for reference. The active 2D ablation of ``flower_actor_3d`` is
    ``FlowerActor2DFA`` below, registered as ``flower_actor_2d``.
    """

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
                fps_scene_feats, fps_scene_pos, valid_mask=None):
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
            valid_mask: accepted for API parity with the 3D head; unused (the 2D
                ablation does not mask padded hands).

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
        )[None, :, None, :].expand(len(traj_feats), -1, nhand, -1)
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


class JointFlowHead(nn.Module):
    """Non-RoPE rectified-flow head over joint-space action tokens.

    Joint angles are not spatial, so -- unlike the EEF head -- this head uses NO
    3D RoPE. ``H`` joint-action tokens (one per horizon step, x nhand) carry a
    temporal positional embedding, self-attend among themselves, and cross-attend
    (content-only) to the concatenated conditioning context
    ``{scene, language, motion latent, EEF "thought"}``. Each layer is
    cross-attn -> self-attn, both AdaLN-conditioned on the denoising timestep and
    the joint proprio. Outputs joint velocity (flow) + a gripper logit.
    """

    def __init__(self, embedding_dim, joint_action_dim, num_attn_heads,
                 num_layers=4, joint_nhist=1, gripper_dim=1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.joint_action_dim = joint_action_dim
        self.gripper_dim = gripper_dim

        self.joint_in = nn.Linear(joint_action_dim, embedding_dim)
        self.step_emb = SinusoidalPosEmb(embedding_dim)            # temporal PE over horizon
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.proprio_emb = nn.Sequential(
            nn.Linear(embedding_dim * joint_nhist, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.cross_layers = nn.ModuleList([
            AttentionModule(1, embedding_dim, 4 * embedding_dim, dropout,
                            num_attn_heads, rotary_pe=False, use_adaln=True,
                            is_self=False)
            for _ in range(num_layers)
        ])
        self.self_layers = nn.ModuleList([
            AttentionModule(1, embedding_dim, 4 * embedding_dim, dropout,
                            num_attn_heads, rotary_pe=False, use_adaln=True,
                            is_self=True)
            for _ in range(num_layers)
        ])

        self.joint_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, joint_action_dim),
        )
        self.gripper_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, gripper_dim),
        )

    def _ada(self, timesteps, proprio_feats):
        # timesteps: (B,)  proprio_feats: (B, joint_nhist, C)
        t = self.time_emb(timesteps)
        p = self.proprio_emb(proprio_feats.flatten(1))
        return t + p

    def forward(self, noised_joints, timesteps, context_feats, proprio_feats,
                context_kpm=None):
        """
        noised_joints: (B, H*nhand, joint_action_dim)
        timesteps:     (B,)
        context_feats: (B, M, C) concatenated conditioning tokens
        proprio_feats: (B, joint_nhist, C)
        Returns: (B, H*nhand, joint_action_dim + gripper_dim)
        """
        B, T, _ = noised_joints.shape
        x = self.joint_in(noised_joints)
        step_pos = self.step_emb(
            torch.arange(T, device=x.device)
        )[None].expand(B, -1, -1)                                   # (B, T, C)
        ada = self._ada(timesteps, proprio_feats)
        for i in range(self.num_layers):
            x = self.cross_layers[i](
                x, context_feats,
                seq2_key_padding_mask=context_kpm,
                seq1_sem_pos=step_pos,
                ada_sgnl=ada,
            )[-1]
            x = self.self_layers[i](
                x, x,
                seq1_sem_pos=step_pos,
                ada_sgnl=ada,
            )[-1]
        return torch.cat(
            (self.joint_predictor(x), self.gripper_predictor(x)), -1
        )


class FlowerActor3DMotionLatentJoint(FlowerActor3DMotionLatent):
    """Joint-angle decoder on top of the 3D motion-latent EEF actor.

    The EEF expert (``MotionLatentTransformerHead``) is kept and co-trained by its
    normal flow-matching loss (calibration-dependent, EEF space). Its per-waypoint
    hidden -- read via a single one-step partial-denoise forward at the noise point
    (mirroring the motion latent's ``extract_one_ode_step_latent``) -- is the
    *3D chain-of-thought* that conditions a separate rectified-flow ``JointFlowHead``
    predicting calibration-free joint actions. The EEF feature is detached
    (stop-grad bridge), so the joint loss never warps the EEF representation and the
    feature extraction is identical at train and eval.
    """

    def __init__(self,
                 *args,
                 joint_action_dim=7,
                 joint_proprio_dim=8,
                 gripper_dim=1,
                 joint_num_layers=4,
                 joint_embedding_dim=None,   # None/0 -> share the backbone dim
                 joint_num_attn_heads=None,  # None/0 -> share the backbone heads
                 joint_loss_weight=1.0,
                 joint_action_weight=30.0,
                 joint_norm_clamp=2.0,
                 nhist=3,
                 nhand=1,
                 num_attn_heads=9,
                 embedding_dim=60,
                 # Motion-expert kwargs declared explicitly (not absorbed by
                 # **kwargs) so the trainer's signature-based optional-kwarg
                 # injection still passes them through to the parent.
                 motion_expert_config=None,
                 motion_expert_checkpoint=None,
                 motion_solver_steps=10,
                 motion_steps_to_run=1,
                 motion_finetune=True,
                 motion_scene_fusion=True,
                 motion_action_fusion=True,
                 proprio_nhand=None,
                 **kwargs):
        super().__init__(
            *args,
            nhist=nhist,
            nhand=nhand,
            num_attn_heads=num_attn_heads,
            embedding_dim=embedding_dim,
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            motion_action_fusion=motion_action_fusion,
            proprio_nhand=proprio_nhand,
            **kwargs,
        )
        self.joint_action_dim = int(joint_action_dim)
        self.joint_proprio_dim = int(joint_proprio_dim)
        self.gripper_dim = int(gripper_dim)
        self.joint_loss_weight = float(joint_loss_weight)
        self.joint_action_weight = float(joint_action_weight)
        joint_nhist = int(nhist) * int(nhand)

        # Joint head can be sized independently of the backbone. When its dim
        # differs, the conditioning context (scene + language + h_eef, all at
        # embedding_dim) is projected down/up to the joint dim.
        jdim = int(joint_embedding_dim) if joint_embedding_dim else int(embedding_dim)
        jheads = int(joint_num_attn_heads) if joint_num_attn_heads else int(num_attn_heads)
        assert jdim % jheads == 0, (
            f"joint_embedding_dim ({jdim}) must be divisible by "
            f"joint_num_attn_heads ({jheads})."
        )
        self.joint_proprio_in = nn.Linear(joint_proprio_dim, jdim)
        self.joint_context_proj = (
            nn.Linear(embedding_dim, jdim) if jdim != embedding_dim else nn.Identity()
        )
        self.joint_head = JointFlowHead(
            embedding_dim=jdim,
            joint_action_dim=joint_action_dim,
            num_attn_heads=jheads,
            num_layers=joint_num_layers,
            joint_nhist=joint_nhist,
            gripper_dim=gripper_dim,
        )
        # Per-joint [min, max] normalizer for the (anchor-relative) arm-joint
        # targets. Identity by default; the trainer overwrites from dataset stats.
        self.joint_normalizer = nn.Parameter(
            torch.stack([
                -torch.ones(joint_action_dim),
                torch.ones(joint_action_dim),
            ]),
            requires_grad=False,
        )
        # Symmetric clamp on the *normalized* arm-joint targets. The range is set
        # so ~99.9% of values map into [-1, 1]; this caps the rare tail (e.g. one
        # CALVIN joint reaches ~2.14) so all targets stay in [-clamp, clamp].
        self.joint_norm_clamp = float(joint_norm_clamp)

    # ---- joint normalization ------------------------------------------------
    def normalize_joints(self, x):
        lo, hi = self.joint_normalizer[0], self.joint_normalizer[1]
        x_n = (x - lo) / (hi - lo).clamp_min(1e-6) * 2.0 - 1.0
        if self.joint_norm_clamp and self.joint_norm_clamp > 0:
            x_n = x_n.clamp(-self.joint_norm_clamp, self.joint_norm_clamp)
        return x_n

    def unnormalize_joints(self, x):
        lo, hi = self.joint_normalizer[0], self.joint_normalizer[1]
        return (x + 1.0) / 2.0 * (hi - lo).clamp_min(1e-6) + lo

    # ---- EEF chain-of-thought + joint context -------------------------------
    def _eef_hidden(self, fixed_inputs, B, traj_len, nhand, device, dtype):
        """EEF chain-of-thought: the EEF head's per-waypoint velocity hidden read
        at the early/noise stage (rectified-flow t=1), mirroring the motion
        latent's noise-point read. Random noise (in-distribution for the velocity
        field; the conditioning dominates the hidden) and DETACHED -- the EEF head
        is supervised only by its own pos+rot flow loss, the joint loss never
        backprops into it. Returns (B, traj_len, nhand, C)."""
        out_dim = 6 if self._rotation_format in ('euler', 'axisangle') else 9
        noise_traj = torch.randn(
            B, traj_len, nhand, out_dim, device=device, dtype=dtype
        )
        timesteps = torch.ones(B, device=device, dtype=dtype)  # fully-noised point
        with torch.no_grad():
            _, h_eef = self.policy_forward_pass(
                noise_traj, timesteps, fixed_inputs, return_hidden=True
            )
        return h_eef.detach()

    def _joint_context(self, fixed_inputs, h_eef):
        # Joint head conditions on the scene-MOTION-fused scene latent
        # (rgb3d_feats already has the motion residual added in encode_inputs),
        # language, and the EEF chain-of-thought. Motion enters only via the
        # fused scene latent and h_eef -- motion_tokens are NOT attended directly.
        (
            _query_trajectory,
            rgb3d_feats, _pcd,
            _rgb2d_feats, _rgb2d_pos,
            instr_feats, _instr_pos,
            _proprio_feats,
            _fps_scene_feats, _fps_scene_pos,
            _motion_tokens, _motion_pos,
        ) = fixed_inputs
        h_eef_flat = h_eef.flatten(1, 2)  # (B, traj_len*nhand, C)
        context = torch.cat(
            [rgb3d_feats, instr_feats, h_eef_flat], dim=1
        )
        context = self.joint_context_proj(context)  # -> joint dim (Identity if same)
        return context, None

    def _embed_joint_proprio(self, joint_proprio):
        # joint_proprio: (B, nhist, nhand, joint_proprio_dim)
        B = joint_proprio.shape[0]
        flat = joint_proprio.reshape(B, -1, self.joint_proprio_dim)
        return self.joint_proprio_in(flat.to(self.joint_proprio_in.weight.dtype))

    # ---- losses / sampling --------------------------------------------------
    def _joint_flow_loss(self, fixed_inputs, h_eef, gt_joint, joint_proprio):
        gt_arm = gt_joint[..., :self.joint_action_dim]
        gt_grip = gt_joint[..., self.joint_action_dim:]
        gt_arm_n = self.normalize_joints(gt_arm)
        B, H, nh, _ = gt_arm.shape

        noise = torch.randn_like(gt_arm_n)
        t = self.position_scheduler.sample_noise_step(B, gt_arm_n.device)
        zt = self.position_scheduler.add_noise(gt_arm_n, noise, t)

        context, kpm = self._joint_context(fixed_inputs, h_eef)
        proprio_emb = self._embed_joint_proprio(joint_proprio)
        pred = self.joint_head(zt.flatten(1, 2), t, context, proprio_emb, kpm)

        pred_v = pred[..., :self.joint_action_dim]
        pred_grip = pred[..., self.joint_action_dim:]
        target = self.position_scheduler.prepare_target(
            noise.flatten(1, 2), gt_arm_n.flatten(1, 2)
        )
        arm_loss = F.l1_loss(pred_v, target, reduction='mean')
        grip_loss = F.binary_cross_entropy_with_logits(
            pred_grip, gt_grip.flatten(1, 2)
        )
        return self.joint_action_weight * arm_loss + grip_loss

    def compute_loss(self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
                     instruction, proprio, instr_mask=None, instr_lens=None,
                     gt_joint_trajectory=None, joint_proprio=None):
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )
        # (1) Supervise the EEF expert with a position+rotation flow loss only
        #     (no openness -- the gripper is owned solely by the joint head).
        eef_loss = self.flow_loss_given_fixed(
            gt_trajectory, trajectory_mask, fixed_inputs, include_openness=False
        )
        # (2) Read the detached EEF chain-of-thought, then the joint flow loss.
        B, H, nh, _ = gt_joint_trajectory.shape
        h_eef = self._eef_hidden(
            fixed_inputs, B, H, nh, gt_joint_trajectory.device, eef_loss.dtype
        )
        joint_loss = self._joint_flow_loss(
            fixed_inputs, h_eef, gt_joint_trajectory, joint_proprio
        )
        return eef_loss + self.joint_loss_weight * joint_loss

    @torch.no_grad()
    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, pcd, instruction,
                           proprio, instr_mask=None, instr_lens=None,
                           joint_proprio=None):
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )
        B, H, nh = trajectory_mask.shape
        device = trajectory_mask.device
        dtype = self.joint_proprio_in.weight.dtype
        h_eef = self._eef_hidden(fixed_inputs, B, H, nh, device, dtype)
        context, kpm = self._joint_context(fixed_inputs, h_eef)
        proprio_emb = self._embed_joint_proprio(joint_proprio)

        zt = torch.randn(B, H * nh, self.joint_action_dim, device=device)
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        pred = None
        for t_ind, t in enumerate(self.position_scheduler.timesteps):
            pred = self.joint_head(
                zt, t * torch.ones(B, device=device), context, proprio_emb, kpm
            )
            v = pred[..., :self.joint_action_dim]
            zt = self.position_scheduler.step(v, t_ind, zt).prev_sample
        arm = self.unnormalize_joints(zt)
        grip = pred[..., self.joint_action_dim:].sigmoid()
        joints = torch.cat((arm, grip), -1).unflatten(1, (H, nh))
        return joints

    def forward(self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
                instruction, proprio, run_inference=False, instr_mask=None,
                instr_lens=None, gt_joint_trajectory=None, joint_proprio=None,
                **kwargs):
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio,
                instr_mask=instr_mask, instr_lens=instr_lens,
                joint_proprio=joint_proprio,
            )
        return self.compute_loss(
            gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd, instruction,
            proprio, instr_mask=instr_mask, instr_lens=instr_lens,
            gt_joint_trajectory=gt_joint_trajectory, joint_proprio=joint_proprio,
        )


class FlowerActor2DJoint(FlowerActor2DFA):
    """2D visual-language joint-action baseline.

    This is the ablation counterpart to ``flower_actor_3d_motion_latent_joint``:
    it keeps the same rectified-flow ``JointFlowHead`` and joint normalization, but
    removes the MotionExpert, scene-motion fusion, and EEF chain-of-thought branch.
    The joint head conditions only on 2DFA visual tokens and instruction tokens.
    """

    def __init__(self,
                 *args,
                 joint_action_dim=7,
                 joint_proprio_dim=8,
                 gripper_dim=1,
                 joint_num_layers=4,
                 joint_embedding_dim=None,
                 joint_num_attn_heads=None,
                 joint_loss_weight=1.0,
                 joint_action_weight=30.0,
                 joint_norm_clamp=2.0,
                 nhist=3,
                 nhand=1,
                 num_attn_heads=9,
                 embedding_dim=60,
                 proprio_nhand=None,
                 num_camera_ids=2,
                 **kwargs):
        super().__init__(
            *args,
            nhist=nhist,
            nhand=nhand,
            num_attn_heads=num_attn_heads,
            embedding_dim=embedding_dim,
            proprio_nhand=proprio_nhand,
            num_camera_ids=num_camera_ids,
            **kwargs,
        )

        # This baseline predicts joints directly; remove the inherited EEF decoder
        # so DDP does not carry unused trainable parameters.
        del self.traj_encoder
        del self.prediction_head
        for module_name in ("curr_gripper_embed", "gripper_context_head"):
            module = getattr(self.encoder, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False

        self.joint_action_dim = int(joint_action_dim)
        self.joint_proprio_dim = int(joint_proprio_dim)
        self.gripper_dim = int(gripper_dim)
        self.joint_loss_weight = float(joint_loss_weight)
        self.joint_action_weight = float(joint_action_weight)
        joint_nhist = int(nhist) * int(nhand)

        jdim = int(joint_embedding_dim) if joint_embedding_dim else int(embedding_dim)
        jheads = int(joint_num_attn_heads) if joint_num_attn_heads else int(num_attn_heads)
        assert jdim % jheads == 0, (
            f"joint_embedding_dim ({jdim}) must be divisible by "
            f"joint_num_attn_heads ({jheads})."
        )
        self.joint_proprio_in = nn.Linear(joint_proprio_dim, jdim)
        self.joint_context_proj = (
            nn.Linear(embedding_dim, jdim) if jdim != embedding_dim else nn.Identity()
        )
        self.joint_head = JointFlowHead(
            embedding_dim=jdim,
            joint_action_dim=joint_action_dim,
            num_attn_heads=jheads,
            num_layers=joint_num_layers,
            joint_nhist=joint_nhist,
            gripper_dim=gripper_dim,
        )
        self.joint_normalizer = nn.Parameter(
            torch.stack([
                -torch.ones(joint_action_dim),
                torch.ones(joint_action_dim),
            ]),
            requires_grad=False,
        )
        self.joint_norm_clamp = float(joint_norm_clamp)

    normalize_joints = FlowerActor3DMotionLatentJoint.normalize_joints
    unnormalize_joints = FlowerActor3DMotionLatentJoint.unnormalize_joints
    _embed_joint_proprio = FlowerActor3DMotionLatentJoint._embed_joint_proprio
    _joint_flow_loss = FlowerActor3DMotionLatentJoint._joint_flow_loss

    def _joint_context(self, fixed_inputs, h_eef=None):
        del h_eef
        (
            _query_trajectory,
            rgb3d_feats, _pcd,
            rgb2d_feats, _rgb2d_pos,
            instr_feats, _instr_pos,
            _proprio_feats,
            _fps_scene_feats, _fps_scene_pos,
        ) = fixed_inputs
        context_parts = [rgb3d_feats]
        if rgb2d_feats is not None:
            context_parts.append(rgb2d_feats)
        context_parts.append(instr_feats)
        context = torch.cat(context_parts, dim=1)
        context = self.joint_context_proj(context)
        return context, None

    def compute_loss(self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
                     instruction, proprio, instr_mask=None, instr_lens=None,
                     gt_joint_trajectory=None, joint_proprio=None):
        del gt_trajectory, trajectory_mask
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )
        joint_loss = self._joint_flow_loss(
            fixed_inputs, None, gt_joint_trajectory, joint_proprio
        )
        return self.joint_loss_weight * joint_loss

    @torch.no_grad()
    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, pcd, instruction,
                           proprio, instr_mask=None, instr_lens=None,
                           joint_proprio=None):
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
            instr_mask=instr_mask, instr_lens=instr_lens,
        )
        B, H, nh = trajectory_mask.shape
        device = trajectory_mask.device
        context, kpm = self._joint_context(fixed_inputs, None)
        proprio_emb = self._embed_joint_proprio(joint_proprio)

        zt = torch.randn(B, H * nh, self.joint_action_dim, device=device)
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        pred = None
        for t_ind, t in enumerate(self.position_scheduler.timesteps):
            pred = self.joint_head(
                zt, t * torch.ones(B, device=device), context, proprio_emb, kpm
            )
            v = pred[..., :self.joint_action_dim]
            zt = self.position_scheduler.step(v, t_ind, zt).prev_sample
        arm = self.unnormalize_joints(zt)
        grip = pred[..., self.joint_action_dim:].sigmoid()
        return torch.cat((arm, grip), -1).unflatten(1, (H, nh))

    def forward(self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd,
                instruction, proprio, run_inference=False, instr_mask=None,
                instr_lens=None, gt_joint_trajectory=None, joint_proprio=None,
                **kwargs):
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio,
                instr_mask=instr_mask, instr_lens=instr_lens,
                joint_proprio=joint_proprio,
            )
        return self.compute_loss(
            gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd, instruction,
            proprio, instr_mask=instr_mask, instr_lens=instr_lens,
            gt_joint_trajectory=gt_joint_trajectory, joint_proprio=joint_proprio,
        )
