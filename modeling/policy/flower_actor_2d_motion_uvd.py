"""Plain-2D CALVIN VLA fused with the UVD MotionExpert (LaMP-style ablation).

Additive counterpart to ``FlowerActor3DMotionLatentUVD``. That model keeps the
full 3D stack -- lifted point clouds, world-XYZ token positions, RoPE-3D in both
the scene<->motion fusion and the action decoder -- and only the motion expert's
internal geometry is UVD. This one strips the 3D stack entirely:

  * ``Encoder2DFA`` never lifts pixels to 3D; scene tokens get camera-id + 2D
    sinusoidal position embeddings (inherited from ``FlowerActor2DFA``);
  * ``TransformerHead2D`` runs the same decoder with ``rotary_pe=False``, so no
    3D RoPE anywhere in action prediction;
  * scene<->motion fusion is ``CrossAttentionNoRoPE`` behind the learned
    ``alpha_scene`` gate -- positions play no part;
  * the motion latent is injected ONLY through the scene tokens (no
    action<->motion fusion), exactly like ``FlowerActor2DMotion2D``.

So the whole policy is a 2D VLA predicting EEF actions, and the only thing that
knows about depth is the UVD expert itself -- which needs ``(u, v, depth)`` by
construction. That depth is read off the front point cloud through CALVIN's
fixed static-camera transform, identically to ``FlowerActor3DMotionLatentUVD``,
so a UVD expert checkpoint transfers between the two without retraining. The
point cloud never reaches a learnable layer of the policy.

Registered as ``flower_actor_2d_motion_uvd``.
"""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from motion_expert.config import load_motion_expert_config
from motion_expert.data import build_regular_query_uv
from motion_expert.model_uvd import MotionExpertUVDModel
from motion_expert.zarr_dataset_uvd import (
    DEFAULT_QUERY_DEPTH_MEAN,
    DEFAULT_QUERY_DEPTH_STD,
)
from utils.calvin_camera_frame import CALVIN_WORLD2CAM

from .flower_actor import (
    CrossAttentionNoRoPE,
    FlowerActor2DFA,
    FlowerActor3DMotionLatentUVD,
    TemporalAttentionPool,
)


class FlowerActor2DMotionUVD(FlowerActor2DFA):
    """2DFA-VLA policy with a UVD MotionExpert scene-fusion latent."""

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
                 # UVD query-depth standardization (camera-frame metres). Defaults
                 # match compute_calvin_uvd_norm_stats.py / the expert's training.
                 query_depth_mean=None,
                 query_depth_std=None,
                 # Number of 3D scene cameras to tag with camera-id embeddings.
                 num_camera_ids=2,
                 # Training arguments
                 proprio_nhand=None,
                 lv2_batch_size=1):
        if backbone != "florence2":
            raise ValueError(
                f"{type(self).__name__} currently supports backbone='florence2', "
                f"got {backbone!r}."
            )
        if motion_expert_config is None:
            raise ValueError(
                "flower_actor_2d_motion_uvd requires --motion_expert_config."
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
            num_camera_ids=num_camera_ids,
            lv2_batch_size=lv2_batch_size,
        )

        motion_config = load_motion_expert_config(str(motion_expert_config))
        if str(motion_config.backbone_type).lower() != "florence2":
            raise ValueError(
                "flower_actor_2d_motion_uvd expects a Florence2 UVD Motion Expert "
                f"config, got backbone_type={motion_config.backbone_type!r}."
            )
        motion_feature_dim = int(self.encoder.cond_linear.in_features)
        # UVD conditioning bank = pooled global-visual tokens + text tokens (no
        # per-query tokens), matching the UVD expert's training-time layout.
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

        self.query_depth_mean = float(
            DEFAULT_QUERY_DEPTH_MEAN if query_depth_mean is None else query_depth_mean
        )
        self.query_depth_std = max(
            float(DEFAULT_QUERY_DEPTH_STD if query_depth_std is None else query_depth_std),
            1e-6,
        )
        # world -> static cam, for recovering the expert's camera-frame depth from
        # the front point cloud. Non-persistent buffer: follows .to()/.cuda() moves
        # but is never written to / read from checkpoints.
        self.register_buffer(
            "world2cam",
            torch.from_numpy(np.ascontiguousarray(CALVIN_WORLD2CAM)).float(),
            persistent=False,
        )

        self.motion_temporal_pool = TemporalAttentionPool(
            motion_config.sparse_decoder.hidden_dim,
            num_attn_heads,
            dropout=motion_config.dropout,
        )
        self.motion_proj = nn.Linear(
            motion_config.sparse_decoder.hidden_dim, embedding_dim
        )
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

    # Same checkpoint layout as the 3D UVD policy (both hold a MotionExpertUVDModel
    # under `motion_expert.`), so reuse its loader rather than forking the
    # accepted-prefix list.
    _load_motion_expert_checkpoint = (
        FlowerActor3DMotionLatentUVD._load_motion_expert_checkpoint
    )

    def _build_motion_queries(self, pcd):
        """Regular query grid + front-camera XYZ, identical to the 3D UVD policy.

        The point cloud is used ONLY here, to look up the expert's query depth;
        it never reaches a learnable layer, so the policy itself stays 2D.
        """
        if pcd is None:
            raise ValueError(
                "flower_actor_2d_motion_uvd needs the front point cloud to build "
                "the UVD expert's query depth."
            )
        cam_pcd = pcd[:, 0].float()
        batch_size, _, height, width = cam_pcd.shape
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

    def _query_xyz_to_cam_depth(self, query_xyz):
        """Camera-frame metric depth (B, P) for world-frame query points.

        The front cloud is ``world = T_world_cam[:3] @ [cam_xyz, 1]`` with
        ``cam_xyz.z = -depth``, so ``depth = -(world2cam @ [world, 1]).z``.
        """
        w2c = self.world2cam.to(query_xyz.dtype)
        cam_xyz = query_xyz @ w2c[:3, :3].t() + w2c[:3, 3]
        return -cam_xyz[..., 2]

    def _build_motion_tokens(self, query_uv, query_xyz, instr_mask=None):
        visual_tokens = getattr(self.encoder, "_last_motion_visual_tokens", None)
        visual_grid = getattr(self.encoder, "_last_motion_visual_grid", None)
        text_tokens = getattr(self.encoder, "_last_motion_text_tokens", None)
        if visual_tokens is None or visual_grid is None or text_tokens is None:
            raise RuntimeError(
                "UVD motion latent features were not cached by the Florence2 "
                "encoder. Ensure backbone='florence2' and encode_inputs ran "
                "before motion extraction."
            )
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
        return self.motion_proj(pooled.to(self.motion_proj.weight.dtype))

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
        (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = super().encode_inputs(
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )
        query_uv, query_xyz = self._build_motion_queries(pcd)
        motion_tokens = self._build_motion_tokens(
            query_uv, query_xyz, instr_mask=instr_mask
        )
        if self.scene_motion_attn is not None:
            # No positions anywhere: gated, RoPE-free scene<->motion fusion.
            rgb3d_feats = rgb3d_feats + self.alpha_scene * self.scene_motion_attn(
                rgb3d_feats, motion_tokens
            )
        return (
            query_trajectory,
            rgb3d_feats, pcd_tokens,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        )
