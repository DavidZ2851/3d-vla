import torch
from torch import nn
import einops

from ..encoder.multimodal.encoder3d import Encoder as Encoder3D
from ..encoder.multimodal.encoder2d import Encoder as Encoder2D
from ..utils.layers import DoubleCrossSelfAttentionModule, AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding3D

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


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
        self._pending_grounding_3d = grounding_3d
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
        if grounding_3d is not None:
            if grounding_3d.ndim == 3 and grounding_3d.shape[1] == 1:
                grounding_3d = grounding_3d.squeeze(1)
            grounding_feats = self.grounding_proj(grounding_3d).unsqueeze(1)
            grounding_pos = grounding_3d.unsqueeze(1)
        else:
            grounding_feats = None
            grounding_pos = None

        return base_result + (grounding_feats, grounding_pos)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            grounding_feats, grounding_pos
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
                    time_embs, traj_len, token_mask_float=None, token_mask_bool=None):
        position_features = self.position_self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
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
                    time_embs, traj_len, token_mask_float=None, token_mask_bool=None):
        rotation_features = self.rotation_self_attn(
            seq=traj_feats,
            seq0=rgb3d_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation


class GroundingCondTransformerHead(TransformerHead):
    """TransformerHead variant that injects an optional grounding token."""

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, valid_mask=None,
                grounding_feats=None, grounding_pos=None):
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
        if grounding_feats is not None and grounding_pos is not None:
            context_feats = torch.cat((context_feats, grounding_feats), dim=1)
            context_pos = torch.cat((context_pos, grounding_pos), dim=1)

        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(context_pos)

        traj_feats = self.self_attn(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs
        )[-1]

        rotation = self.predict_rot(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool
        )
        position, position_features = self.predict_pos(
            traj_feats, context_feats, rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1],
            token_mask_float=token_mask_float, token_mask_bool=token_mask_bool
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
