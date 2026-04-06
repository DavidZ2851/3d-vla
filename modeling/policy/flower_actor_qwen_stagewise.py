import einops, torch
from torch import nn

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor
from .flower_actor import TransformerHead
from ..encoder.multimodal.encoder3d_qwen_stagewise import Qwen3VLEncoder3DStagewise
from ..utils.layers import DoubleCrossSelfAttentionModule


class QwenFlowerActor3DStagewise(BaseDenoiseActor):
    """
    Parallel Qwen policy variant with decoder-depth stagewise multiscale routing.
    Existing qwen_flower_actor_3d remains unchanged.
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
        llm_lora_target_modules="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        gradient_checkpointing=False,
        use_multiscale=True,
        multiscale_stage_count=3,
        multiscale_stage_init="biased_low_mid_high",
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

        self.encoder = Qwen3VLEncoder3DStagewise(
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
            multiscale_stage_count=multiscale_stage_count,
            multiscale_stage_init=multiscale_stage_init,
        )

        self.prediction_head = QwenStagewiseTransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ["euler", "axisangle"] else 6,
        )
        if nhand == 1:
            self.prediction_head.hand_embed = None

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        (
            query_trajectory,
            rgb3d_feats,
            pcd,
            rgb2d_feats,
            rgb2d_pos,
            instr_feats,
            instr_pos,
            proprio_feats,
            fps_scene_feats,
            fps_scene_pos,
            stage_rgb3d_feats,
        ) = fixed_inputs

        trajectory_feats = self.traj_encoder(trajectory)
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:
            traj_xyz = query_trajectory[..., :3] + traj_xyz.cumsum(dim=1)

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
            stage_rgb3d_feats=stage_rgb3d_feats,
        )


class QwenStagewiseTransformerHead(TransformerHead):
    """
    Qwen-specific parallel head:
    - shared decoder depth reads early/mid/late stage memories in thirds,
    - output heads keep late-stage memory by default (risk-controlled v1).
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

        l0, l1, l2 = _split_depth_into_thirds(num_shared_attn_layers)
        self.shared_stage_depths = (l0, l1, l2)
        self.self_attn_early = _make_cross_self_block(
            l0, embedding_dim, num_attn_heads, rotary_pe
        )
        self.self_attn_mid = _make_cross_self_block(
            l1, embedding_dim, num_attn_heads, rotary_pe
        )
        self.self_attn_late = _make_cross_self_block(
            l2, embedding_dim, num_attn_heads, rotary_pe
        )
        # Disable base monolithic shared stack; stagewise branches replace it.
        self.self_attn = None

    def _run_shared_block(self, block, traj_feats, context_feats,
        rel_traj_pos, rel_scene_pos, time_embs, token_mask_float, token_mask_bool,):
        if block is None:
            return traj_feats
        return block(
            seq=traj_feats,
            seq0=context_feats,
            seq_token_mask=token_mask_float,
            seq_token_mask_bool=token_mask_bool,
            seq_pos_0=rel_traj_pos,
            seq0_pos=rel_scene_pos,
            seq_pos_1=rel_traj_pos,
            ada_sgnl=time_embs,
        )[-1]

    def forward(self, traj_feats, trajectory, timesteps,
        rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
        instr_feats, instr_pos, proprio_feats,
        fps_scene_feats, fps_scene_pos, valid_mask=None,
        stage_rgb3d_feats=None,
    ):
        batch_size, traj_len, nhand, _ = trajectory.shape

        if nhand > 1 and self.hand_embed is not None:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, "b l h c -> b (l h) c")
        trajectory = einops.rearrange(trajectory, "b l h c -> b (l h) c")

        if valid_mask is not None:
            token_mask = valid_mask.squeeze(-1).reshape(batch_size, -1)
            token_mask_float = token_mask.float()
            token_mask_bool = token_mask == 0
        else:
            token_mask_float = None
            token_mask_bool = None

        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(batch_size, 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, "b l h c -> b (l h) c")
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
            nhist = proprio_feats.shape[1] // 2
            valid_mask_hist = valid_mask_hist.expand(-1, nhist, -1)
            valid_mask_hist = valid_mask_hist.reshape(batch_size, nhist * nhand).unsqueeze(-1).float()
        time_embs = self.encode_denoising_timestep(
            timesteps,
            proprio_feats,
            valid_mask_hist,
        )

        if stage_rgb3d_feats is None or len(stage_rgb3d_feats) < 3:
            stage_rgb3d_feats = [rgb3d_feats, rgb3d_feats, rgb3d_feats]

        stage_context_feats = []
        stage_context_pos = torch.cat((rgb3d_pos, instr_pos), dim=1)
        for stage_feats in stage_rgb3d_feats[:3]:
            stage_context_feats.append(torch.cat((stage_feats, instr_feats), dim=1))

        rel_traj_pos = self.relative_pe_layer(traj_xyz)
        rel_scene_pos = self.relative_pe_layer(stage_context_pos)
        
        traj_feats = self._run_shared_block(self.self_attn_early, traj_feats, stage_context_feats[0],
            rel_traj_pos, rel_scene_pos, time_embs,
            token_mask_float, token_mask_bool,
        )
        traj_feats = self._run_shared_block(self.self_attn_mid, traj_feats, stage_context_feats[1],
            rel_traj_pos, rel_scene_pos, time_embs,
            token_mask_float, token_mask_bool,
        )
        traj_feats = self._run_shared_block(self.self_attn_late, traj_feats, stage_context_feats[2],
            rel_traj_pos, rel_scene_pos, time_embs,
            token_mask_float, token_mask_bool,
        )

        rotation = self.predict_rot(
            traj_feats, stage_context_feats[2], rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1], token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
        )
        position, position_features = self.predict_pos(
            traj_feats, stage_context_feats[2], rel_traj_pos, rel_scene_pos,
            time_embs, traj_feats.shape[1], token_mask_float=token_mask_float, token_mask_bool=token_mask_bool,
        )
        openess = self.openess_predictor(position_features)
        return [torch.cat((position, rotation, openess), dim=-1).unflatten(1, (traj_len, nhand))]


def _split_depth_into_thirds(num_layers):
    base = int(num_layers) // 3
    rem = int(num_layers) - 3 * base
    out = [base, base, base]
    out[2] += rem
    return out

def _make_cross_self_block(num_layers, embedding_dim, num_attn_heads, rotary_pe):
    if num_layers <= 0:
        return None
    return DoubleCrossSelfAttentionModule(
        num_layers=num_layers,
        d_model=embedding_dim,
        dim_fw=4 * embedding_dim,
        dropout=0.1,
        n_heads=num_attn_heads,
        pre_norm=False,
        rotary_pe_0=rotary_pe,
        rotary_pe_1=rotary_pe,
        use_adaln=True,
    )
