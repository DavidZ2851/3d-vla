import einops
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F

from ...policy.flower.transformers import RmsNorm
from ...utils.position_encodings import RotaryPositionEncoding3D
from ...utils.layers import AttentionModule
from ..vision.qwen3vl import Qwen3VLBackbone, Qwen3VLNormalize  # noqa: direct import avoids __init__.py side-effects


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    B, N, C = features.shape
    dists = torch.cdist(features, features, p=2)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)
    M = N // subsample_factor
    sampled_inds = density.topk(M, dim=-1, largest=True).indices
    return sampled_inds


class Qwen3VLEncoder3D(nn.Module):
    """
    Uses Qwen3-VL pre-merge vision features (stride 16, dim=vision_hidden_size) for scene tokens and fused LLM hidden states (dim=llm_hidden_size) for language tokens.
    Both modalities use ``Linear(RmsNorm(x))`` into ``embedding_dim`` (bias-free linear): same pattern as Florence2 text in ``Encoder.encode_florence2``, extended to vision tokens.
    """
    def __init__(
        self,
        backbone="qwen3vl",
        embedding_dim=60,
        nhist=1,
        num_attn_heads=9,
        fps_subsampling_factor=5,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        backbone_module=None,
        normalize_module=None,
        **kwargs,
    ):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor

        # ---------- Qwen3-VL backbone ----------
        self.backbone = backbone_module or Qwen3VLBackbone()
        self.normalize = normalize_module or Qwen3VLNormalize()
        self.finetune_backbone = finetune_backbone
        self.num_finetune_vision_layers = num_finetune_vision_layers

        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone
        if num_finetune_vision_layers > 0 and not finetune_backbone:
            visual = self.backbone.vlm.visual
            for block in visual.blocks[-num_finetune_vision_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in visual.merger.parameters():
                p.requires_grad = True
            for merger in visual.deepstack_merger_list:
                for p in merger.parameters():
                    p.requires_grad = True

        # ---------- Bridge layers (trained from scratch) ----------
        self.vis_norm = RmsNorm(self.backbone.vision_hidden_size)
        self.vis_proj = nn.Linear(
            self.backbone.vision_hidden_size, embedding_dim, bias=False
        )
        # Text: cond_linear(cond_norm(lang_feats)); LLM dim is Qwen hidden size.
        self.cond_norm = RmsNorm(self.backbone.llm_hidden_size)
        self.cond_linear = nn.Linear(
            self.backbone.llm_hidden_size, embedding_dim, bias=False
        )

        # ---------- 3D proprioception encoder (reused from Encoder3D) ----------
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            n_heads=num_attn_heads,
            rotary_pe=True,
            use_adaln=False,
            pre_norm=False,
        )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio, instr_mask=None):
        """
        Args:
            rgb3d:       (B, ncam, 3, H, W) in [0, 1]
            rgb2d:       ignored
            pcd:         (B, ncam, 3, H, W)
            instruction: (B, L) tokenised text
            proprio:     (B, nhist*nhand, D)
            instr_mask:  (B, L) optional; 1 = real token, 0 = pad (from Qwen tokenizer)

        Returns:
            Same 9-tuple as ``Encoder3D.forward``:
            (rgb3d_feats, pcd, rgb2d_feats, rgb2d_pos,
             instr_feats, instr_pos, proprio_feats,
             fps_scene_feats, fps_scene_pos)
        """
        rgb3d_normed = self.normalize(rgb3d)

        # ---- Qwen3-VL forward ----
        needs_grad = self.finetune_backbone or self.num_finetune_vision_layers > 0
        grad_ctx = nullcontext() if needs_grad else torch.no_grad()
        with grad_ctx:
            pre_merge, text_hidden = self.backbone(
                rgb3d_normed, instruction, text_attention_mask=instr_mask
            )
        # pre_merge:   (B, ncam, h16, w16, C_vis)
        # text_hidden: (B, L, C_llm)

        B, ncam, h16, w16, _ = pre_merge.shape

        # ---- Vision: RmsNorm on C_vis then Linear → embedding_dim (same pattern as text) ----
        rgb3d_feats = pre_merge.reshape(B, ncam * h16 * w16, -1)
        rgb3d_feats = self.vis_proj(self.vis_norm(rgb3d_feats))

        # ---- Language: same pattern as Florence2 (RmsNorm + bias-free Linear) ----
        instr_feats = self.cond_linear(self.cond_norm(text_hidden))

        # ---- Align PCD to the stride-16 grid ----
        num_cameras = pcd.shape[1]
        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (h16, w16),
            mode="bilinear",
        )
        pcd_flat = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=num_cameras
        )

        # ---- Language "position" = last gripper xyz ----
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # ---- Proprioception encoding ----
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd_flat)

        # ---- FPS subsampling ----
        fps_feats, fps_pos = self.run_fps(rgb3d_feats, pcd_flat)

        return (
            rgb3d_feats, pcd_flat,
            None, None,
            instr_feats, instr_pos,
            proprio_feats,
            fps_feats, fps_pos,
        )

    def encode_proprio(self, proprio, context_feats, context_pos):
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).expand(
            len(proprio), -1, -1
        )
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        ctx_pos = self.relative_pe_layer(context_pos)
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=ctx_pos,
        )[-1]
        return proprio_feats

    def run_fps(self, features, pos):
        if self.subsampling_factor == 1:
            return features, pos
        bs, npts, ch = features.shape
        inds = density_based_sampler(features, self.subsampling_factor)
        exp = inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_feats = torch.gather(features, 1, exp)
        if pos is None:
            return sampled_feats, None
        exp3 = inds.unsqueeze(-1).expand(-1, -1, 3)
        sampled_pos = torch.gather(pos, 1, exp3)
        return sampled_feats, sampled_pos
