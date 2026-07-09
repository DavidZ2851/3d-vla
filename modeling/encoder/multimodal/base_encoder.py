import math

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...policy.flower.transformers import RmsNorm

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders
from ..vision.vit_fpn import ViTFPN


class Encoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor
        self._backbone_name = backbone
        self.use_rgb2d_context = False

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(backbone)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        if backbone == 'clip':
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

        # Specific to florence2
        if backbone == 'florence2':
            self.cond_linear = nn.Linear(1024, embedding_dim, bias=False)
            self.cond_norm = RmsNorm(1024)
            self.vit_fpn = ViTFPN("res3", embedding_dim, [128, 256, 512, 1024], 1024)

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
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: (B, nt), tokens
            - proprio: (B, nhist, 3+6+X)
            - instr_mask: unused in base encoder (Qwen3-VL only; see Qwen3VLEncoder3D)
            - instr_lens: unused in base encoder; kept for Qwen3-VL API parity

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, N2d, F)
            - rgb2d_pos: (B, N2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, n, F), n < N
            - fps_scene_pos: (B, n, 3), n < N
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
            'florence2': self.encode_florence2
            # 'siglip2_256': self.encode_siglip,
            # 'siglip2_512': self.encode_siglip
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        # Use the current end-effector position as language 'position'
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)
        rgb2d_pos = None
        if rgb2d_feats is not None:
            # 2D-only camera tokens do not have per-pixel depth. Anchor them at
            # the current end-effector xyz so attention can use wrist RGB
            # without pretending each pixel has metric 3D geometry.
            rgb2d_pos = proprio[:, -1:, :3].repeat(1, rgb2d_feats.size(1), 1)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        # FPS based on scene features
        fps_scene_feats, fps_scene_pos = self.run_fps(rgb3d_feats, pcd)

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        )

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features and positional embeddings.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        return None

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        return None, None, None, None

    def encode_florence2(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        def _infer_grid_hw(num_tokens, height, width):
            expected_h = max(1, math.ceil(height / 32))
            expected_w = max(1, math.ceil(width / 32))
            if expected_h * expected_w == num_tokens:
                return expected_h, expected_w

            aspect = width / float(height)
            candidates = []
            for feat_h in range(1, int(math.sqrt(num_tokens)) + 1):
                if num_tokens % feat_h != 0:
                    continue
                feat_w = num_tokens // feat_h
                candidates.append((feat_h, feat_w))
                if feat_h != feat_w:
                    candidates.append((feat_w, feat_h))
            if not candidates:
                raise ValueError(
                    "Cannot infer visual feature grid from "
                    f"{num_tokens} tokens for input {height}x{width}."
                )
            return min(
                candidates,
                key=lambda hw: abs(math.log((hw[1] / float(hw[0])) / aspect)),
            )

        def _resolve_grid_hw(rgb, feats, nc):
            if feats.shape[2] % nc != 0:
                raise ValueError(
                    "Visual token count must be divisible by camera count: "
                    f"{feats.shape[2]} tokens for {nc} cameras."
                )
            tokens_per_cam = feats.shape[2] // nc
            grid_hw = getattr(self.backbone, "_last_feature_hw", None)
            if (
                grid_hw is not None
                and grid_hw[0] * grid_hw[1] == tokens_per_cam
            ):
                return grid_hw
            return _infer_grid_hw(tokens_per_cam, rgb.shape[-2], rgb.shape[-1])

        def _encode_visual_grid(rgb, save_motion_cache=False):
            rgb = self.normalize(rgb)
            feats, text_feats = self.backbone(rgb, text, output_hidden_states=True)
            nc = rgb.shape[1]
            h_feat, w_feat = _resolve_grid_hw(rgb, feats, nc)
            if save_motion_cache:
                motion_visual_tokens = feats[:, -1]
                self._last_motion_visual_grid = einops.rearrange(
                    motion_visual_tokens,
                    'b (nc h w) c -> b nc h w c',
                    nc=nc,
                    h=h_feat,
                    w=w_feat,
                )
                self._last_motion_visual_tokens = motion_visual_tokens
                self._last_motion_text_tokens = text_feats
            feats = self.vit_fpn({
                'res5': einops.rearrange(feats[:, 3], 'b (nc h w) c -> (b nc) c h w', nc=nc, h=h_feat, w=w_feat),
                'res4': einops.rearrange(feats[:, 2], 'b (nc h w) c -> (b nc) c h w', nc=nc, h=h_feat, w=w_feat),
                'res3': einops.rearrange(feats[:, 1], 'b (nc h w) c -> (b nc) c h w', nc=nc, h=h_feat, w=w_feat),
                'res2': einops.rearrange(feats[:, 0], 'b (nc h w) c -> (b nc) c h w', nc=nc, h=h_feat, w=w_feat)
            })
            return feats, text_feats

        rgb3d_feats, instr_feats = _encode_visual_grid(rgb3d, save_motion_cache=True)
        h_feat_pcd, w_feat_pcd = rgb3d_feats.shape[-2:]
        nc = rgb3d.shape[1]
        # Patch grid for the 3D scene tokens, used to map per-patch quantities back
        # to image space. Layout below is "(b nc) c h w -> b (nc h w) c", so camera
        # `i` occupies tokens [i*h*w : (i+1)*h*w] in row-major (h, w) order.
        self._last_rgb3d_grid = (int(nc), int(h_feat_pcd), int(w_feat_pcd))
        rgb3d_feats = einops.rearrange(rgb3d_feats, "(b nc) c h w -> b (nc h w) c", nc=nc)
        instr_feats = self.cond_linear(self.cond_norm(instr_feats))

        rgb2d_feats = None
        if rgb2d is not None and self.use_rgb2d_context:
            n2d = rgb2d.shape[1]
            rgb2d_feats, _ = _encode_visual_grid(rgb2d, save_motion_cache=False)
            _, _, h2d, w2d = rgb2d_feats.shape
            rgb2d_feats = einops.rearrange(
                rgb2d_feats,
                "(b nc) c h w -> b nc c h w",
                nc=n2d,
            )
            if hasattr(self, "camera_ids"):
                rgb2d_feats = rgb2d_feats + self.camera_ids.weight[
                    None, :n2d, :, None, None
                ]
            if hasattr(self, "pos_embed_2d"):
                pos_2d = self.pos_embed_2d(
                    torch.arange(0, h2d * w2d, device=rgb2d_feats.device)
                ).reshape(h2d, w2d, -1)
                pos_2d = einops.rearrange(pos_2d, "h w c -> c h w")
                rgb2d_feats = rgb2d_feats + pos_2d[None, None]
            rgb2d_feats = einops.rearrange(
                rgb2d_feats, "b nc c h w -> b (nc h w) c"
            )

        # Point cloud
        num_cameras = pcd.shape[1]
        #h_ = int(math.sqrt(rgb3d_feats.shape[1] // num_cameras))
        # Interpolate point cloud to get the corresponding locations
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (h_feat_pcd, w_feat_pcd),  # NOTE: was (h_, h_), modify to accept rectangular input
            mode='bilinear'
        )
        # Merge different cameras
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        return rgb3d_feats, rgb2d_feats, pcd, instr_feats

    def run_fps(self, features, pos):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        if self.subsampling_factor == 1:
            return features, pos

        bs, npts, ch = features.shape
        sampled_inds = density_based_sampler(features, self.subsampling_factor)

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B Np F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # If positions are None, return
        if pos is None:
            return sampled_features, None

        # Else sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B Np 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    """
    Args:
        features: Tensor of shape (B, N, C)
        subsample_factor: downsampling factor, e.g., 4 keeps 25% of the points
        k: number of neighbors to compute local density (default: 8)

    Returns:
        sampled_inds: LongTensor (B, N//factor) with sampled point indices
    """
    B, N, C = features.shape
    # (B, N, N) pairwise distances in feature space
    dists = torch.cdist(features, features, p=2)  # L2 distance

    # Get average distance to k nearest neighbors (as inverse density estimate)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)  # (B, N), higher = more sparse

    # Choose top M points with highest avg distance (i.e. lowest density)
    M = N // subsample_factor
    sampled_inds = density.topk(M, dim=-1, largest=True).indices  # (B, M)

    return sampled_inds
