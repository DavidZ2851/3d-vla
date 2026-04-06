import einops
from contextlib import nullcontext
import torch
from torch.nn import functional as F

from .encoder3d_qwen import Qwen3VLEncoder3D
from .multiscale_stagewise import QwenStagewiseMultiScaleFusion


class Qwen3VLEncoder3DStagewise(Qwen3VLEncoder3D):
    """
    Parallel Qwen encoder that keeps existing behavior and additionally exposes
    stagewise visual memories for decoder-depth routing.
    """

    def __init__(
        self,
        *args,
        use_multiscale=True,
        multiscale_stage_count=3,
        **kwargs,
    ):
        # Stagewise routing requires multiscale hooks from Qwen visual blocks.
        super().__init__(*args, use_multiscale=use_multiscale, **kwargs)

        self.multiscale_stage_count = int(multiscale_stage_count)
        self.multiscale_fuser = QwenStagewiseMultiScaleFusion(
            self.backbone.vision_hidden_size,
            self.cond_linear.out_features,
            self.backbone.multiscale_layers,
            num_stages=self.multiscale_stage_count,
        )

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
        rgb3d_normed = self.normalize(rgb3d)
        visual_dtype = self._backbone_visual_dtype()
        if rgb3d_normed.dtype != visual_dtype:
            rgb3d_normed = rgb3d_normed.to(dtype=visual_dtype)

        needs_grad = (
            self.finetune_backbone
            or self.finetune_text_encoder
            or self.num_finetune_vision_layers > 0
            or self._use_vision_lora
            or self._use_llm_lora
        )
        grad_ctx = nullcontext() if needs_grad else torch.no_grad()
        with grad_ctx:
            pre_merge, text_hidden, multiscale_pre_merge = self.backbone(
                rgb3d_normed,
                instruction,
                text_attention_mask=instr_mask,
                text_valid_lens=instr_lens,
            )

        batch_size, ncam, h16, w16, _ = pre_merge.shape
        if multiscale_pre_merge is None:
            raise RuntimeError(
                "Expected multiscale_pre_merge from Qwen backbone, got None."
            )

        rgb3d_feats, stage_rgb3d_feats = self.multiscale_fuser(multiscale_pre_merge)

        # Language path is unchanged from the current Qwen encoder.
        instr_feats = self.cond_linear(self.cond_norm(text_hidden))

        num_cameras = pcd.shape[1]
        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (h16, w16),
            mode="bilinear",
        )
        pcd_flat = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=num_cameras
        )

        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd_flat)
        fps_feats, fps_pos = self.run_fps(rgb3d_feats, pcd_flat)

        return (
            rgb3d_feats,
            pcd_flat,
            None,
            None,
            instr_feats,
            instr_pos,
            proprio_feats,
            fps_feats,
            fps_pos,
            stage_rgb3d_feats,
        )
