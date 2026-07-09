import einops
import torch

from ...utils.layers import AttentionModule
from .encoder3d import Encoder as Encoder3D


class Encoder(Encoder3D):
    """2D ablation of the Florence2 3D encoder used by ``FlowerActor2DFA``.

    Keeps the exact same submodules (and therefore parameter count) as
    ``Encoder3D`` so the ablation isolates the effect of 3D geometry. The only
    differences are in the data flow:

      * the point cloud is never lifted to 3D positions — it is discarded;
      * scene tokens receive learnable camera-id + 2D sinusoidal position
        embeddings instead of per-pixel 3D coordinates;
      * proprioception attends to the scene without any 3D rotary positional
        encoding.

    No ``RotaryPositionEncoding3D`` is applied anywhere in this encoder.
    """

    def __init__(self,
                 backbone="florence2",
                 output_level="res3",
                 upsample=False,
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_camera_ids=2):
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )
        # Rebuild proprio<-scene attention without 3D rotary PE. RoPE is
        # parameter-free, so this keeps the parameter count identical to the 3D
        # encoder while removing all 3D geometry from the model.
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=False, use_adaln=False,
            pre_norm=False
        )
        # Unlike the 3D encoder (where camera-id embeddings tag the 2 auxiliary
        # 2D cameras), this 2D ablation tags the *3D scene cameras* — there is
        # no 3D geometry to disambiguate views, so each scene camera needs its
        # own learnable id. Size the embedding to the number of scene cameras
        # (CALVIN/LIBERO: 2; RoboTwin2 head/left/right: 3). The inherited
        # nn.Embedding(2, ...) only fits 2, so override it here.
        self.camera_ids = torch.nn.Embedding(num_camera_ids, embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos):
        """Compute proprioception features without 3D positional encoding.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: ignored (no 3D geometry in the 2D ablation)

        Returns:
            - proprio_feats: (B, nhist, F)
        """
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )
        # Attend to scene tokens with no positional encoding.
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats
        )[-1]
        return proprio_feats

    def encode_florence2(self, rgb3d, rgb2d, pcd, text):
        """Encode Florence2 scene/language features without lifting to 3D.

        Reuses the 3D path to run the backbone + ViT-FPN, then discards the
        lifted point cloud and instead tags scene tokens with camera-id and 2D
        sinusoidal position embeddings.

        Returns:
            - rgb3d_feats: (B, ncam*h*w, F) with 2D positional embeddings
            - rgb2d_feats: None
            - pcd:         None (no 3D geometry)
            - instr_feats: (B, L, F)
        """
        rgb3d_feats, _rgb2d_feats, _pcd, instr_feats = super().encode_florence2(
            rgb3d, rgb2d, pcd, text
        )
        # The lifted point cloud (_pcd) is intentionally dropped here: it never
        # reaches any learnable layer, so the model stays purely 2D.
        nc, h, w = self._last_rgb3d_grid
        feats = einops.rearrange(
            rgb3d_feats, "b (nc h w) c -> b nc c h w", nc=nc, h=h, w=w
        )
        # Per-scene-camera learnable id embeddings. The embedding must have at
        # least one row per scene camera (see __init__'s num_camera_ids).
        if nc > self.camera_ids.num_embeddings:
            raise ValueError(
                f"Encoder2DFA was built with num_camera_ids="
                f"{self.camera_ids.num_embeddings} but received {nc} scene "
                f"cameras. Pass --num_camera_ids {nc} for this dataset."
            )
        feats = feats + self.camera_ids.weight[None, :nc, :, None, None]
        # 2D sinusoidal position embeddings over the patch grid.
        pos_2d = self.pos_embed_2d(
            torch.arange(0, h * w, device=feats.device)
        ).reshape(h, w, -1)
        pos_2d = einops.rearrange(pos_2d, "h w c -> c h w")
        feats = feats + pos_2d[None, None]
        rgb3d_feats = einops.rearrange(feats, "b nc c h w -> b (nc h w) c")
        return rgb3d_feats, None, None, instr_feats
