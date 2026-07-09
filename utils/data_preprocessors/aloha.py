"""
ALOHA preprocessor: front + back kinects as 3D cameras (depth -> world PCD via
fixed calibration), with an optional wrist 2D RGB camera. RGB/depth are stored at
224x398 (non-square; preserves source 720:1280 aspect).
"""

from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class AlohaDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=(224, 398), custom_imsize=None, depth2cloud=None):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud,
        )
        if isinstance(orig_imsize, int):
            orig_h, orig_w = orig_imsize, orig_imsize
        else:
            orig_h, orig_w = orig_imsize
        self.orig_hw = (orig_h, orig_w)
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0.0,
                translate=(10 / orig_w, 10 / orig_h),
                padding_mode="border",
                p=1.0,
            )
        ).cuda()

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand=1, 10)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        proprio = proprio[:, -1 * max(self.num_history, 1):]
        return proprio

    def _maybe_resize(self, rgb3d, rgb2d, pcd):
        if self.custom_imsize is None:
            return rgb3d, rgb2d, pcd
        target_hw = (self.custom_imsize, self.custom_imsize) \
            if isinstance(self.custom_imsize, int) else tuple(self.custom_imsize)
        if (rgb3d.size(-2), rgb3d.size(-1)) == target_hw:
            return rgb3d, rgb2d, pcd

        def _interp(x, mode):  # (B, nc, 3, h, w) -> (B, nc, 3, *target_hw)
            b, nc = x.shape[:2]
            x = F.interpolate(x.reshape(b * nc, 3, *x.shape[-2:]), target_hw,
                              mode=mode, antialias=(mode == "bilinear"))
            return x.reshape(b, nc, 3, *target_hw)

        rgb2d = None if rgb2d is None else _interp(rgb2d, "bilinear")
        return _interp(rgb3d, "bilinear"), rgb2d, _interp(pcd, "nearest")

    def process_obs(self, rgb_front, rgb_back, rgb_wrist,
                    depth_front, depth_back, augment=False,
                    wrist_as_scene=False):
        """
        rgb_front/back       : (B, 1, 3, H, W) uint8
        rgb_wrist            : optional (B, 1, 3, H, W) uint8
        depth_front/back     : (B, 1, H, W) float32, meters
        wrist_as_scene       : if True, route the wrist as a THIRD scene (rgb3d)
                               camera instead of a 2D-context camera. Used by the
                               2D visual-language model (flower_actor_2d_joint),
                               whose 2DFA encoder tags rgb3d scene cameras with
                               camera-id embeddings and DISCARDS both rgb2d and the
                               point cloud. The wrist has no depth, so its pcd slot
                               is a zeros placeholder (never reaches a learnable
                               layer). num_camera_ids must then be 3.
        Returns:
            rgb3d : (B, ncam3d, 3, H', W') float32 in [0, 1]
                    (front, back) or (front, back, wrist) when wrist_as_scene
            rgb2d : None, or (B, 1, 3, H', W') float32 in [0, 1] (wrist)
            pcd   : (B, ncam3d, 3, H', W') float32 world frame; the wrist slot is
                    zeros when wrist_as_scene
        """
        df = depth_front.cuda(non_blocking=True).float()[:, 0]  # (B, H, W)
        db = depth_back.cuda(non_blocking=True).float()[:, 0]
        pcd_front = self.depth2cloud.deproject_cam(df, "front")[:, None]  # (B,1,3,H,W)
        pcd_back = self.depth2cloud.deproject_cam(db, "back")[:, None]

        use_wrist_scene = wrist_as_scene and rgb_wrist is not None
        if use_wrist_scene:
            # Wrist is a scene camera: append its RGB and a zeros pcd placeholder
            # (the 2DFA encoder discards pcd, so the wrist needs no real geometry).
            pcd = torch.cat([pcd_front, pcd_back, torch.zeros_like(pcd_front)], dim=1)
            rgb3d = torch.cat([rgb_front, rgb_back, rgb_wrist], dim=1)  # (B,3,3,H,W)
            rgb2d = None
        else:
            pcd = torch.cat([pcd_front, pcd_back], dim=1)            # (B,2,3,H,W)
            rgb3d = torch.cat([rgb_front, rgb_back], dim=1)          # (B,2,3,H,W) uint8
            rgb2d = rgb_wrist                                        # optional (B,1,3,H,W) uint8

        if augment:
            # 3D cameras: augment rgb + world-pcd jointly (geometric consistency),
            # one independent affine per (sample, camera) — matches the original front path.
            b, nc, _, h, w = rgb3d.shape
            obs = torch.cat(
                (rgb3d.cuda(non_blocking=True).half() / 255, pcd.half()), 2
            ).reshape(-1, 6, h, w)
            obs = self.aug(obs)
            rgb3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd = obs[:, 3:].reshape(b, nc, 3, h, w).float()
            # 2D wrist: RGB only.
            if rgb2d is not None:
                b2, nc2, _, h2, w2 = rgb2d.shape
                rgb2d = self.aug(
                    rgb2d.cuda(non_blocking=True).half().reshape(-1, 3, h2, w2) / 255
                ).reshape(b2, nc2, 3, h2, w2).float()
        else:
            rgb3d = rgb3d.cuda(non_blocking=True).float() / 255
            if rgb2d is not None:
                rgb2d = rgb2d.cuda(non_blocking=True).float() / 255

        return self._maybe_resize(rgb3d, rgb2d, pcd)
