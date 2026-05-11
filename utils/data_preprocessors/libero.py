from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class LiberoDataPreprocessor(DataPreprocessor):
    def __init__(self, keypose_only=False, num_history=3,
                 orig_imsize=224, custom_imsize=None, depth2cloud=None):
        if custom_imsize not in (None, orig_imsize):
            raise ValueError(
                f"LIBERO zarr stores RGB/depth/intrinsics at {orig_imsize}x{orig_imsize}. "
                f"Use custom_img_size={orig_imsize} or None, got {custom_imsize}."
            )
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0.0,
                translate=(10 / orig_imsize, 10 / orig_imsize),
                padding_mode="border",
                p=1.0,
            )
        ).cuda()

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        proprio = proprio[:, -1*max(self.num_history, 1):]
        return proprio

    def process_obs(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        extrinsics of shape (B, ncam, 4, 4).
        intrinsics of shape (B, ncam, 3, 3).
        """
        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).float(),
            extrinsics.cuda(non_blocking=True).float(),
            intrinsics.cuda(non_blocking=True).float()
        )

        # Agent (cam 0): optional augmentation. Wrist (cam 1): no augmentation.
        if augment:
            b, nc, _, h, w = rgbs.shape
            rgbs_cuda = rgbs.cuda(non_blocking=True)
            obs0 = torch.cat((
                rgbs_cuda[:, 0:1].half() / 255,
                pcds[:, 0:1].half()
            ), 2).reshape(-1, 6, h, w)
            obs0 = self.aug(obs0)
            rgb0 = obs0[:, :3].reshape(b, 1, 3, h, w).float()
            pcd0 = obs0[:, 3:].reshape(b, 1, 3, h, w).float()
            if nc > 1:
                rgb_rest = rgbs_cuda[:, 1:].float() / 255
                pcd_rest = pcds[:, 1:].float()
                rgb_3d = torch.cat((rgb0, rgb_rest), dim=1)
                pcd_3d = torch.cat((pcd0, pcd_rest), dim=1)
            else:
                rgb_3d = rgb0
                pcd_3d = pcd0
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds[:, :rgb_3d.size(1)].float()

        # Handle 2d cameras, default is None
        rgb_2d = None
        if rgb2d is not None:
            rgb_2d = rgb2d.cuda(non_blocking=True).float() / 255

        # Concatenate
        if rgb_2d is not None:
            rgbs = torch.cat((rgb_3d, rgb_2d), 1)
        else:
            rgbs = rgb_3d
        if pcd_3d.size(1) < pcds.size(1):
            pcds = torch.cat((pcd_3d, pcds[:, :pcd_3d.size(1)].float()))
        else:
            pcds = pcd_3d
        return rgbs, pcds
