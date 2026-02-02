from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class CALVINDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=200, custom_imsize=None, depth2cloud=None):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0.0,
                translate=(10/orig_imsize, 10/orig_imsize),
                padding_mode='border',
                p=1.0
            )
        ).cuda()

    def process_obs(self, rgb_front, rgb_wrist,
                    depth_front, depth_wrist, extrinsics_wrist,
                    augment=False):
        """
        RGBs of shape (B, ncam=1, 3, h_i, w_i),
        depths of shape (B, ncam=1, h_i, w_i).
        """
        # Get point cloud from depth
        pcd_front, pcd_wrist = self.depth2cloud(
            depth_front.cuda(non_blocking=True).float()[:, 0],  # one camera
            depth_wrist.cuda(non_blocking=True).float()[:, 0],
            extrinsics_wrist.cuda(non_blocking=True).float()
        )
        pcd_front = pcd_front[:, None]
        pcd_wrist = pcd_wrist[:, None]

        # Handle front camera, which may require augmentations
        if augment:
            b, nc, _, h, w = rgb_front.shape
            # Augment in half precision
            obs = torch.cat((
                rgb_front.cuda(non_blocking=True).half() / 255,
                pcd_front.half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_front = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_front = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision
            rgb_front = rgb_front.cuda(non_blocking=True).float() / 255
        if self.custom_imsize is not None and self.custom_imsize != rgb_front.size(-1):
            rgb_front = F.interpolate(
                rgb_front[:, 0], (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            )[:, None]

        # Handle wrist camera, no augmentations
        rgb_wrist = rgb_wrist.cuda(non_blocking=True).float() / 255
        rgb_wrist = F.interpolate(
            rgb_wrist[:, 0], (rgb_front.size(-1), rgb_front.size(-1)),
            mode='bilinear', antialias=True
        )[:, None]
        pcd_wrist = pcd_wrist.float()
        pcd_wrist = F.interpolate(
            pcd_wrist[:, 0], (pcd_front.size(-2), pcd_front.size(-1)),
            mode='nearest'
        )[:, None]
        # yes, rgbs and pcds can have different h, w

        # Concatenate
        rgbs = torch.cat((rgb_front, rgb_wrist), 1)
        pcds = torch.cat((pcd_front, pcd_wrist), 1)
        return rgbs, pcds
