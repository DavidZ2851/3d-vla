from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class RLBenchDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        self.aug = K.AugmentationSequential(
            # K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=0.8
            ),
            # K.RandomRotation((-5, 5), p=0.3),
            K.RandomResizedCrop(
                size=(orig_imsize, orig_imsize),
                scale=(0.95, 1.05),
                p=0.1
            )
        ).cuda()

    def process_obs(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16)
        )

        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            
            # Augment in half precision
            obs = torch.cat((
                rgbs.cuda(non_blocking=True).half() / 255,
                pcds[:, :rgbs.size(1)].half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)

            current_aug = K.AugmentationSequential(
                K.RandomAffine(
                    degrees=0,
                    translate=0.0,
                    scale=(0.75, 1.25),
                    padding_mode="reflection",
                    p=0.8
                ),
                K.RandomResizedCrop(
                    size=(h, w),
                    scale=(0.95, 1.05),
                    p=0.1
                )
            ).cuda()

            obs_in = obs
            success = False
            for _ in range(3):
                try:
                    obs = current_aug(obs_in) 
                    success = True
                    break
                except RuntimeError as e:
                    if "singular" in str(e) or "solve" in str(e):
                        continue
                    else:
                        raise e
            if not success:
                obs = obs_in

            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()
            
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds[:, :rgb_3d.size(1)].float()
        
        self.custom_imsize = None
        if self.custom_imsize is not None and self.custom_imsize != rgb_3d.size(-1):
            b, nc, _, _, _ = rgb_3d.shape
            rgb_3d = F.interpolate(
                rgb_3d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Handle wrist cameras, no augmentations
        rgb_2d = None
        if rgb2d is not None:
            rgb_2d = rgb2d.cuda(non_blocking=True).float() / 255
            if self.custom_imsize is not None and self.custom_imsize != rgb_2d.size(-1):
                b, nc, _, _, _ = rgb_2d.shape
                rgb_2d = F.interpolate(
                    rgb_2d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                    mode='bilinear', antialias=True
                ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Concatenate RGBs
        if rgb_2d is not None:
            if rgb_3d.shape[-2:] != rgb_2d.shape[-2:]:
                 b_2d, nc_2d, _, _, _ = rgb_2d.shape
                 rgb_2d = F.interpolate(
                    rgb_2d.flatten(0, 1), 
                    size=(rgb_3d.shape[-2], rgb_3d.shape[-1]),
                    mode='bilinear', antialias=True
                ).reshape(b_2d, nc_2d, -1, rgb_3d.shape[-2], rgb_3d.shape[-1])
            rgbs = torch.cat((rgb_3d, rgb_2d), 1)
        else:
            rgbs = rgb_3d
            
        # Concatenate PCDs
        if pcd_3d.size(1) < pcds.size(1):
            wrist_pcds = pcds[:, pcd_3d.size(1):].float()
            if wrist_pcds.shape[-1] != pcd_3d.shape[-1]:
                 wrist_pcds = F.interpolate(
                    wrist_pcds.flatten(0, 1),
                    size=(pcd_3d.shape[-2], pcd_3d.shape[-1]),
                    mode='nearest'
                 ).reshape(b, -1, 3, pcd_3d.shape[-2], pcd_3d.shape[-1])
            pcds = torch.cat((pcd_3d, wrist_pcds), dim=1)
        else:
            pcds = pcd_3d
        return rgbs, pcds
    
    def process_obs_original(self, rgbs, rgb2d, depth, extrinsics, intrinsics,
                    augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Get point cloud from depth
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).to(torch.bfloat16),
            extrinsics.cuda(non_blocking=True).to(torch.bfloat16),
            intrinsics.cuda(non_blocking=True).to(torch.bfloat16)
        )

        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            # Augment in half precision
            obs = torch.cat((
                rgbs.cuda(non_blocking=True).half() / 255,
                pcds[:, :rgbs.size(1)].half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)

            obs_in = obs
            success = False
            for _ in range(3):
                try:
                    obs = self.aug(obs_in)
                    success = True
                    break
                except RuntimeError as e:
                    if "singular" in str(e) or "solve" in str(e):
                        continue
                    else:
                        raise e
            if not success:
                obs = obs_in

            # obs = self.aug(obs)  # NOTE: modify for some random failure
            # Convert to full precision
            rgb_3d = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_3d = obs[:, 3:].reshape(b, nc, 3, h, w).float()

            
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds[:, :rgb_3d.size(1)].float()
        if self.custom_imsize is not None and self.custom_imsize != rgb_3d.size(-1):
            b, nc, _, _, _ = rgb_3d.shape
            rgb_3d = F.interpolate(
                rgb_3d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

        # Handle wrist cameras, no augmentations
        rgb_2d = None
        if rgb2d is not None:
            rgb_2d = rgb2d.cuda(non_blocking=True).float() / 255
            if self.custom_imsize is not None and self.custom_imsize != rgb_2d.size(-1):
                b, nc, _, _, _ = rgb_2d.shape
                rgb_2d = F.interpolate(
                    rgb_2d.flatten(0, 1), (self.custom_imsize, self.custom_imsize),
                    mode='bilinear', antialias=True
                ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)

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
