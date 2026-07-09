from kornia import augmentation as K
import torch
from torch.nn import functional as F

from .base import DataPreprocessor


class RoboTwinDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=224, custom_imsize=None, depth2cloud=None):
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

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist=1, nhand=2, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        # the first proprio is the current state
        proprio = proprio[:, -1*max(self.num_history, 1):]
        return proprio

    def process_joint_proprio(self, joint_proprio):
        """Joint proprio shape: (B, history, nhand=2, n_joint+1).

        RoboTwin2 stores joint proprio with history offsets [-16, 0], so the
        CURRENT state is the LAST index (matching ``process_proprio``). The base
        class takes the first ``num_history`` frames, which would pick the oldest
        (t-16) frame -- wrong for RoboTwin2 -- so override to take the most recent.
        """
        joint_proprio = joint_proprio.cuda(non_blocking=True).float()
        nhist_ = joint_proprio.size(1)
        assert nhist_ >= self.num_history, "not enough joint proprio timesteps"
        joint_proprio = joint_proprio[:, -1 * max(self.num_history, 1):]
        return joint_proprio
    
    def process_obs(self, rgb_front, rgb_left, rgb_right,
                    depth_front, depth_left, depth_right, 
                    extrinsics_front, extrinsics_left, extrinsics_right,
                    augment=False):
        """
        RGBs of shape (B, ncam=1, 3, h_i, w_i),
        depths of shape (B, ncam=1, h_i, w_i).
        """
        front_hw = (rgb_front.size(-2), rgb_front.size(-1))

        rgb_front = rgb_front.cuda(non_blocking=True)
        rgb_left = rgb_left.cuda(non_blocking=True)
        rgb_right = rgb_right.cuda(non_blocking=True)
        depth_front = depth_front.cuda(non_blocking=True).float()
        depth_left = depth_left.cuda(non_blocking=True).float()
        depth_right = depth_right.cuda(non_blocking=True).float()
        extrinsics_front = extrinsics_front.cuda(non_blocking=True).float()
        extrinsics_left = extrinsics_left.cuda(non_blocking=True).float()
        extrinsics_right = extrinsics_right.cuda(non_blocking=True).float()

        # Merged RoboTwin2 stores wrist RGB/depth at 80x80. Restore them to the
        # front-camera grid before deprojection so fixed 224 intrinsics produce
        # a pcd aligned with the resized wrist RGB/depth tensors.
        if tuple(rgb_left.shape[-2:]) != front_hw:
            rgb_left = F.interpolate(
                rgb_left[:, 0].float(), front_hw, mode='bilinear', antialias=False
            )[:, None]
        if tuple(rgb_right.shape[-2:]) != front_hw:
            rgb_right = F.interpolate(
                rgb_right[:, 0].float(), front_hw, mode='bilinear', antialias=False
            )[:, None]
        if tuple(depth_left.shape[-2:]) != front_hw:
            depth_left = F.interpolate(depth_left, front_hw, mode='nearest')
        if tuple(depth_right.shape[-2:]) != front_hw:
            depth_right = F.interpolate(depth_right, front_hw, mode='nearest')

        # Get point cloud from depth after any wrist restoration.
        pcd_front, pcd_left, pcd_right = self.depth2cloud(
            depth_front[:, 0],  # one camera
            depth_left[:, 0],
            depth_right[:, 0],
            extrinsics_front,
            extrinsics_left,
            extrinsics_right,
        )
        pcd_front = pcd_front[:, None]
        pcd_left = pcd_left[:, None]
        pcd_right = pcd_right[:, None]

        # Handle front camera, which may require augmentations
        if augment:
            b, nc, _, h, w = rgb_front.shape
            # Augment in half precision
            obs = torch.cat((
                rgb_front.half() / 255,
                pcd_front.half()
            ), 2)  # (B, ncam, 6, H, W)
            obs = obs.reshape(-1, 6, h, w)
            obs = self.aug(obs)
            # Convert to full precision
            rgb_front = obs[:, :3].reshape(b, nc, 3, h, w).float()
            pcd_front = obs[:, 3:].reshape(b, nc, 3, h, w).float()
        else:
            # Simply convert to full precision
            rgb_front = rgb_front.float() / 255
        if self.custom_imsize is not None and self.custom_imsize != rgb_front.size(-1):
            rgb_front = F.interpolate(
                rgb_front[:, 0], (self.custom_imsize, self.custom_imsize),
                mode='bilinear', antialias=True
            )[:, None]
            pcd_front = F.interpolate(
                pcd_front[:, 0], (self.custom_imsize, self.custom_imsize),
                mode='nearest'
            )[:, None]

        # Handle wrist cameras. At this point RGB/depth/pcd are already on the
        # front-camera grid; only custom_imsize may require another resize.
        target_hw = (rgb_front.size(-2), rgb_front.size(-1))
        rgb_left = rgb_left.float() / 255
        if tuple(rgb_left.shape[-2:]) != target_hw:
            rgb_left = F.interpolate(
                rgb_left[:, 0], target_hw,
                mode='bilinear', antialias=True
            )[:, None]
        pcd_left = pcd_left.float()
        if tuple(pcd_left.shape[-2:]) != target_hw:
            pcd_left = F.interpolate(
                pcd_left[:, 0], target_hw,
                mode='nearest'
            )[:, None]
        rgb_right = rgb_right.float() / 255
        if tuple(rgb_right.shape[-2:]) != target_hw:
            rgb_right = F.interpolate(
                rgb_right[:, 0], target_hw,
                mode='bilinear', antialias=True
            )[:, None]
        pcd_right = pcd_right.float()
        if tuple(pcd_right.shape[-2:]) != target_hw:
            pcd_right = F.interpolate(
                pcd_right[:, 0], target_hw,
                mode='nearest'
            )[:, None]

        # Concatenate
        rgbs = torch.cat((rgb_front, rgb_left, rgb_right), 1)
        pcds = torch.cat((pcd_front, pcd_left, pcd_right), 1)
        return rgbs, pcds
