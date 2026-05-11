import numpy as np
import torch


class RoboTwinDepth2Cloud:
    """Unproject RoboTwin depth maps using OpenCV intrinsics/extrinsics.

    Online evaluation receives head/left/right camera images at 224x224, so
    the simulator-provided per-camera intrinsics are used directly.
    """

    def __init__(self):
        self._uv_cache = {}

    @staticmethod
    def _create_uniform_pixel_coords_image(resolution):
        h, w = resolution
        u = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        v = np.tile(np.arange(h, dtype=np.float32), (w, 1)).T
        ones = np.ones_like(u, dtype=np.float32)
        return np.stack((u, v, ones), axis=0)  # (3, H, W)

    def _get_uniforms(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._uv_cache:
            self._uv_cache[key] = torch.from_numpy(
                self._create_uniform_pixel_coords_image((h, w))
            ).to(device=device)
        return self._uv_cache[key]

    def unproject(self, depth, extrinsics, intrinsics):
        """
        Args:
            depth: (B, H, W), positive depth in meters.
            extrinsics: (B, 4, 4), world->camera OpenCV extrinsics.
            intrinsics: (B, 3, 3), OpenCV intrinsics.

        Returns:
            (B, 3, H, W) world-frame point cloud.
        """
        depth = depth.float()
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()

        b, h, w = depth.shape
        uv1 = self._get_uniforms(h, w, depth.device).unsqueeze(0).expand(b, -1, -1, -1)
        uvz = (uv1 * depth[:, None]).reshape(b, 3, -1)  # (B, 3, HW)

        cam_xyz = torch.bmm(torch.linalg.inv(intrinsics), uvz)  # (B, 3, HW)
        cam_xyz_h = torch.cat([cam_xyz, torch.ones_like(cam_xyz[:, :1])], dim=1)
        world_from_cam = torch.linalg.inv(extrinsics)
        world_xyz = torch.bmm(world_from_cam[:, :3], cam_xyz_h)
        return world_xyz.reshape(b, 3, h, w)

    def __call__(self, depth, extrinsics, intrinsics):
        b, nc, h, w = depth.shape
        world_xyz = self.unproject(
            depth.reshape(-1, h, w),
            extrinsics.reshape(-1, 4, 4),
            intrinsics.reshape(-1, 3, 3),
        )
        return world_xyz.reshape(b, nc, 3, h, w)
