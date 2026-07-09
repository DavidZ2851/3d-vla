import numpy as np
import torch


class RoboTwinDepth2Cloud:
    """Unproject RoboTwin2 depth maps using fixed 224 intrinsics.

    RoboTwin2 camera intrinsics are calibrated for 224x224. The training
    preprocessor restores 80x80 wrist depth to the front-camera grid before
    deprojection; this class also rescales intrinsics when called on any other
    resolution.
    """
    BASE_INTRINSICS_224 = np.array(
        [
            [251.04947, 0.0, 112.0],
            [0.0, 251.04947, 112.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, source_hw=(224, 224)):
        self.source_hw = tuple(source_hw)
        self._uv_cache = {}
        self._intrinsics_cache = {}

    @staticmethod
    def _create_uniform_pixel_coords_image(resolution):
        h, w = resolution
        u = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        v = np.tile(np.arange(h, dtype=np.float32), (w, 1)).T
        ones = np.ones_like(u, dtype=np.float32)
        return np.stack((u, v, ones), axis=0)  # (3, H, W)

    @staticmethod
    def _rescale_intrinsics(intrinsic, source_hw, target_hw):
        src_h, src_w = source_hw
        tgt_h, tgt_w = target_hw
        scale_x = float(tgt_w) / float(src_w)
        scale_y = float(tgt_h) / float(src_h)
        out = np.asarray(intrinsic, dtype=np.float32).copy()
        out[0, 0] *= scale_x
        out[1, 1] *= scale_y
        out[0, 2] *= scale_x
        out[1, 2] *= scale_y
        return out

    def _get_uniforms(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._uv_cache:
            self._uv_cache[key] = torch.from_numpy(
                self._create_uniform_pixel_coords_image((h, w))
            ).to(device=device)
        return self._uv_cache[key]

    def _get_intrinsics(self, h, w, device, dtype):
        key = (h, w, str(device), str(dtype))
        if key not in self._intrinsics_cache:
            scaled = self._rescale_intrinsics(
                self.BASE_INTRINSICS_224,
                source_hw=self.source_hw,
                target_hw=(h, w),
            )
            self._intrinsics_cache[key] = torch.from_numpy(scaled).to(
                device=device, dtype=dtype
            )
        return self._intrinsics_cache[key]

    def unproject(self, depth, extrinsics_w2c, intrinsics):
        """
        Args:
            depth: (B, H, W), positive depth in meters.
            extrinsics_w2c: (B, 4, 4), RoboTwin2 OpenCV world->camera extrinsics.
            intrinsics: (B, 3, 3), OpenCV intrinsics for the depth resolution.
        Returns:
            (B, 3, H, W) world-frame point cloud.
        """
        depth = depth.float()
        extrinsics_w2c = extrinsics_w2c.float()
        intrinsics = intrinsics.float()

        b, h, w = depth.shape
        uv1 = self._get_uniforms(h, w, depth.device).unsqueeze(0).expand(b, -1, -1, -1)
        uvz = (uv1 * depth[:, None]).reshape(b, 3, -1)

        cam_xyz = torch.bmm(torch.linalg.inv(intrinsics), uvz)
        cam_xyz_h = torch.cat([cam_xyz, torch.ones_like(cam_xyz[:, :1])], dim=1)
        world_from_cam = torch.linalg.inv(extrinsics_w2c)
        world_xyz = torch.bmm(world_from_cam[:, :3], cam_xyz_h)
        return world_xyz.reshape(b, 3, h, w)

    def deproject(self, depth, extrinsics_w2c):
        if depth is None:
            return None
        b, h, w = depth.shape
        intrinsics = self._get_intrinsics(h, w, depth.device, torch.float32)
        intrinsics = intrinsics.unsqueeze(0).expand(b, -1, -1)
        return self.unproject(depth, extrinsics_w2c, intrinsics)

    def __call__(
        self,
        depth_front,
        depth_left,
        depth_right,
        extrinsics_front,
        extrinsics_left,
        extrinsics_right,
    ):
        pcd_front = self.deproject(depth_front, extrinsics_front)
        pcd_left = self.deproject(depth_left, extrinsics_left)
        pcd_right = self.deproject(depth_right, extrinsics_right)
        return pcd_front, pcd_left, pcd_right
