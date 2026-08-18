import numpy as np
import torch

from .calvin import CALVINDepth2Cloud
from ..calvin_camera_frame import CALVIN_WORLD2CAM


class CALVINCamDepth2Cloud(CALVINDepth2Cloud):
    """CALVIN depth -> point cloud in the **static front-camera frame**.

    Same intrinsics / uv grids / crop behavior as ``CALVINDepth2Cloud``; only the
    output frame differs:

      * front camera: deprojected with identity extrinsics, i.e. the raw camera
        coordinates the parent would then push through ``T_world_cam``;
      * wrist camera: ``T_cam_world @ T_world_wristcam``, so the wrist cloud lands
        in the same static-camera frame as the front cloud.

    CALVIN's front camera is a single fixed pose for the whole dataset (see
    ``utils/calvin_camera_frame``), so this is a constant rigid transform and
    needs no extra per-frame inputs.
    """

    def __init__(self, front_shape=(160, 160)):
        super().__init__(front_shape=front_shape)
        # Front camera is the frame we anchor to -> identity extrinsics.
        self.T_cam_cam = torch.eye(4).float().cuda(non_blocking=True)
        # world -> static cam, for re-expressing the per-frame wrist extrinsics.
        self.T_cam_world = torch.from_numpy(
            np.ascontiguousarray(CALVIN_WORLD2CAM)
        ).float().cuda(non_blocking=True)

    def __call__(self, depth_front, depth_wrist, extrinsics_wrist):
        pcd_front = None
        if depth_front is not None:
            pcd_front = self.deproject(
                depth_front,
                self.T_cam_cam[None].repeat(len(depth_front), 1, 1),
                'front'
            )
        pcd_wrist = None
        if depth_wrist is not None:
            # extrinsics_wrist is wristcam -> world; compose into wristcam -> cam.
            ext = torch.matmul(
                self.T_cam_world[None].to(extrinsics_wrist.dtype),
                extrinsics_wrist
            )
            pcd_wrist = self.deproject(depth_wrist, ext, 'wrist')
        return pcd_front, pcd_wrist
