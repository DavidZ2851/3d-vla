import numpy as np
import torch

from .calvin import CALVINDepth2Cloud
from ..calvin_camera_frame import get_frame_transforms


class CALVINFrameDepth2Cloud(CALVINDepth2Cloud):
    """CALVIN depth -> point cloud in an arbitrary fixed frame.

    Same intrinsics / uv grids / crop behavior as ``CALVINDepth2Cloud``; only the
    output frame differs. Both cameras are deprojected to world exactly as the
    parent does, then composed with one constant ``T_frame_world``:

      * front camera: ``T_frame_world @ T_world_cam``
      * wrist camera: ``T_frame_world @ T_world_wristcam``

    (The static-camera variant in ``calvin_cam.py`` is the special case where
    ``T_frame_world == T_cam_world``, which collapses the front transform to
    identity. It keeps its own implementation so that path is untouched.)
    """

    FRAME_NAME = "world"

    def __init__(self, front_shape=(160, 160)):
        super().__init__(front_shape=front_shape)
        world2frame, _ = get_frame_transforms(self.FRAME_NAME)
        if world2frame is None:
            raise ValueError(
                f"{type(self).__name__} needs a non-world FRAME_NAME; use "
                "CALVINDepth2Cloud for the world frame."
            )
        self.T_frame_world = torch.from_numpy(
            np.ascontiguousarray(world2frame)
        ).float().cuda(non_blocking=True)
        # front camera -> target frame, constant for the whole dataset.
        self.T_frame_cam = torch.matmul(self.T_frame_world, self.T_world_cam)

    def __call__(self, depth_front, depth_wrist, extrinsics_wrist):
        pcd_front = None
        if depth_front is not None:
            pcd_front = self.deproject(
                depth_front,
                self.T_frame_cam[None].repeat(len(depth_front), 1, 1),
                'front'
            )
        pcd_wrist = None
        if depth_wrist is not None:
            # extrinsics_wrist is wristcam -> world; compose into wristcam -> frame.
            ext = torch.matmul(
                self.T_frame_world[None].to(extrinsics_wrist.dtype),
                extrinsics_wrist
            )
            pcd_wrist = self.deproject(depth_wrist, ext, 'wrist')
        return pcd_front, pcd_wrist


class CALVINYaw45Depth2Cloud(CALVINFrameDepth2Cloud):
    """Point clouds in the robot-base yaw45 frame."""

    FRAME_NAME = "calvin_base_yaw45"
