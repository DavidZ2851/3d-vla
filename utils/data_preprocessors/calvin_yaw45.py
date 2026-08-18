import torch

from .calvin import CALVINDataPreprocessor
from ..calvin_camera_frame import (
    get_frame_transforms,
    pose_world_to_frame,
    rel_action_world_to_frame,
)


class CALVINFrameDataPreprocessor(CALVINDataPreprocessor):
    """CALVIN preprocessor that anchors everything to an arbitrary fixed frame.

    Frame-generic twin of ``CALVINCamDataPreprocessor``; see that class for why
    proprio and the action chunk must move with the geometry (the head's 3D RoPE
    encodes relative positions between scene and action tokens, so mixing frames
    would be meaningless).

    Observation side is inherited verbatim -- the frame change lives entirely in
    the injected depth2cloud. The Euler math runs in float64 on tiny
    ``(B, T, nhand, 7)`` tensors and casts back, so the round trip stays exact.
    """

    FRAME_NAME = "world"

    @property
    def _world2frame(self):
        world2frame, _ = get_frame_transforms(self.FRAME_NAME)
        return world2frame

    def process_proprio(self, proprio):
        proprio = super().process_proprio(proprio)
        return pose_world_to_frame(
            proprio.double(), self._world2frame
        ).to(proprio.dtype)

    def process_abs_actions(self, actions):
        """Absolute world-frame action chunk -> target frame."""
        actions = super().process_actions(actions)
        return pose_world_to_frame(
            actions.double(), self._world2frame
        ).to(actions.dtype)

    def process_rel_actions(self, actions, proprio):
        """World-frame per-step deltas -> target-frame per-step deltas.

        ``proprio`` must be the RAW world-frame proprioception (as loaded from
        the zarr), not the already-converted one.

        The frame change is applied to the FULL horizon before any
        ``keypose_only`` slicing, because the conversion runs a cumsum over the
        chunk -- dropping steps first would silently change the anchor.
        """
        actions = actions.cuda(non_blocking=True)
        proprio = proprio.cuda(non_blocking=True)
        rel_frame = rel_action_world_to_frame(
            actions.double(), proprio.double(), self._world2frame
        ).to(actions.dtype)
        if self.keypose_only:
            rel_frame = rel_frame[:, [-1]]
        return rel_frame


class CALVINYaw45DataPreprocessor(CALVINFrameDataPreprocessor):
    """Proprio + actions in the robot-base yaw45 frame."""

    FRAME_NAME = "calvin_base_yaw45"
