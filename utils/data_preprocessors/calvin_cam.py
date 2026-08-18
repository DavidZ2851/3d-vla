import torch

from .calvin import CALVINDataPreprocessor
from ..calvin_camera_frame import (
    pose_world_to_cam,
    rel_action_world_to_cam,
)


class CALVINCamDataPreprocessor(CALVINDataPreprocessor):
    """CALVIN preprocessor that anchors everything to the static camera frame.

    Observation side: ``process_obs`` is inherited verbatim -- the frame change
    lives entirely in the injected ``CALVINCamDepth2Cloud``, which emits both
    point clouds in the static-camera frame.

    Action side: proprioception and the action chunk are mapped world -> camera
    here, so that scene geometry, proprio, and the decoder's trajectory position
    queries all live in one frame (the 3D RoPE in the head encodes *relative*
    positions between scene and action tokens, so mixing frames would be
    meaningless).

    Absolute poses go through the rigid transform directly. Relative actions
    need the anchor as well, so :meth:`process_rel_actions` takes both -- see
    ``utils/calvin_camera_frame`` for why that keeps ``abs = proprio +
    cumsum(rel)`` exactly invertible.

    The Euler math runs in float64 (the tensors are tiny: ``(B, T, nhand, 7)``)
    and is cast back to the caller's dtype, so the round trip stays exact.
    """

    def process_proprio(self, proprio):
        proprio = super().process_proprio(proprio)
        return pose_world_to_cam(proprio.double()).to(proprio.dtype)

    def process_abs_actions(self, actions):
        """Absolute world-frame action chunk -> camera frame."""
        actions = super().process_actions(actions)
        return pose_world_to_cam(actions.double()).to(actions.dtype)

    def process_rel_actions(self, actions, proprio):
        """World-frame per-step deltas -> camera-frame per-step deltas.

        ``proprio`` must be the RAW world-frame proprioception (as loaded from
        the zarr), not the already-converted camera-frame one.

        The frame change is applied to the FULL horizon before any
        ``keypose_only`` slicing, because the conversion runs a cumsum over the
        chunk -- dropping steps first would silently change the anchor.
        """
        actions = actions.cuda(non_blocking=True)
        proprio = proprio.cuda(non_blocking=True)
        rel_cam = rel_action_world_to_cam(
            actions.double(), proprio.double()
        ).to(actions.dtype)
        if self.keypose_only:
            rel_cam = rel_cam[:, [-1]]
        return rel_cam
