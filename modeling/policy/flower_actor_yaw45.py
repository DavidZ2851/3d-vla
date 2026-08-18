"""Policy anchored to a second fixed frame, for the "any fixed frame" ablation.

The frame is CALVIN's robot-base position with a synthetic 45 deg yaw about
world z (see ``utils/calvin_camera_frame``). It complements the static-camera
variant by probing a *different kind* of re-anchoring:

  * camera frame: 59 deg tilt of the z axis, gravity alignment destroyed,
    motion energy concentrated onto the depth axis;
  * this frame:   z axis untouched (0 deg tilt, still gravity aligned), only
    x/y remixed.

If both match the world-frame baseline, the claim covers both a
gravity-preserving and a gravity-breaking transform, which is much stronger
than either alone.

Everything else -- architecture, parameter count, the frame<->world hop around
the motion expert -- is inherited unchanged from the camera variant.
"""

from .flower_actor_cam import FlowerActor3DMotionLatentCam


class FlowerActor3DMotionLatentYaw45(FlowerActor3DMotionLatentCam):
    """``flower_actor_3d_motion_latent`` anchored to the robot-base yaw45 frame.

    Only ``FRAME_NAME`` changes; the parent resolves every transform from it.
    ``motion_expert_frame='camera'`` here means "the expert was trained in THIS
    frame" -- the flag name is inherited and frame-agnostic in effect.
    """

    FRAME_NAME = "calvin_base_yaw45"
