"""Fixed world <-> static-camera frame transform for CALVIN.

CALVIN's front ("static") camera sits at ONE pose for the entire dataset. It is
declared in the env config (``cameras.static.look_from / look_at / up_vector``),
those values are byte-identical between ``merged_config_train_abc_d.yaml`` and
``merged_config_val_abc_d.yaml``, and scenes A/B/C/D vary the table and objects
but never the camera. That is why ``data_processing/calvin_to_zarr.py`` stores
only ``extrinsics_wrist`` -- there is no per-frame front extrinsic to store --
and why ``utils/depth2cloud/calvin.py`` can hardcode a single ``T_world_cam``.

So "anchor everything to the fixed camera frame" is a single constant rigid
transform, applied to every geometric quantity the policy sees:

  * the front point cloud (deprojected with identity extrinsics instead of
    ``T_world_cam``),
  * the wrist point cloud (``T_cam_world @ T_world_wristcam``),
  * proprioception and the absolute/relative EEF action chunk.

Conventions
-----------
``CALVIN_VIEW_MATRIX_FLAT`` is pybullet's ``computeViewMatrix`` output (OpenGL:
camera looks down -z, +y up in the image). Reshaped ``(4, 4).T`` it IS the
world->cam matrix; ``utils/depth2cloud/calvin.py`` inverts it to get cam->world.

Poses are the 7-vector this repo uses everywhere for CALVIN:
``[x, y, z, e_x, e_y, e_z, gripper]`` where the Euler triple follows pybullet's
``getEulerFromQuaternion`` convention -- **extrinsic** xyz, i.e.
``R = Rz(e_z) @ Ry(e_y) @ Rx(e_x)`` (scipy's lowercase ``'xyz'``). The gripper
channel is a scalar openness and is never touched by any transform here.

Relative actions
----------------
The zarr's ``rel_action`` is CALVIN's own per-step ``rel_actions`` (a clipped
world-frame delta), and both eval paths reconstruct absolutes as
``abs = proprio + cumsum(rel)``. The camera-frame deltas produced here are
defined so that reconstruction stays exact under the same rule:

    abs_w[k] = prop_w + cumsum(rel_w)[k]          (the world reconstruction)
    abs_c[k] = W2C(abs_w[k]),  prop_c = W2C(prop_w)
    rel_c[k] = abs_c[k] - abs_c[k-1],  abs_c[-1] := prop_c   (Euler wrapped)

Running ``abs_c = prop_c + cumsum(rel_c)`` then ``C2W(abs_c)`` recovers
``abs_w`` exactly, so the camera-frame policy is a faithful reparameterization
of what the world-frame policy actually executes. Euler components are only
recovered modulo 2*pi, which ``C2W`` (euler -> matrix -> euler) normalizes away.
"""

import numpy as np
import torch


# Copied verbatim from utils/depth2cloud/calvin.py (which owns the world-frame
# path and must keep working unchanged).
CALVIN_VIEW_MATRIX_FLAT = (
    0.7232445478439331, -0.030260592699050903, 0.6899287700653076, 0.0,
    0.5141233801841736, 0.6906105875968933, -0.5086592435836792, 0.0,
    -0.46107974648475647, 0.7225935459136963, 0.5150379538536072, 0.0,
    0.21526622772216797, -0.26317155361175537, -4.399168968200684, 1.0,
)

# world -> cam (the pybullet view matrix) and its inverse.
CALVIN_WORLD2CAM = np.asarray(
    CALVIN_VIEW_MATRIX_FLAT, dtype=np.float64
).reshape(4, 4).T
CALVIN_CAM2WORLD = np.linalg.inv(CALVIN_WORLD2CAM)


# ----------------------------------------------------------------------------
# A second fixed frame, for the "any fixed frame works" ablation.
# ----------------------------------------------------------------------------
# CALVIN's robot base pose, identical across scenes A/B/C/D
# (conf/scene/calvin_scene_*.yaml: robot_base_position / robot_base_orientation).
CALVIN_ROBOT_BASE_POSITION = np.array([-0.34, -0.46, 0.24], dtype=np.float64)

# NOTE: the robot's true base ORIENTATION is [0, 0, 0] -- the base is aligned
# with world. A frame that differs from world by pure translation is EXACTLY
# world for anything trained on per-sample-centered geometry and delta targets
# (verified numerically to 3e-15), so the real robot-base frame carries zero
# information as an ablation arm. The yaw below is therefore deliberately
# synthetic: it is NOT a physical quantity of the scene, only a fixed rotation
# chosen so the frame is genuinely different from world while staying z-up.
# 45 deg happens to point +x from the base at the scene centroid (measured
# 45.5 deg), and avoids the degenerate 90 deg case, which is an exact x<->y
# axis swap and therefore close to the decoder's own `z-trans` serialization.
CALVIN_BASE_YAW_DEG = 45.0


def _yaw_frame_world2frame(yaw_deg, origin):
    """World -> a z-up frame yawed ``yaw_deg`` about world z, centered at ``origin``."""
    c = np.cos(np.radians(yaw_deg))
    s = np.sin(np.radians(yaw_deg))
    # Columns are the new frame's axes expressed in world.
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ np.asarray(origin, dtype=np.float64)
    return out


CALVIN_WORLD2BASEYAW45 = _yaw_frame_world2frame(
    CALVIN_BASE_YAW_DEG, CALVIN_ROBOT_BASE_POSITION
)
CALVIN_BASEYAW45_2WORLD = np.linalg.inv(CALVIN_WORLD2BASEYAW45)


#: Frame name -> (world->frame, frame->world). ``world`` maps to ``None`` so
#: callers can cheaply detect the no-op case.
CALVIN_FRAMES = {
    "world": (None, None),
    "calvin_static": (CALVIN_WORLD2CAM, CALVIN_CAM2WORLD),
    "calvin_base_yaw45": (CALVIN_WORLD2BASEYAW45, CALVIN_BASEYAW45_2WORLD),
}


def get_frame_transforms(frame_name):
    """Resolve a frame name to ``(world2frame, frame2world)``.

    ``None``/``''``/``'world'`` yield ``(None, None)``.
    """
    key = "world" if frame_name is None else str(frame_name).strip().lower()
    if key in ("", "none"):
        key = "world"
    if key not in CALVIN_FRAMES:
        raise ValueError(
            f"Unknown CALVIN frame {frame_name!r}; expected one of "
            f"{sorted(CALVIN_FRAMES)}."
        )
    return CALVIN_FRAMES[key]


# ----------------------------------------------------------------------------
# Euler <-> rotation matrix (pybullet extrinsic-xyz convention), batched torch.
# ----------------------------------------------------------------------------
def euler_to_matrix(euler):
    """(..., 3) extrinsic-xyz Euler angles -> (..., 3, 3) rotation matrices."""
    a, b, c = euler[..., 0], euler[..., 1], euler[..., 2]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    # R = Rz(c) @ Ry(b) @ Rx(a)
    row0 = torch.stack([cb * cc, cc * sa * sb - ca * sc, ca * cc * sb + sa * sc], -1)
    row1 = torch.stack([cb * sc, ca * cc + sa * sb * sc, -cc * sa + ca * sb * sc], -1)
    row2 = torch.stack([-sb, cb * sa, ca * cb], -1)
    return torch.stack([row0, row1, row2], -2)


def matrix_to_euler(matrix):
    """(..., 3, 3) rotation matrices -> (..., 3) extrinsic-xyz Euler angles.

    Falls back to the gimbal-lock branch (``e_z := 0``) where ``cos(e_y)`` is
    numerically zero, which is the standard degenerate-case convention.
    """
    r00, r10, r20 = matrix[..., 0, 0], matrix[..., 1, 0], matrix[..., 2, 0]
    r21, r22 = matrix[..., 2, 1], matrix[..., 2, 2]
    r11, r12 = matrix[..., 1, 1], matrix[..., 1, 2]
    cy = torch.sqrt(r00 * r00 + r10 * r10)
    singular = cy < 1e-6
    e_y = torch.atan2(-r20, cy)
    e_x = torch.where(singular, torch.atan2(-r12, r11), torch.atan2(r21, r22))
    e_z = torch.where(singular, torch.zeros_like(r10), torch.atan2(r10, r00))
    return torch.stack([e_x, e_y, e_z], -1)


def wrap_to_pi(angles):
    """Wrap angles to (-pi, pi]."""
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


# ----------------------------------------------------------------------------
# Pose transforms
# ----------------------------------------------------------------------------
def _rigid(transform, device, dtype):
    t = torch.as_tensor(transform, device=device, dtype=dtype)
    return t[:3, :3], t[:3, 3]


def _transform_pose(pose, transform):
    """Apply a 4x4 rigid transform to ``(..., >=6)`` poses, gripper untouched."""
    rot, trans = _rigid(transform, pose.device, pose.dtype)
    xyz = pose[..., :3] @ rot.transpose(0, 1) + trans
    euler = matrix_to_euler(rot @ euler_to_matrix(pose[..., 3:6]))
    return torch.cat([xyz, euler, pose[..., 6:]], -1)


def pose_world_to_frame(pose, world2frame):
    """World-frame ``(..., 7)`` poses -> an arbitrary fixed frame."""
    if world2frame is None:
        return pose
    return _transform_pose(pose, world2frame)


def pose_frame_to_world(pose, frame2world):
    """Fixed-frame ``(..., 7)`` poses -> world frame."""
    if frame2world is None:
        return pose
    return _transform_pose(pose, frame2world)


def pose_world_to_cam(pose):
    """World-frame ``(..., 7)`` poses -> static-camera frame."""
    return _transform_pose(pose, CALVIN_WORLD2CAM)


def pose_cam_to_world(pose):
    """Static-camera-frame ``(..., 7)`` poses -> world frame."""
    return _transform_pose(pose, CALVIN_CAM2WORLD)


# ----------------------------------------------------------------------------
# Relative-action transforms
# ----------------------------------------------------------------------------
def rel_action_world_to_frame(rel_action, proprio, world2frame, time_dim=1):
    """World per-step deltas -> fixed-frame per-step deltas.

    Args:
        rel_action: ``(B, T, nhand, 7)`` world-frame deltas (the zarr's
            ``rel_action``). Channel 6 is absolute gripper openness and is
            passed through unchanged.
        proprio: ``(B, nhist, nhand, 7)`` world-frame state; the LAST history
            step is the anchor, matching ``_to_abs`` at eval.
        world2frame: 4x4 world -> target frame, or ``None`` for a no-op.
        time_dim: axis of ``rel_action`` holding the prediction horizon.

    Returns:
        ``(B, T, nhand, 7)`` target-frame deltas.
    """
    if world2frame is None:
        return rel_action
    anchor_w = proprio[:, -1:].to(rel_action.dtype)  # (B, 1, nhand, 7)
    # World absolute trajectory exactly as eval reconstructs it.
    abs_w = torch.cat([
        anchor_w[..., :6] + torch.cumsum(rel_action[..., :6], dim=time_dim),
        rel_action[..., 6:],
    ], -1)
    abs_c = pose_world_to_frame(abs_w, world2frame)
    anchor_c = pose_world_to_frame(anchor_w, world2frame)
    prev_c = torch.cat(
        [anchor_c, abs_c.narrow(time_dim, 0, abs_c.size(time_dim) - 1)],
        dim=time_dim,
    )
    return torch.cat([
        abs_c[..., :3] - prev_c[..., :3],
        wrap_to_pi(abs_c[..., 3:6] - prev_c[..., 3:6]),
        rel_action[..., 6:],
    ], -1)


def rel_action_world_to_cam(rel_action, proprio, time_dim=1):
    """World per-step deltas -> static-camera-frame per-step deltas."""
    return rel_action_world_to_frame(
        rel_action, proprio, CALVIN_WORLD2CAM, time_dim=time_dim
    )


def rel_action_frame_to_abs_world(rel_action_f, proprio_f, frame2world, time_dim=1):
    """Fixed-frame per-step deltas + fixed-frame anchor -> absolute world poses.

    The eval-time inverse of :func:`rel_action_world_to_frame`.
    """
    abs_c = torch.cat([
        proprio_f[..., :6] + torch.cumsum(rel_action_f[..., :6], dim=time_dim),
        rel_action_f[..., 6:],
    ], -1)
    return pose_frame_to_world(abs_c, frame2world)


def rel_action_cam_to_abs_world(rel_action_cam, proprio_cam, time_dim=1):
    """Camera-frame per-step deltas + camera-frame anchor -> absolute world poses.

    The eval-time inverse of :func:`rel_action_world_to_cam`.
    """
    return rel_action_frame_to_abs_world(
        rel_action_cam, proprio_cam, CALVIN_CAM2WORLD, time_dim=time_dim
    )


# ----------------------------------------------------------------------------
# Numpy conveniences for the online-eval path (small tensors, CPU).
# ----------------------------------------------------------------------------
def pose_world_to_frame_np(pose, world2frame):
    if world2frame is None:
        return np.asarray(pose)
    return pose_world_to_frame(
        torch.from_numpy(np.asarray(pose)).double(), world2frame
    ).numpy()


def pose_frame_to_world_np(pose, frame2world):
    if frame2world is None:
        return np.asarray(pose)
    return pose_frame_to_world(
        torch.from_numpy(np.asarray(pose)).double(), frame2world
    ).numpy()


def pose_world_to_cam_np(pose):
    return pose_world_to_cam(torch.from_numpy(np.asarray(pose)).double()).numpy()


def pose_cam_to_world_np(pose):
    return pose_cam_to_world(torch.from_numpy(np.asarray(pose)).double()).numpy()


def position_delta_bound_frame(world2frame, world_bound=0.02):
    """Exact per-axis bound on fixed-frame position deltas.

    CALVIN clips its world-frame position delta to ``|d|_inf <= world_bound``
    (``rel_actions[:3] * 0.02`` in ``calvin_to_zarr.py``), which is why the
    world-frame workspace normalizer is +/-0.02. Rotating that cube into the
    target frame gives a per-axis maximum of ``bound * sum_j |R[i, j]|``.
    """
    if world2frame is None:
        return np.full(3, float(world_bound))
    return world_bound * np.abs(world2frame[:3, :3]).sum(axis=1)


def position_delta_bound_cam(world_bound=0.02):
    """Exact per-axis bound on camera-frame position deltas."""
    return position_delta_bound_frame(CALVIN_WORLD2CAM, world_bound=world_bound)
