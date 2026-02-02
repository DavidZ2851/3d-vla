import torch

from modeling.utils.utils import (
    quaternion_to_matrix,
    normalise_quat,
)
from modeling.utils.rotation_utils import rotation_geodesic_distance


def compute_metrics(pred, gt):
    """
    pred, gt: (B, L, nhands, D)
    """

    B, L, nhands, D = pred.shape
    assert nhands == 2, "Expect nhands=2 (left, right)"

    tr = 'traj_'

    # ------------------------
    # Position
    # ------------------------
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # shape: (B, L, nhands)

    # ------------------------
    # Rotation
    # ------------------------
    pred_quat = normalise_quat(pred[..., 3:-1])
    gt_quat   = normalise_quat(gt[..., 3:-1])
    pred_mat = quaternion_to_matrix(
        pred_quat.reshape(B * L * nhands, 4)
    )
    gt_mat = quaternion_to_matrix(
        gt_quat.reshape(B * L * nhands, 4)
    )

    rot_angle = rotation_geodesic_distance(pred_mat, gt_mat)
    rot_angle = rot_angle.reshape(B, L, nhands)


    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)

    # ------------------------
    # Gripper
    # ------------------------
    openess = ((pred[..., -1] >= 0.5) == (gt[..., -1] >= 0.5)).float()
    # shape: (B, L, nhands)

    # ------------------------
    # Aggregate
    # ------------------------

    # 1️⃣ 全局 overall，保留原 key 以兼容 trainer
    ret_global = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2.mean() < 0.02).float().mean(),  # overall for both hands
        tr + 'rot_angle': rot_angle.mean(),
        tr + 'rot_acc_angle_01': (rot_angle.mean(-1) < 0.1).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.03).float().mean(),
        tr + 'gripper': openess.flatten().float().mean(),
    }

    # 2️⃣ 每只手
    ret_per_hand = {
        # left
        tr + 'left_pos_l2': pos_l2[..., 0].mean(),
        tr + 'left_pos_acc_001': (pos_l2[..., 0] < 0.02).float().mean(),
        tr + 'left_rot_angle': rot_angle[..., 0].mean(),
        tr + 'left_rot_acc_angle_01': (rot_angle[..., 0] < 0.1).float().mean(),
        tr + 'left_gripper': openess[..., 0].mean(),

        # right
        tr + 'right_pos_l2': pos_l2[..., 1].mean(),
        tr + 'right_pos_acc_001': (pos_l2[..., 1] < 0.02).float().mean(),
        tr + 'right_rot_angle': rot_angle[..., 1].mean(),
        tr + 'right_rot_acc_angle_01': (rot_angle[..., 1] < 0.1).float().mean(),
        tr + 'right_gripper': openess[..., 1].mean(),
    }

    # 3️⃣ 轨迹级别（B, L）均值
    ret_traj = {
        tr + 'left_pos_l2': pos_l2[..., 0].mean(-1),     # (B,)
        tr + 'right_pos_l2': pos_l2[..., 1].mean(-1),
        tr + 'left_rot_angle': rot_angle[..., 0].mean(-1),
        tr + 'right_rot_angle': rot_angle[..., 1].mean(-1),
    }

    return ret_global | ret_per_hand, ret_traj