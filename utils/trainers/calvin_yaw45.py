"""CALVIN trainer anchored to the robot-base yaw45 frame.

Second fixed frame for the "any fixed frame works" ablation, additive in the
same way ``calvin_cam.py`` is. Routed by ``--dataset CalvinYaw45*`` through the
``calvinyaw45`` branches in the three registries.

Why this frame complements the camera one
-----------------------------------------
The camera frame tilts the z axis 59 deg (gravity alignment destroyed) and
pushes TCP orientations toward the Euler ``e_y = +/-pi/2`` singularity, which
amplifies angular rates ~3.7x. This frame is a pure yaw about world z: the z
axis is untouched, gravity alignment is preserved, and only x/y are remixed.
Together they cover a gravity-breaking and a gravity-preserving re-anchoring.
"""

import numpy as np
import torch
from torch import nn

from .calvin import CALVINTrainTester
from ..calvin_camera_frame import (
    CALVIN_WORLD2BASEYAW45,
    get_frame_transforms,
    pose_frame_to_world,
    position_delta_bound_frame,
    rel_action_frame_to_abs_world,
)


# Counterpart of CALVINTrainTester.get_workspace_normalizer's
# +/-[0.02] * 3 + [0.05] * 3.
#
# Positions: CALVIN clips its world-frame delta to |d|_inf <= 0.02, so the exact
# per-axis max is 0.02 * sum_j |R[i, j]| -- analytic, not a sample estimate.
# For a 45 deg yaw that is [0.0283, 0.0283, 0.02] (z untouched).
#
# Rotations: UNCHANGED from world, and this one is analytic too. The Euler
# convention here is extrinsic xyz (R = Rz(e_z) Ry(e_y) Rx(e_x)), so a pure yaw
# about world z left-multiplies as Rz(45) Rz(e_z) Ry(e_y) Rx(e_x)
# = Rz(e_z + 45) Ry(e_y) Rx(e_x) -- e_x and e_y are untouched and e_z picks up a
# constant, which cancels in a per-step difference. Confirmed on real data:
# measured max over 2M steps is exactly [0.05, 0.05, 0.05] (see
# datasets/calvin_yaw45_rel_action_stats.json, regenerate with
# data_processing/compute_calvin_cam_rel_action_stats.py --frame
# calvin_base_yaw45). No margin is applied because the bound is exact rather
# than a sample estimate.
#
# (The branch-flip hazard that makes this fail for arbitrary rotations needs
# |e_y| > pi/2; CALVIN's TCP stays within ~37 deg, so it never triggers here.)
YAW45_FRAME_ROTATION_HI = (0.05, 0.05, 0.05)


class CALVINYaw45TrainTester(CALVINTrainTester):
    """CALVIN train/test in the robot-base yaw45 frame."""

    FRAME_NAME = "calvin_base_yaw45"

    @torch.no_grad()
    def get_workspace_normalizer(self):
        hi = np.concatenate([
            position_delta_bound_frame(CALVIN_WORLD2BASEYAW45),
            np.asarray(YAW45_FRAME_ROTATION_HI, dtype=np.float64),
        ])
        hi = torch.tensor(hi, dtype=torch.float32)
        return nn.Parameter(torch.stack([-hi, hi]), requires_grad=False)

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # Convert the action chunk BEFORE proprio is overwritten: the relative
        # conversion needs the raw world-frame anchor.
        raw_proprio = sample["proprioception"]
        if bool(self.args.relative_action):
            action = self.preprocessor.process_rel_actions(
                sample["action"], raw_proprio
            )
        else:
            action = self.preprocessor.process_abs_actions(sample["action"])
        # Write back like CALVINTrainTester.prepare_batch does:
        # BaseTrainTester.evaluate_nsteps reads its ground truth straight from
        # sample["action"], so without this the val metrics would score a
        # yaw45-frame prediction against a world-frame target.
        sample["action"] = action
        proprio = self.preprocessor.process_proprio(raw_proprio)
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["wrist_depth"], sample["extrinsics_wrist"],
            augment=augment
        )
        return (
            action,
            torch.zeros(action.shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )


class Yaw45EvalWrapper:
    """Online-eval wrapper for a yaw45-frame policy.

    Mirrors ``utils.trainers.calvin_cam.CamEvalWrapper``: converts the
    observation into the yaw45 frame on the way in and the predicted trajectory
    back into world on the way out, so the CALVIN env still receives absolute
    world-frame TCP targets.
    """

    FRAME_NAME = "calvin_base_yaw45"

    def __init__(self, policy, preprocessor, tokenizer,
                 pre_tokenize=True, relative_action=True, pred_len=10):
        self.policy = policy
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.pre_tokenize = pre_tokenize
        self.relative_action = relative_action
        self.pred_len = pred_len
        _, self.frame2world = get_frame_transforms(self.FRAME_NAME)

    def reset(self):
        pass

    @torch.no_grad()
    def preprocess(self, obs, env):
        proprio_world = torch.from_numpy(np.concatenate([
            obs['robot_obs'][:6],
            (obs['robot_obs'][[-1]] + 1) / 2
        ], -1))[None, None, None].float()
        proprio = self.preprocessor.process_proprio(proprio_world)
        rgbs, pcds = self.preprocessor.process_obs(
            torch.from_numpy(obs["rgb_obs"]["rgb_static"].transpose(2, 0, 1)[None, None]),
            torch.from_numpy(obs["rgb_obs"]["rgb_gripper"].transpose(2, 0, 1)[None, None]),
            torch.from_numpy(obs["depth_obs"]["depth_static"][None, None]),
            torch.from_numpy(obs["depth_obs"]["depth_gripper"][None, None]),
            torch.from_numpy(np.linalg.inv(
                np.array(env.cameras[1].view_matrix).reshape((4, 4)).T
            )[None]),
            augment=False
        )
        return {'rgb': rgbs, 'pcd': pcds, 'proprio': proprio}

    @torch.no_grad()
    def step(self, obs, instruction):
        trajectory_mask = torch.full([self.pred_len], False).cuda(non_blocking=True)
        trajectory_mask = trajectory_mask[None, :, None]

        instr_mask = None
        if not self.pre_tokenize:
            instr = [instruction]
        else:
            tok_out = self.tokenizer([instruction])
            if isinstance(tok_out, torch.Tensor):
                instr = tok_out.cuda(non_blocking=True)
            else:
                instr = tok_out.input_ids.cuda(non_blocking=True)
                instr_mask = tok_out.attention_mask.cuda(non_blocking=True)

        trajectory = self.policy(
            None,
            trajectory_mask,
            obs['rgb'],
            None,
            obs['pcd'],
            instr,
            obs['proprio'],
            run_inference=True,
            instr_mask=instr_mask,
        )
        trajectory = trajectory.view(1, self.pred_len, 7).double()
        trajectory[..., -1] = 2 * (trajectory[..., -1] >= 0.5).double() - 1

        # Frame deltas -> absolute WORLD poses for the env.
        proprio_frame = obs['proprio'][:, [-1], 0].double()
        if bool(self.relative_action):
            trajectory = rel_action_frame_to_abs_world(
                trajectory, proprio_frame, self.frame2world
            )
        else:
            trajectory = pose_frame_to_world(trajectory, self.frame2world)
        return trajectory.float().cpu().numpy()
