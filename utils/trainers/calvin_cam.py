"""CALVIN trainer that anchors the whole policy to the fixed static-camera frame.

Additive counterpart to ``CALVINTrainTester``: same data, same models, same
losses -- the only change is the coordinate frame that scene geometry,
proprioception and the EEF action chunk live in. See
``utils/calvin_camera_frame`` for the transform and why it is a single
constant (CALVIN's front camera never moves).

Nothing in the world-frame path is touched; ``--dataset CalvinCam*`` routes
here through the ``calvincam`` branches added to the three registries.
"""

import numpy as np
import torch
from torch import nn

from .calvin import CALVINTrainTester
from ..calvin_camera_frame import (
    pose_cam_to_world,
    rel_action_cam_to_abs_world,
    position_delta_bound_cam,
)


# Camera-frame counterpart of CALVINTrainTester.get_workspace_normalizer's
# +/-[0.02] * 3 + [0.05] * 3.
#
# Positions: CALVIN clips its world-frame delta to |d|_inf <= 0.02, so the exact
# camera-frame per-axis max is 0.02 * sum_j |R_cam_world[i, j]| -- analytic, not
# a sample estimate (see position_delta_bound_cam).
#
# Rotations: Euler deltas do NOT rotate rigidly. Camera-frame TCP orientations
# sit closer to the e_y = +/-pi/2 Euler singularity (min |cos(e_y)| ~ 0.39 in
# camera frame vs ~0.80 in world), which amplifies the angular rates roughly
# 3x. Measured over 2M steps of the train split by
# data_processing/compute_calvin_cam_rel_action_stats.py (max |delta| =
# [0.154, 0.070, 0.174]) and carried with a 1.05 margin; the full report lives in
# datasets/calvin_cam_rel_action_stats.json. Regenerate both if the data changes.
CAM_FRAME_ROTATION_HI = (0.162, 0.074, 0.184)


class CALVINCamTrainTester(CALVINTrainTester):
    """CALVIN train/test in the static-camera frame."""

    @torch.no_grad()
    def get_workspace_normalizer(self):
        hi = np.concatenate([
            position_delta_bound_cam(),
            np.asarray(CAM_FRAME_ROTATION_HI, dtype=np.float64),
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
        # camera-frame prediction against a world-frame target.
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


class CamEvalWrapper:
    """Online-eval wrapper for a camera-frame policy.

    Mirrors ``utils.trainers.calvin.EvalWrapper`` but converts the observation
    into the camera frame on the way in and the predicted trajectory back into
    the world frame on the way out, so the CALVIN env still receives the
    absolute world-frame TCP targets it expects.
    """

    def __init__(self, policy, preprocessor, tokenizer,
                 pre_tokenize=True, relative_action=True, pred_len=10):
        self.policy = policy
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.pre_tokenize = pre_tokenize
        self.relative_action = relative_action
        self.pred_len = pred_len

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

        # Camera-frame deltas -> absolute WORLD poses for the env.
        proprio_cam = obs['proprio'][:, [-1], 0].double()
        if bool(self.relative_action):
            trajectory = rel_action_cam_to_abs_world(trajectory, proprio_cam)
        else:
            trajectory = pose_cam_to_world(trajectory)
        return trajectory.float().cpu().numpy()
