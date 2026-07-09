import torch
from torch import nn
from torch.nn import functional as F
import einops

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix,
    euler_to_6d,
    rotate6d_to_euler
)


class DenoiseActor(nn.Module):

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        self._nhand = nhand  # NOTE: only depends on args.bimanual, True->2, False->1
        self._continuous_openness = False
        self._normalized_action_clamp = None
        self._signed_binary_openness_output = False
        # Binary default keeps existing dataset behavior: positive raw gripper
        # labels train the positive class. RobotWin2 overrides this to learn
        # closed logits from raw openness labels.
        self._binary_openness_target_mode = "positive"
        self._binary_openness_target_threshold = 0.5
        self._binary_openness_inference_threshold = 0.5
        # Default behavior preserves the existing relative-action semantics used
        # by other datasets. Dataset-specific trainers can narrow this.
        self._relative_position_mode = "cumulative"
        self._relative_rotation = relative

        # Vision-language encoder, runs only once
        self.encoder = None  # Implement this!

        # Action decoder, runs at every denoising timestep
        # euler/axisangle: xyz(3)+rot(3) = 6 passed through as-is
        # quat_*/rotmatrix: xyz(3)+rot6d(6) = 9 after convert_rot
        self.traj_encoder = nn.Linear(
            6 if rotation_format in ('euler', 'axisangle') else 9,
            embedding_dim
        )
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * 2,  # NOTE: build model with always 2 hands (was nhist=nhist * nhand)
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ('euler', 'axisangle') else 6
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process.
        # euler/axisangle: normalize pos+rot (6D), set by trainer.
        # quat_*/rotmatrix: normalize pos only (3D), rotations are not normalized.
        if rotation_format in ('euler', 'axisangle'):
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0., 0.], [1., 1., 1.]]),
                requires_grad=False
            )
        self.nrm_dim = int(self.workspace_normalizer.size(-1))
        self.openness_normalizer = nn.Parameter(
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            requires_grad=False
        )

    def encode_inputs(
        self,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        encoder_input_proprio = proprio
        # NOTE: fix for flower_actor_2d — convert external rotation rep to rot6d for encoder
        if hasattr(self.encoder, "proprio_feat"):
            layer = self.encoder.proprio_feat
            if isinstance(layer, nn.Linear) and layer.in_features > 8:
                if proprio.shape[-1] == 8:  # quat proprio: xyz(3)+quat(4)+gripper(1)
                    pos = proprio[..., :3]
                    quat = proprio[..., 3:7]
                    gripper = proprio[..., 7:]
                    if self._rotation_format == 'quat_xyzw':
                        quat = quat[..., [3, 0, 1, 2]]
                    mat = quaternion_to_matrix(quat)
                    batch_dims = mat.shape[:-2]
                    mat_flat = mat.flatten(0, -3)
                    rot6d_flat = get_ortho6d_from_rotation_matrix(mat_flat)  # (N, 6)
                    rot6d = rot6d_flat.view(*batch_dims, 6)
                    encoder_input_proprio = torch.cat([pos, rot6d, gripper], dim=-1)
                elif proprio.shape[-1] == 13:  # rotmatrix proprio: xyz(3)+rotmat(9)+gripper(1)
                    pos = proprio[..., :3]
                    mat = proprio[..., 3:12]
                    gripper = proprio[..., 12:]
                    batch_dims = mat.shape[:-1]
                    mat_flat = mat.flatten(0, -2).reshape(-1, 3, 3)
                    rot6d_flat = get_ortho6d_from_rotation_matrix(mat_flat)  # (N, 6)
                    rot6d = rot6d_flat.view(*batch_dims, 6)
                    encoder_input_proprio = torch.cat([pos, rot6d, gripper], dim=-1)
        # NOTE: fix end
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            encoder_input_proprio.flatten(1, 2),
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs

    def _absolute_query_xyz(self, query_trajectory, traj_xyz):
        """Map (possibly relative) decoder query positions to absolute world xyz.

        Honors ``self._relative_position_mode``:
          - "anchor"     : each step is relative to the current anchor pose.
          - "cumulative" : steps are incremental deltas to be cumulatively summed.
        Subclasses that override ``policy_forward_pass`` MUST call this instead of
        hardcoding a mode, so dataset-specific relative xyz semantics stay correct.
        """
        if not self._relative:
            return traj_xyz
        if self._relative_position_mode == "anchor":
            return query_trajectory[..., :3] + traj_xyz
        if self._relative_position_mode == "cumulative":
            return query_trajectory[..., :3] + torch.cumsum(traj_xyz, dim=1)
        raise ValueError(
            f"Unknown relative position mode: {self._relative_position_mode!r}"
        )

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, valid_mask=None):
        # Parse inputs
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = fixed_inputs
        # Get features from normalized (relative) trajectory
        trajectory_feats = self.traj_encoder(trajectory)

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        # relative to absolute for decoder position queries
        traj_xyz = self._absolute_query_xyz(query_trajectory, traj_xyz)

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos,
            valid_mask=valid_mask
        )

    def conditional_sample(self, trajectory, device, fixed_inputs, valid_mask=None):
        # Set schedulers
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(device).long(),
                fixed_inputs, valid_mask=valid_mask
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_scheduler.step(
                out[..., :3],
                t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1],
                t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)
        return torch.cat((trajectory, out[..., -1:]), -1)

    def _make_binary_openness_target(self, gt_openness):
        mode = getattr(self, "_binary_openness_target_mode", "positive")
        threshold = float(getattr(self, "_binary_openness_target_threshold", 0.5))
        if mode == "positive":
            return (gt_openness > 0).to(gt_openness.dtype)
        if mode == "closed_from_open":
            return (gt_openness < threshold).to(gt_openness.dtype)
        if mode == "soft_closed_from_open":
            return (1.0 - gt_openness).clamp(0.0, 1.0).to(gt_openness.dtype)
        if mode == "open_from_raw":
            return (gt_openness >= threshold).to(gt_openness.dtype)
        raise ValueError(f"Unknown binary openness target mode: {mode!r}")

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        valid_mask = (~trajectory_mask)[..., None].float()  # (B,T,H,1)
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )

        # Sample from learned model starting from noise
        # euler/axisangle: xyz(3)+rot(3)=6;  quat_*/rotmatrix: xyz(3)+rot6d(6)=9
        out_dim = 6 if self._rotation_format in ('euler', 'axisangle') else 9
        trajectory = torch.randn(
            size=tuple(trajectory_mask.shape) + (out_dim,),
            device=trajectory_mask.device
        )
        trajectory = self.conditional_sample(
            trajectory,
            device=trajectory_mask.device,
            fixed_inputs=fixed_inputs,
            # valid_mask=valid_mask  # NOTE NOTE: for action space alignment
        )

        # Back to quaternion and euler
        _, traj_len, nhand, _ = trajectory.shape
        trajectory = self.unconvert_rot(
            trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)
        if self._continuous_openness:
            trajectory[..., -1:] = self.unnormalize_openness(
                torch.tanh(trajectory[..., -1:])
            )
        else:
            prob_close = trajectory[..., -1].sigmoid()
            if self._signed_binary_openness_output:
                trajectory[..., -1] = torch.where(
                    prob_close >= 0.5,
                    torch.ones_like(prob_close),
                    -torch.ones_like(prob_close),
                )
            else:
                trajectory[..., -1] = prob_close
        # trajectory = trajectory * valid_mask  # NOTE NOTE: ensure padded hand output zeros

        return trajectory

    def compute_loss(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )
        return self.flow_loss_given_fixed(
            gt_trajectory, trajectory_mask, fixed_inputs
        )

    def flow_loss_given_fixed(self, gt_trajectory, trajectory_mask, fixed_inputs,
                              include_openness=True):
        """EEF flow-matching loss given already-encoded ``fixed_inputs``. Split
        out of ``compute_loss`` so subclasses (e.g. the joint-action model) can
        reuse the EEF supervision without re-running the (expensive) encoder.

        ``include_openness`` defaults to True (unchanged behavior for every
        existing model). The joint-action model passes False so the EEF expert is
        supervised on position+rotation only, leaving the gripper entirely to the
        joint head."""
        # Process gt_trajectory
        gt_openess = gt_trajectory[..., -1:]
        if self._continuous_openness:
            gt_openess = self.normalize_openness(gt_openess)
        else:
            gt_openess = self._make_binary_openness_target(gt_openess)
        gt_trajectory = gt_trajectory[..., :-1]
        # Normalize all pos
        gt_trajectory = self.normalize_pos(gt_trajectory)
        # Convert rotation parametrization (quat/euler -> 6d)
        _, traj_len, nhand, _ = gt_trajectory.shape
        gt_trajectory = self.convert_rot(
            gt_trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        valid_mask = (~trajectory_mask)[..., None].float()  # (B, T, 2, 1)

        # Loop lv2_batch_size times and sample different noises with same input
        # Trick to effectively increase the batch size without re-encoding
        total_loss = 0
        for _ in range(self._lv2_batch_size):
            # Sample noise
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

            # Sample a random timestep
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            # Add noise to the clean trajectories
            pos = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3],
                timesteps
            )
            rot = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:],
                timesteps
            )
            noisy_trajectory = torch.cat((pos, rot), -1)

            # Predict the noise residual
            pred = self.policy_forward_pass(
                noisy_trajectory,
                timesteps, fixed_inputs,
                # valid_mask=valid_mask  # NOTE NOTE: for action space alignment
            )

            # Compute loss
            for layer_pred in pred:
                pos = layer_pred[..., :3]
                rot = layer_pred[..., 3:-1]
                openess = layer_pred[..., -1:]
                denoise_target = self.position_scheduler.prepare_target(
                    noise, gt_trajectory
                )
                pos_loss = F.l1_loss(pos, denoise_target[..., :3], reduction='mean')
                rot_loss = F.l1_loss(rot, denoise_target[..., 3:], reduction='mean')
                if not include_openness:
                    openness_loss = pos_loss.new_zeros(())
                elif self._continuous_openness:
                    openness_loss = F.smooth_l1_loss(
                        torch.tanh(openess),
                        gt_openess,
                        reduction='mean'
                    )
                else:
                    openness_loss = F.binary_cross_entropy_with_logits(openess, gt_openess)
                
                # NOTE NOTE: for action space alignment
                # valid_mask_bool = valid_mask.bool()
                # valid_count = valid_mask.sum().clamp(min=1)  # avoid div by zero if all masked
                # pos_loss = F.l1_loss(pos, denoise_target[..., :3], reduction='none')
                # pos_loss = torch.where(valid_mask_bool, pos_loss, torch.zeros_like(pos_loss))
                # pos_loss = pos_loss.sum() / valid_count  # NOTE: ensure padded hand doesn't include in loss
                # rot_loss = F.l1_loss(rot, denoise_target[..., 3:], reduction='none')
                # rot_loss = torch.where(valid_mask_bool, rot_loss, torch.zeros_like(rot_loss))
                # rot_loss = rot_loss.sum() / valid_count
                # openness_loss = F.binary_cross_entropy_with_logits(openess, gt_openess, reduction='none')
                # openness_loss = torch.where(valid_mask_bool, openness_loss, torch.zeros_like(openness_loss))
                # openness_loss = openness_loss.sum() / valid_count
                
                loss = (
                    30 * pos_loss
                    + 10 * rot_loss
                    + openness_loss
                )
                total_loss = total_loss + loss
        return total_loss / self._lv2_batch_size

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] - _min) / diff * 2.0
            - 1.0
        )
        if self._normalized_action_clamp is not None:
            clamp = float(self._normalized_action_clamp)
            out[..., :self.nrm_dim] = out[..., :self.nrm_dim].clamp(-clamp, clamp)
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    def normalize_openness(self, signal):
        _min = self.openness_normalizer[0]
        _max = self.openness_normalizer[1]
        diff = (_max - _min).clamp_min(1e-6)
        out = (signal - _min) / diff * 2.0 - 1.0
        if self._normalized_action_clamp is not None:
            clamp = float(self._normalized_action_clamp)
            out = out.clamp(-clamp, clamp)
        return out

    def unnormalize_openness(self, signal):
        _min = self.openness_normalizer[0]
        _max = self.openness_normalizer[1]
        diff = (_max - _min).clamp_min(1e-6)
        return (signal + 1.0) / 2.0 * diff + _min

    @staticmethod
    def _identity_rot6d_like(rot6d):
        identity = rot6d.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        return identity.view(*([1] * (rot6d.ndim - 1)), 6)

    def convert_rot(self, signal):
        """Convert external rotation representation to rot6d for diffusion.
        euler/axisangle : already 3, no conversion needed.
        rot6d           : input is already xyz(3) + rot6d(6) [+ res], no conversion.
                          For relative-rotation mode we subtract the identity
                          rot6d so the rest pose maps to zeros.
        quat_*          : quat(4) → rotmat(3x3) → rot6d(6).
        rotmatrix       : rotmat flat(9) → rot6d(6).
        """
        if self._rotation_format in ('euler', 'axisangle'):
            return signal
        if self._rotation_format == 'rot6d':
            # signal: (..., 3 + 6 [+ res])
            rot6d = signal[..., 3:9]
            if self._relative_rotation:
                rot6d = rot6d - self._identity_rot6d_like(rot6d)
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            out = torch.cat([signal[..., :3], rot6d], dim=-1)
            if res is not None:
                out = torch.cat((out, res), -1)
            return out
            # NOTE: if use euler->6d
            # rot = signal[..., 3:6]
            # res = signal[..., 6:] if signal.size(-1) > 6 else None
            # rot6d = euler_to_6d(rot)
            # signal = torch.cat([signal[..., :3], rot6d], dim=-1)
            # if res is not None:
            #     signal = torch.cat((signal, res), -1)
            # return signal
        # Else assume quaternion
        if self._rotation_format == 'rotmatrix':
            # signal: (..., 3 + 9 + 1)  — flattened row-major 3×3 matrix
            res = signal[..., 12:] if signal.size(-1) > 12 else None
            rot_flat = signal[..., 3:12]
            *bdims, _ = rot_flat.shape
            mat = rot_flat.reshape(-1, 3, 3)
            rot6d = get_ortho6d_from_rotation_matrix(mat).reshape(*bdims, 6)
            signal = torch.cat([signal[..., :3], rot6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            return signal
        # quat_xyzw / quat_wxyz
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        rot = quaternion_to_matrix(rot)
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        if self._relative_rotation:
            rot = rot - self._identity_rot6d_like(rot)
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        """Convert rot6d back to the external rotation representation.
        euler/axisangle : no conversion needed.
        rot6d           : add identity back in relative-rotation mode; output is still rot6d.
        quat_*          : rot6d(6) → rotmat(3x3) → quat(4).
        rotmatrix       : rot6d(6) → rotmat(3x3) → flat(9).
        """
        if self._rotation_format in ('euler', 'axisangle'):
            return signal
        if self._rotation_format == 'rot6d':
            rot6d = signal[..., 3:9]
            if self._relative_rotation:
                rot6d = rot6d + self._identity_rot6d_like(rot6d)
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            out = torch.cat([signal[..., :3], rot6d], dim=-1)
            if res is not None:
                out = torch.cat((out, res), -1)
            return out
            # NOTE: if use euler->6d->euler
            # res = signal[..., 9:] if signal.size(-1) > 9 else None
            # rot6d = signal[..., 3:9]
            # euler = rotate6d_to_euler(rot6d)
            # signal = torch.cat([signal[..., :3], euler], dim=-1)
            # if res is not None:
            #     signal = torch.cat((signal, res), -1)
            # return signal
        # Else assume quaternion
        if self._rotation_format == 'rotmatrix':
            # signal: (..., 3 + 6 + 1)  — rot6d
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            rot6d = signal[..., 3:9]
            *bdims, _ = rot6d.shape
            mat = compute_rotation_matrix_from_ortho6d(rot6d.reshape(-1, 6))
            mat_flat = mat.reshape(*bdims, 9)
            signal = torch.cat([signal[..., :3], mat_flat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            return signal
        # quat_xyzw / quat_wxyz
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            if self._relative_rotation:
                rot = rot + self._identity_rot6d_like(rot)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            if self._relative_rotation:
                rot = rot + self._identity_rot6d_like(rot)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False,
        instr_mask=None,
        instr_lens=None,
        **kwargs,
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, nhand, 3+4+X)
            trajectory_mask: (B, trajectory_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            rgb2d: (B, num_2d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)
            instr_mask: optional (B, L) text padding mask for Qwen3-VL (1 = token, 0 = pad)

        Note:
            The input rotation is expressed as one of:
                a) quaternion (4D)  — converted to 6D internally.
                b) Euler angles (3D) — passed through as-is.
                c) axis-angle (3D)  — passed through as-is (relative LIBERO).
                d) rotation matrix flat (9D) — converted to 6D internally (absolute LIBERO).

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, nhand, 3+rot+1), at inference
        """
        # Inference, don't use gt_trajectory
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb3d,
                rgb2d,
                pcd,
                instruction,
                proprio,
                instr_mask=instr_mask,
                instr_lens=instr_lens,
            )

        # Training, use gt_trajectory to compute loss
        return self.compute_loss(
            gt_trajectory, trajectory_mask,
            rgb3d,
            rgb2d,
            pcd,
            instruction,
            proprio,
            instr_mask=instr_mask,
            instr_lens=instr_lens,
        )


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats
        )

        # Positional embeddings
        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, traj_feats,
            rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
            timesteps, proprio_feats,
            fps_scene_feats, fps_scene_pos,
            instr_feats, instr_pos
        )

        # Cross attention from gripper to full context
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = self.get_sa_feature_sequence(
            traj_feats, fps_scene_feats,
            rgb3d_feats, rgb2d_feats, instr_feats
        )
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

    def encode_denoising_timestep(self, timestep, proprio_feats):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        proprio_feats = proprio_feats.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
        return time_feats + curr_gripper_feats

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        return None, None, None

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        return torch.cat([traj_feats, fps_scene_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
