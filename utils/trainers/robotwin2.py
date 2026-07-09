import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .base import BaseTrainTester


class RoboTwinTrainTester(BaseTrainTester):
    # Hand-set normalization for RoboTwin2 anchor-relative xyz labels.
    # These are authoritative when USE_HARDCODED_XYZ_NORM=True: the
    # --robotwin2_action_stats / ROBOTWIN2_ACTION_STATS JSON is NOT used to
    # override the xyz range. Tuned from the 6-task action16 stats
    # (robotwin2_action16_raw_6tasks_rel_action_stats.json); clamp = 2x range
    # => ~+/-[0.56, 0.52, 0.40], which covers ~p99.9 of the deltas.
    USE_HARDCODED_XYZ_NORM = True
    XYZ_MIN = torch.tensor([-0.28, -0.26, -0.20], dtype=torch.float32)
    XYZ_MAX = torch.tensor([ 0.28,  0.26,  0.20], dtype=torch.float32)
    # Euler XYZ (roll=dim0, pitch=dim1, yaw=dim2), clamp = 2 × norm range:
    EULER_MIN = torch.tensor([-0.055, -0.085, -0.050], dtype=torch.float32)
    EULER_MAX = torch.tensor([ 0.055,  0.085,  0.050], dtype=torch.float32)
    EULER_CLAMP_MIN = torch.tensor([-0.110, -0.170, -0.100], dtype=torch.float32)
    EULER_CLAMP_MAX = torch.tensor([ 0.110,  0.170,  0.100], dtype=torch.float32)
    # Openness: binary absolute target from zarr, where 0=closed and 1=open.
    # The preprocessing thresholds raw openness at 0.5.
    OPENNESS_MIN = torch.tensor(0.0, dtype=torch.float32)
    OPENNESS_MAX = torch.tensor(1.0, dtype=torch.float32)
    OPENNESS_LABEL_THRESH = 0.5
    CLOSED_PROB_EXEC_THRESH = 0.5

    def _is_rank0(self):
        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    @staticmethod
    def _stats_path_from_args(args):
        path = getattr(args, "robotwin2_action_stats", None)
        if path:
            return str(path)
        return os.environ.get("ROBOTWIN2_ACTION_STATS", "")

    @staticmethod
    def _load_xyz_bounds_from_stats(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "train" in payload and isinstance(payload["train"], dict):
            payload = payload["train"]
        if "quat_wxyz" not in payload:
            raise KeyError(f"{path} does not contain quat_wxyz rel_action stats.")

        quat_stats = payload["quat_wxyz"]
        lows = []
        highs = []
        for dim_name in ("dx", "dy", "dz"):
            dim_lows = []
            dim_highs = []
            for arm_name in ("left", "right"):
                arm_stats = quat_stats[arm_name][dim_name]
                lo = arm_stats.get("p0.05", arm_stats.get("min"))
                hi = arm_stats.get("p99.95", arm_stats.get("max"))
                if lo is None or hi is None:
                    raise KeyError(f"Missing percentile/minmax stats for {arm_name}.{dim_name}")
                dim_lows.append(float(lo))
                dim_highs.append(float(hi))
            lo = min(dim_lows)
            hi = max(dim_highs)
            pad = max((hi - lo) * 0.20, 1e-4)
            lows.append(lo - pad)
            highs.append(hi + pad)
        return (
            torch.tensor(lows, dtype=torch.float32),
            torch.tensor(highs, dtype=torch.float32),
        )

    @torch.no_grad()
    def _compute_action_stats(self):
        if hasattr(self, "_robotwin_action_stats"):
            return self._robotwin_action_stats

        xyz_min = self.XYZ_MIN.clone()
        xyz_max = self.XYZ_MAX.clone()
        stats_path = self._stats_path_from_args(self.args)
        # When USE_HARDCODED_XYZ_NORM is set, the hand-set XYZ_MIN/XYZ_MAX above
        # are authoritative and the stats JSON is ignored for the xyz range.
        if stats_path and not self.USE_HARDCODED_XYZ_NORM:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"RoboTwin2 action stats file does not exist: {stats_path}"
                )
            xyz_min, xyz_max = self._load_xyz_bounds_from_stats(stats_path)

        self._robotwin_action_stats = {
            "xyz_min": xyz_min,
            "xyz_max": xyz_max,
            "euler_min": self.EULER_MIN.clone(),
            "euler_max": self.EULER_MAX.clone(),
            "openness_min": self.OPENNESS_MIN.clone(),
            "openness_max": self.OPENNESS_MAX.clone(),
        }
        if self._is_rank0():
            src = "HARDCODED (stats JSON ignored)" if self.USE_HARDCODED_XYZ_NORM else "stats"
            msg = (
                f"RoboTwin2 rel_action xyz normalization [{src}]: "
                f"xyz_min={xyz_min.tolist()}, xyz_max={xyz_max.tolist()}"
            )
            if self.args.rotation_format == "euler":
                msg += (
                    f", euler_min={self.EULER_MIN.tolist()}, "
                    f"euler_max={self.EULER_MAX.tolist()}, "
                    f"euler_clamp_min={self.EULER_CLAMP_MIN.tolist()}, "
                    f"euler_clamp_max={self.EULER_CLAMP_MAX.tolist()}"
                )
            msg += (
                f", openness=[{float(self.OPENNESS_MIN):.8f}, "
                f"{float(self.OPENNESS_MAX):.8f}]"
            )
            msg += (
                f", relative_rotation="
                f"{bool(getattr(self.args, 'robotwin2_relative_rotation', False))}"
            )
            if stats_path and not self.USE_HARDCODED_XYZ_NORM:
                msg += f", stats={stats_path}"
            print(msg)
        return self._robotwin_action_stats

    def get_model(self):
        if self.args.rotation_format != "quat_wxyz":
            raise ValueError(
                "RoboTwin2 training expects robotwin2_to_zarr.py rel_action "
                "labels: anchor-relative xyz, quat_wxyz, and binary command "
                f"0/1 openness. Use --rotation_format quat_wxyz, got {self.args.rotation_format!r}."
            )
        model = super().get_model()
        self._configure_robotwin_model(model)
        return model

    def _configure_robotwin_model(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        # robotwin2_to_zarr.py labels every future xyz target relative to current
        # proprio, optionally anchors future quat_wxyz to current proprio, and
        # stores future openness as binary command 0=closed, 1=open.
        module._continuous_openness = False
        module._signed_binary_openness_output = False
        module._binary_openness_target_mode = "closed_from_open"
        module._binary_openness_target_threshold = self.OPENNESS_LABEL_THRESH
        module._binary_openness_inference_threshold = self.CLOSED_PROB_EXEC_THRESH
        module._relative_position_mode = "anchor"
        module._relative_rotation = bool(getattr(self.args, "robotwin2_relative_rotation", False))
        stats = self._compute_action_stats()
        if hasattr(module, "openness_normalizer"):
            module.openness_normalizer.copy_(
                torch.stack([stats["openness_min"], stats["openness_max"]]).to(
                    device=module.openness_normalizer.device,
                    dtype=module.openness_normalizer.dtype,
                )
            )

    def _copy_openness_normalizer(self, model):
        self._configure_robotwin_model(model)

    def load_checkpoint(self, model, ema_model, optimizer, lr_scheduler=None, scaler=None):
        result = super().load_checkpoint(
            model, ema_model, optimizer, lr_scheduler, scaler
        )
        self._copy_openness_normalizer(model)
        self._copy_openness_normalizer(ema_model)
        return result

    @torch.no_grad()
    def get_workspace_normalizer(self):
        stats = self._compute_action_stats()
        min_ = stats["xyz_min"].clone()
        max_ = stats["xyz_max"].clone()
        if self.args.rotation_format == "euler":
            min_ = torch.cat([min_, stats["euler_min"].clone()])
            max_ = torch.cat([max_, stats["euler_max"].clone()])
        buffer = float(self.args.workspace_normalizer_buffer)
        min_ = min_ - buffer
        max_ = max_ + buffer
        self._xyz_min = min_.clone()
        self._xyz_max = max_.clone()
        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    @torch.no_grad()
    def _clamp_action_xyz(self, action):
        stats = self._compute_action_stats()
        # Clamp at 2× norm range so out-of-range values hit ≤ ±2 in normalized space.
        xyz_min = (2 * stats["xyz_min"]).to(action.device, action.dtype)
        xyz_max = (2 * stats["xyz_max"]).to(action.device, action.dtype)

        out = action.clone()
        xyz = out[..., :3]
        clamped_xyz = torch.minimum(torch.maximum(xyz, xyz_min), xyz_max)
        self._last_xyz_clamp_rate = (
            (clamped_xyz != xyz).any(dim=-1).float().mean().detach()
        )
        out[..., :3] = clamped_xyz
        self._last_euler_clamp_rate = None
        if self.args.rotation_format == "euler":
            euler_min = self.EULER_CLAMP_MIN.to(action.device, action.dtype)
            euler_max = self.EULER_CLAMP_MAX.to(action.device, action.dtype)
            euler = out[..., 3:6]
            clamped_euler = torch.minimum(torch.maximum(euler, euler_min), euler_max)
            self._last_euler_clamp_rate = (
                (clamped_euler != euler).any(dim=-1).float().mean().detach()
            )
            out[..., 3:6] = clamped_euler
        return out

    @staticmethod
    def _quat_l1(pred_quat, gt_quat):
        quat_l1 = (pred_quat - gt_quat).abs().sum(-1)
        quat_l1_alt = (pred_quat + gt_quat).abs().sum(-1)
        return torch.minimum(quat_l1, quat_l1_alt)

    def _compute_robotwin_metrics(self, pred, gt):
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        pred_quat = F.normalize(pred[..., 3:7], dim=-1, eps=1e-6)
        gt_quat = F.normalize(gt[..., 3:7], dim=-1, eps=1e-6)
        quat_l1 = self._quat_l1(pred_quat, gt_quat)
        # pred[..., -1] is p_closed (high=closed), gt[..., -1] is binary openness
        # (high=open). Compare predicted-closed vs gt-closed at the 0.5 boundary.
        gt_closed = gt[..., -1] < self.OPENNESS_LABEL_THRESH
        pred_closed = pred[..., -1] >= 0.5
        return {
            "traj_pos_l2": pos_l2.mean(),
            # 0.005 m (5 mm) tracks endpoint precision for anchor-relative TCP targets.
            "traj_pos_acc_0005": (pos_l2 < 0.005).float().mean(),
            "traj_rot_l1": quat_l1.mean(),
            "traj_rot_acc_0025": (quat_l1 < 0.025).float().mean(),
            "traj_gripper": (pred_closed == gt_closed).float().mean(),
        }

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        if self.args.relative_action:
            sample["action"] = self._clamp_action_xyz(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])

        rgb_front = sample["rgb_head"]
        rgb_left = sample["rgb_left"]
        rgb_right = sample["rgb_right"]
        depth_front = sample["depth_head"]
        depth_left = sample["depth_left"]
        depth_right = sample["depth_right"]
        extrinsics = sample["extrinsics"]
        if extrinsics.size(1) != 3:
            raise ValueError(
                "RoboTwin2 expects extrinsics ordered as head/left/right, got "
                f"{tuple(extrinsics.shape)}."
            )
        extrinsics_front = extrinsics[:, 0]
        extrinsics_left = extrinsics[:, 1]
        extrinsics_right = extrinsics[:, 2]

        rgbs, pcds = self.preprocessor.process_obs(
            rgb_front,
            rgb_left,
            rgb_right,
            depth_front,
            depth_left,
            depth_right,
            extrinsics_front,
            extrinsics_left,
            extrinsics_right,
            augment=augment
        )
        return (
            sample["action"],
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val', epoch=None):
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].to(device, non_blocking=True)
            losses = self._compute_robotwin_metrics(pred_action, gt_action)
            clamp_rate = getattr(self, "_last_xyz_clamp_rate", None)
            if clamp_rate is not None:
                losses["traj_xyz_clamp_rate"] = clamp_rate.to(device)
            euler_clamp_rate = getattr(self, "_last_euler_clamp_rate", None)
            if euler_clamp_rate is not None:
                losses["traj_euler_clamp_rate"] = euler_clamp_rate.to(device)

            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        values = {k: v.mean().item() for k, v in values.items()}
        if self._is_rank0():
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)
                if epoch is not None:
                    values["epoch"] = epoch
                import wandb
                wandb.log(values, step=step_id)

            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values[f'{split}-losses/mean/traj_pos_acc_0005']


class RoboTwin2JointTrainer(RoboTwinTrainTester):
    """Trainer for the RoboTwin2 joint-action models
    (``flower_actor_3d_motion_latent_joint`` and ``flower_actor_2d_joint``).

    Reuses the RoboTwin2 EEF pipeline (rgb/depth/pcd, EEF ``action`` = rel_action
    for the 3D model's EEF chain-of-thought, quat workspace normalizer, openness
    config) and additionally threads the bimanual joint targets/state through to
    the model and loads the per-joint normalizer. Validation is offline joint
    metrics (arm-joint L1 + gripper accuracy); online env rollout uses joint
    position control and lives in online_evaluation_robotwin.

    Dataset name: ``RoboTwin2Joint``.
    """

    # Symmetric per-joint [min, max] for the anchor-relative arm-joint deltas
    # (6 joints, shared across both arms). Derived as p99.9 of |value| x1.15
    # margin on the 15-task robotwin2_zarr_joint train split via
    # scripts/robotwin2/compute_joint_rel_action_stats.py (-> datasets/
    # robotwin2_joint_rel_action_stats.json). The model clamps normalized targets
    # to [-joint_norm_clamp, joint_norm_clamp] (default 2.0), bounding the rare
    # tail. Gripper (dim 6) is binary command 0/1 and not normalized.
    JOINT_NORM_HI = torch.tensor([0.751, 1.761, 1.563, 1.346, 1.034, 1.555])

    @torch.no_grad()
    def get_joint_normalizer(self):
        hi = self.JOINT_NORM_HI.clone()
        return torch.stack([-hi, hi])  # (2, 6)

    def get_model(self):
        model = super().get_model()
        module = model.module if hasattr(model, "module") else model
        self._copy_joint_normalizer(module)
        return model

    @torch.no_grad()
    def _copy_joint_normalizer(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        if hasattr(module, "joint_normalizer"):
            jn = self.get_joint_normalizer().to(
                device=module.joint_normalizer.device,
                dtype=module.joint_normalizer.dtype,
            )
            if tuple(jn.shape) != tuple(module.joint_normalizer.shape):
                raise ValueError(
                    "joint_normalizer shape mismatch: model expects "
                    f"{tuple(module.joint_normalizer.shape)} but trainer provides "
                    f"{tuple(jn.shape)}. Set --joint_action_dim 6 for RoboTwin2."
                )
            module.joint_normalizer.copy_(jn)

    def load_checkpoint(self, model, ema_model, optimizer, lr_scheduler=None, scaler=None):
        result = super().load_checkpoint(
            model, ema_model, optimizer, lr_scheduler, scaler
        )
        # Re-assert the canonical normalizer (harmless on resume; fixes older
        # checkpoints that may have shipped an identity normalizer).
        self._copy_joint_normalizer(model)
        self._copy_joint_normalizer(ema_model)
        return result

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        out = super().prepare_batch(sample, augment=augment)
        # Stash the joint targets/state for _run_model_forward (called immediately
        # after prepare_batch inside the inherited _model_forward).
        self._joint_batch = (
            self.preprocessor.process_joint_action(sample["joint_action"]),
            self.preprocessor.process_joint_proprio(sample["joint_proprio"]),
        )
        return out

    def _run_model_forward(self, model, action, action_mask, rgbs, rgb2d, pcds,
                           instr, prop, training, instr_mask=None, instr_lens=None,
                           grounding_3d=None, grounding_keep_mask=None):
        gt_joint, joint_proprio = self._joint_batch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training,
                instr_mask=instr_mask,
                instr_lens=instr_lens,
                gt_joint_trajectory=gt_joint,
                joint_proprio=joint_proprio,
            )

    @torch.no_grad()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val',
                        epoch=None):
        """Offline joint metrics: per-arm-joint L1 and gripper accuracy on the
        joint chunk. Overrides the EEF-action eval (which would shape-mismatch
        the joint output) and skips the online env rollout (separate harness)."""
        model.eval()
        arm_l1_sum, grip_acc_sum, count = 0.0, 0.0, 0
        for i, sample in enumerate(tqdm(loader)):
            if i == val_iters:
                break
            pred = self._model_forward(model, sample, training=False)  # (B,H,nh,7)
            gt = sample["joint_action"].cuda(non_blocking=True).float()
            n_arm = pred.shape[-1] - 1
            arm_l1_sum += (pred[..., :n_arm] - gt[..., :n_arm]).abs().mean().item()
            grip_pred = (pred[..., -1] >= 0.5).float()
            grip_acc_sum += (grip_pred == gt[..., -1]).float().mean().item()
            count += 1
        if count == 0:
            return 0.0
        arm_l1 = arm_l1_sum / count
        grip_acc = grip_acc_sum / count
        if self._is_rank0() and step_id > -1:
            self.writer.add_scalar(f"{split}-joint/arm_l1", arm_l1, step_id)
            self.writer.add_scalar(f"{split}-joint/gripper_acc", grip_acc, step_id)
            try:
                import wandb
                wandb.log(
                    {f"{split}-joint/arm_l1": arm_l1,
                     f"{split}-joint/gripper_acc": grip_acc},
                    step=step_id,
                )
            except Exception:
                pass
            print(f"Step {step_id}: {split} joint arm_l1={arm_l1:.4f} "
                  f"gripper_acc={grip_acc:.4f}")
        return arm_l1
