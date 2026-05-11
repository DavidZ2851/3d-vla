import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm

from modeling.utils.utils import euler_xyz_to_matrix, matrix_to_quaternion
from .base import BaseTrainTester


class RoboTwinTrainTester(BaseTrainTester):
    XYZ_MIN = torch.tensor([-0.025, -0.022, -0.020], dtype=torch.float32)
    XYZ_MAX = torch.tensor([0.025, 0.022, 0.020], dtype=torch.float32)
    EULER_MIN = torch.tensor([-0.15, -0.15, -0.15], dtype=torch.float32)
    EULER_MAX = torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32)
    EULER_CLAMP_MIN = torch.tensor([-0.20, -0.20, -0.20], dtype=torch.float32)
    EULER_CLAMP_MAX = torch.tensor([0.20, 0.20, 0.20], dtype=torch.float32)
    OPENNESS_MIN = torch.tensor(-0.07537692785263062, dtype=torch.float32)
    OPENNESS_MAX = torch.tensor(0.07537692785263062, dtype=torch.float32)
    OPENNESS_L1_ACC_THRESH = 0.01

    def _is_rank0(self):
        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    @torch.no_grad()
    def _compute_action_stats(self):
        if hasattr(self, "_robotwin_action_stats"):
            return self._robotwin_action_stats

        self._robotwin_action_stats = {
            "xyz_min": self.XYZ_MIN.clone(),
            "xyz_max": self.XYZ_MAX.clone(),
            "euler_min": self.EULER_MIN.clone(),
            "euler_max": self.EULER_MAX.clone(),
            "openness_min": self.OPENNESS_MIN.clone(),
            "openness_max": self.OPENNESS_MAX.clone(),
        }
        if self._is_rank0():
            msg = (
                "RoboTwin2 fixed rel_action normalization range: "
                f"xyz_min={self.XYZ_MIN.tolist()}, xyz_max={self.XYZ_MAX.tolist()}"
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
            print(msg)
        return self._robotwin_action_stats

    def get_model(self):
        model = super().get_model()
        model._continuous_openness = True
        stats = self._compute_action_stats()
        if hasattr(model, "openness_normalizer"):
            model.openness_normalizer.copy_(
                torch.stack([stats["openness_min"], stats["openness_max"]]).to(
                    dtype=model.openness_normalizer.dtype
                )
            )
        return model

    def _copy_openness_normalizer(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        if not hasattr(module, "openness_normalizer"):
            return
        stats = self._compute_action_stats()
        module.openness_normalizer.copy_(
            torch.stack([stats["openness_min"], stats["openness_max"]]).to(
                device=module.openness_normalizer.device,
                dtype=module.openness_normalizer.dtype,
            )
        )

    def load_checkpoint(self, model, ema_model, optimizer):
        result = super().load_checkpoint(model, ema_model, optimizer)
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
        buffer = float(self.args.workspace_normalizer_buffer)
        xyz_min = (stats["xyz_min"] - buffer).to(action.device, action.dtype)
        xyz_max = (stats["xyz_max"] + buffer).to(action.device, action.dtype)

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

    @staticmethod
    def _euler_xyz_to_quat_wxyz(euler):
        mat = euler_xyz_to_matrix(euler)
        return matrix_to_quaternion(mat)

    def _compute_robotwin_metrics(self, pred, gt):
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        if self.args.rotation_format == "euler":
            pred_quat = self._euler_xyz_to_quat_wxyz(pred[..., 3:6])
            gt_quat = self._euler_xyz_to_quat_wxyz(gt[..., 3:6])
        else:
            pred_quat = pred[..., 3:-1]
            gt_quat = gt[..., 3:-1]
        quat_l1 = self._quat_l1(pred_quat, gt_quat)
        openness_l1 = (pred[..., -1] - gt[..., -1]).abs()
        return {
            "traj_pos_l2": pos_l2.mean(),
            "traj_pos_acc_001": (pos_l2 < 0.01).float().mean(),
            "traj_rot_l1": quat_l1.mean(),
            "traj_rot_acc_0025": (quat_l1 < 0.025).float().mean(),
            "traj_openness_l1": openness_l1.mean(),
            "traj_openness_l1_acc_001": (
                openness_l1 < RoboTwinTrainTester.OPENNESS_L1_ACC_THRESH
            ).float().mean(),
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

        return values[f'{split}-losses/mean/traj_pos_acc_001']
