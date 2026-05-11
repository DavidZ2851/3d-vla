import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from .base import BaseTrainTester


class LiberoTrainTester(BaseTrainTester):
    """Train/test for LIBERO dataset.
    Normalizer strategy:
      axisangle (relative): ``rel_action`` from zarr is the unnormalized
      OSC_POSE delta command xyz + delta axisangle + signed binary gripper.
      rotmatrix/absolute: dataset-derived min/max for xyz only (3D).
    """
    XYZ_MIN = torch.tensor([-0.05, -0.05, -0.05], dtype=torch.float32)
    XYZ_MAX = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32)
    # Empirical rel_action axis-angle range in LIBERO zarr is within about +/-0.1875.
    # Use a tighter symmetric normalization range to improve rotation signal scale.
    AXISANGLE_MIN = torch.tensor([-0.2, -0.2, -0.2], dtype=torch.float32)
    AXISANGLE_MAX = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32)

    def _is_rank0(self):
        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    @torch.no_grad()
    def _compute_action_stats(self):
        if hasattr(self, "_libero_action_stats"):
            return self._libero_action_stats

        self._libero_action_stats = {
            "xyz_min": self.XYZ_MIN.clone(),
            "xyz_max": self.XYZ_MAX.clone(),
            "axisangle_min": self.AXISANGLE_MIN.clone(),
            "axisangle_max": self.AXISANGLE_MAX.clone(),
        }
        if self._is_rank0():
            print(
                "LIBERO fixed rel_action normalization range: "
                f"xyz_min={self.XYZ_MIN.tolist()}, xyz_max={self.XYZ_MAX.tolist()}, "
                f"axisangle_min={self.AXISANGLE_MIN.tolist()}, "
                f"axisangle_max={self.AXISANGLE_MAX.tolist()}, "
                "gripper=signed_binary"
            )
        return self._libero_action_stats

    def _copy_openness_normalizer(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        module._continuous_openness = False
        module._signed_binary_openness_output = True

    def get_model(self):
        model = super().get_model()
        self._copy_openness_normalizer(model)
        return model

    def load_checkpoint(self, model, ema_model, optimizer):
        result = super().load_checkpoint(model, ema_model, optimizer)
        self._copy_openness_normalizer(model)
        self._copy_openness_normalizer(ema_model)
        return result

    @torch.no_grad()
    def get_workspace_normalizer(self):
        if self.args.relative_action and self.args.rotation_format == 'axisangle':
            stats = self._compute_action_stats()
            min_ = torch.cat([stats["xyz_min"], stats["axisangle_min"]])
            max_ = torch.cat([stats["xyz_max"], stats["axisangle_max"]])
            self._xyz_min = min_.clone()
            self._xyz_max = max_.clone()
            return torch.nn.Parameter(torch.stack([min_, max_]), requires_grad=False)
        # rotmatrix (absolute): compute xyz bounds from training data
        return super().get_workspace_normalizer(ndims=3)

    def _compute_libero_metrics(self, pred, gt):
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        axisangle_l1 = (pred[..., 3:6] - gt[..., 3:6]).abs().sum(-1)
        gripper = (pred[..., 6:] == gt[..., 6:]).all(dim=-1)

        if not self.args.bimanual:
            pos_l2 = pos_l2[..., 0]
            axisangle_l1 = axisangle_l1[..., 0]
            gripper = gripper[..., 0]

        return {
            "traj_pos_l2": pos_l2.mean(),
            "traj_pos_acc_001": (pos_l2 < 0.01).float().mean(),
            "traj_axisangle_l1": axisangle_l1.mean(),
            "traj_axisangle_acc_0025": (axisangle_l1 < 0.025).float().mean(),
            "traj_gripper": gripper.float().mean(),
        }

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["extrinsics"], sample["intrinsics"],
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
        if not (self.args.relative_action and self.args.rotation_format == 'axisangle'):
            return super().evaluate_nsteps(model, loader, step_id, val_iters, split, epoch)

        # Some callers pass val_iters=0 (or a negative value) to effectively disable
        # evaluation on that split. Avoid an empty `values` dict and KeyError.
        if val_iters is not None and val_iters <= 0:
            if self._is_rank0():
                print(f"Step {step_id}: skipped {split} evaluation (val_iters={val_iters}).")
            return 0.0

        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if val_iters is not None and i >= val_iters:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].to(device, non_blocking=True)
            losses = self._compute_libero_metrics(pred_action, gt_action)

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
                wandb.log(values, step=step_id)

            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        # If loader is empty or val_iters=0, return a safe default.
        return values.get(f'{split}-losses/mean/traj_pos_acc_001', 0.0)
