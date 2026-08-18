"""ALOHA train/test (offline only; no online evaluation hook)."""

import json
import math
import os

import torch
import torch.distributed as dist
import wandb
from torch import nn
from tqdm import tqdm

from .base import BaseTrainTester


class AlohaTrainTester(BaseTrainTester):
    """
    ALOHA single-arm trainer.

    Normalization design:
      - Per-axis xyz: ALOHA labels are ANCHOR-RELATIVE, so large values are
        meaningful long-horizon targets, not incremental outliers. The envelope
        is the FULL train-split min/max of the rel_action xyz deltas, made
        symmetric per axis (S = max(|min|, |max|)) so zero/pauses stay centered
        in normalized space, then rounded OUTWARD (XYZ_ROUND_STEP) so nothing
        clips. Do not clamp ALOHA xyz labels: far anchor-relative targets are
        valid supervision.
        By default (USE_DATASET_XYZ_STATS) this envelope is RECOMPUTED at runtime
        from the loaded dataset's own xyz_delta_stats.json (written next to the
        zarr by aloha_to_zarr.py), so each ALOHA set is normalized to ITS data
        rather than inheriting another set's numbers:
            single-arm (5 task) -> S=[0.150, 0.160, 0.225]
            bimanual   (3 task) -> S=[0.200, 0.160, 0.250]
        The hardcoded XYZ_MIN/MAX below are the documented fallback used only if
        that sidecar is missing (or USE_DATASET_XYZ_STATS=False).
      - Openness: binary labels from the preprocessed LeRobot dataset
        (1=open, 0=closed). The model learns a closed probability from those
        open-state labels, then deployment maps closed/open back to safe raw
        gripper targets.
      - 6D rotation: NOT normalized (already unit-norm rows in [-1, 1]); the
        model only normalizes the first nrm_dim = 3 dims. Rot6d deltas stay as
        valid rot6d labels; identity rot6d is not subtracted inside diffusion.
    """
    val_eval_save_freq = 2000
    use_rgb2d_context = True
    # Recompute the xyz envelope from the loaded dataset's xyz_delta_stats.json;
    # fall back to the hardcoded (bimanual) XYZ_MIN/MAX if that sidecar is absent.
    USE_DATASET_XYZ_STATS = True
    XYZ_ROUND_STEP = 0.005  # round the symmetric envelope outward to this step
    XYZ_MIN = torch.tensor([-0.200, -0.159, -0.250], dtype=torch.float32)
    XYZ_MAX = torch.tensor([ 0.200,  0.159,  0.250], dtype=torch.float32)
    OPENNESS_MIN = torch.tensor(0.0, dtype=torch.float32)
    OPENNESS_MAX = torch.tensor(1.0, dtype=torch.float32)
    OPENNESS_LABEL_THRESH = 0.5

    # Unified multi-camera motion fusion (Design B): sample motion queries from
    # BOTH ALOHA head cameras (front=0, back=1 -- the rgb3d/pcd order set by
    # AlohaDataPreprocessor.process_obs) and concatenate the two per-camera latents
    # into ONE world-frame motion bank that all scene patches attend to (single
    # scene_motion_attn / alpha_scene gate). Set to (0,) to fall back to the
    # original front-only single-latent fusion. Applies to every ALOHA motion-latent
    # model (EEF and joint alike); models without a motion expert ignore it.
    MOTION_QUERY_CAMERAS = (0, 1)

    def _is_rank0(self):
        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    @staticmethod
    def _xyz_stats_path(train_data_dir):
        """xyz_delta_stats.json sits next to the *.zarr store (one dir up)."""
        if not train_data_dir:
            return ""
        return os.path.join(
            os.path.dirname(os.path.normpath(str(train_data_dir))),
            "xyz_delta_stats.json",
        )

    @classmethod
    def _design_envelope_from_stats(cls, path):
        """Symmetric, outward-rounded per-axis xyz envelope from a stats file.

        Uses the FULL train-split min/max of the anchor-relative xyz deltas:
        S_axis = ceil(max(|min|, |max|) / XYZ_ROUND_STEP) * XYZ_ROUND_STEP, so the
        envelope encloses every label (no clipping) and stays symmetric about 0.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if "train" in payload and isinstance(payload["train"], dict):
            payload = payload["train"]
        mn, mx = payload["min"], payload["max"]
        step = cls.XYZ_ROUND_STEP
        env = []
        for lo, hi in zip(mn, mx):
            s = max(abs(float(lo)), abs(float(hi)))
            env.append(round(math.ceil(s / step) * step, 6) if step > 0 else s)
        t = torch.tensor(env, dtype=torch.float32)
        return -t, t

    @torch.no_grad()
    def _compute_action_stats(self):
        if hasattr(self, "_aloha_action_stats"):
            return self._aloha_action_stats

        xyz_min = self.XYZ_MIN.clone()
        xyz_max = self.XYZ_MAX.clone()
        source = "hardcoded fallback"
        if self.USE_DATASET_XYZ_STATS:
            path = self._xyz_stats_path(getattr(self.args, "train_data_dir", ""))
            if path and os.path.exists(path):
                xyz_min, xyz_max = self._design_envelope_from_stats(path)
                source = f"dataset stats ({path})"
            elif self._is_rank0():
                print(
                    f"[ALOHA] WARNING: xyz_delta_stats.json not found at {path!r}; "
                    "using hardcoded fallback xyz envelope."
                )

        self._aloha_action_stats = {
            "xyz_min": xyz_min,
            "xyz_max": xyz_max,
            "openness_min": self.OPENNESS_MIN.clone(),
            "openness_max": self.OPENNESS_MAX.clone(),
        }
        if self._is_rank0():
            print(
                f"ALOHA rel_action xyz normalization [{source}]: "
                f"xyz_min={xyz_min.tolist()}, xyz_max={xyz_max.tolist()}, "
                f"openness=[{float(self.OPENNESS_MIN):.3f}, "
                f"{float(self.OPENNESS_MAX):.3f}] binary_open"
            )
        return self._aloha_action_stats

    @torch.no_grad()
    def _configure_motion_cameras(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        if hasattr(module, "_motion_query_cameras"):
            module._motion_query_cameras = tuple(self.MOTION_QUERY_CAMERAS)

    def _configure_aloha_model(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        self._configure_motion_cameras(model)
        # aloha_to_zarr.py labels every future xyz/rot6d target relative to the
        # current proprio anchor. Do not use the default cumulative relative
        # position reconstruction, and do not center rot6d by subtracting identity.
        module._relative_position_mode = "anchor"
        module._relative_rotation = False
        module._continuous_openness = False
        module._signed_binary_openness_output = False
        module._binary_openness_target_mode = "closed_from_open"
        module._binary_openness_target_threshold = self.OPENNESS_LABEL_THRESH
        module._binary_openness_inference_threshold = 0.5
        if getattr(module, "encoder", None) is not None:
            module.encoder.use_rgb2d_context = self.use_rgb2d_context
        if getattr(module, "prediction_head", None) is not None:
            module.prediction_head.use_rgb2d_context = self.use_rgb2d_context
        if not hasattr(module, "openness_normalizer"):
            return
        stats = self._compute_action_stats()
        module.openness_normalizer.copy_(
            torch.stack([stats["openness_min"], stats["openness_max"]]).to(
                device=module.openness_normalizer.device,
                dtype=module.openness_normalizer.dtype,
            )
        )

    def _copy_openness_normalizer(self, model):
        self._configure_aloha_model(model)

    def get_model(self):
        missing = object()
        old_proprio_nhand = getattr(self.args, "proprio_nhand", missing)
        self.args.proprio_nhand = 1
        try:
            model = super().get_model()
        finally:
            if old_proprio_nhand is missing:
                delattr(self.args, "proprio_nhand")
            else:
                self.args.proprio_nhand = old_proprio_nhand
        self._configure_aloha_model(model)
        return model

    def load_checkpoint(self, model, ema_model, optimizer, lr_scheduler=None, scaler=None):
        result = super().load_checkpoint(model, ema_model, optimizer, lr_scheduler, scaler)
        self._configure_aloha_model(model)
        self._configure_aloha_model(ema_model)
        return result

    @torch.no_grad()
    def get_workspace_normalizer(self):
        # xyz only (rot6d is not normalized). Fixed per-axis bounds (see class docstring).
        stats = self._compute_action_stats()
        buffer = float(self.args.workspace_normalizer_buffer)
        min_ = stats["xyz_min"].clone() - buffer
        max_ = stats["xyz_max"].clone() + buffer
        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    @torch.no_grad()
    def _clamp_action_xyz(self, action):
        self._last_xyz_clamp_rate = None
        return action

    @staticmethod
    def _rot6d_geodesic_deg(pred_rot6d, gt_rot6d):
        """Geodesic angle (deg) between rotations recovered from rot6d via Gram-Schmidt.

        Args:
            pred_rot6d, gt_rot6d: (..., 6)
        Returns:
            angle_deg: (...)
        """
        from modeling.utils.utils import compute_rotation_matrix_from_ortho6d

        flat_shape = pred_rot6d.shape[:-1]
        R_pred = compute_rotation_matrix_from_ortho6d(pred_rot6d.reshape(-1, 6))
        R_gt = compute_rotation_matrix_from_ortho6d(gt_rot6d.reshape(-1, 6))
        R_diff = torch.bmm(R_pred, R_gt.transpose(-1, -2))
        trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle = torch.acos(cos)
        return torch.rad2deg(angle).reshape(flat_shape)

    def _compute_aloha_metrics(self, pred, gt):
        # pred/gt: (B, T_or_1, NHAND=1, 10)  [xyz(3), rot6d(6), gripper(1)]
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        rot6d_l1 = (pred[..., 3:9] - gt[..., 3:9]).abs().sum(-1)
        rot_angle_deg = self._rot6d_geodesic_deg(pred[..., 3:9], gt[..., 3:9])
        pred_closed = pred[..., -1] >= 0.5
        gt_closed = gt[..., -1] < self.OPENNESS_LABEL_THRESH
        return {
            "traj_pos_l2": pos_l2.mean(),
            "traj_pos_acc_001": (pos_l2 < 0.01).float().mean(),
            "traj_rot6d_l1": rot6d_l1.mean(),
            "traj_rot_angle_deg": rot_angle_deg.mean(),
            "traj_rot_acc_5deg": (rot_angle_deg < 5.0).float().mean(),
            "traj_gripper": (pred_closed == gt_closed).float().mean(),
        }

    # Pure-2D models (2DFA encoder + non-RoPE head). Neither their encoder nor
    # their action head has an rgb2d branch -- Encoder2DFA discards rgb2d and pcd,
    # and TransformerHead2D / JointFlowHead never read rgb2d_feats -- so a wrist
    # left on the rgb2d path would be silently dropped.
    WRIST_AS_SCENE_MODELS = ("flower_actor_2d", "flower_actor_2d_joint")

    def _feed_wrist_as_scene(self):
        """The pure-2D visual-language models (flower_actor_2d,
        flower_actor_2d_joint) have no rgb2d branch: their 2DFA encoder only
        consumes rgb3d scene cameras (tagged with camera-id embeddings) and
        discards rgb2d + pcd. Route the ALOHA wrist as a third scene camera for
        those models so all three cameras (front, back, wrist) are used; keep the
        default front/back-3D + wrist-2D-context routing for the 3D models. Gate on
        model_type so existing 3D runs are unchanged. Requires --num_camera_ids 3.
        """
        return getattr(self.args, "model_type", "") in self.WRIST_AS_SCENE_MODELS

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        if self.args.relative_action:
            sample["action"] = self._clamp_action_xyz(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgb3d, rgb2d, pcds = self.preprocessor.process_obs(
            sample["rgb_front"], sample["rgb_back"], sample.get("rgb_wrist"),
            sample["depth_front"], sample["depth_back"], augment=augment,
            wrist_as_scene=self._feed_wrist_as_scene(),
        )
        return (
            sample["action"],
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device="cuda"),
            rgb3d,
            rgb2d,
            pcds,
            sample["instr"],
            proprio,
        )

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split="val", epoch=None):
        if val_iters == 0:
            if self._is_rank0():
                print(f"Step {step_id}: skipped {split} evaluation (val_iters={val_iters}).")
            return 0.0
        # val_iters counts dataloader iterations, not raw zarr datapoints. With
        # B_val=32, val_iters=500 means up to 16k datapoints, so ALOHA's 2,854
        # row val split is a full-val pass once the loader exhausts.
        max_eval_batches = val_iters
        if val_iters is not None and val_iters < 0:
            max_eval_batches = None
        if max_eval_batches is not None and hasattr(loader, "__len__"):
            max_eval_batches = min(int(max_eval_batches), len(loader))

        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].to(device, non_blocking=True)
            losses = self._compute_aloha_metrics(pred_action, gt_action)
            clamp_rate = getattr(self, "_last_xyz_clamp_rate", None)
            if clamp_rate is not None:
                losses["traj_xyz_clamp_rate"] = clamp_rate.to(device)

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

        return values.get(f"{split}-losses/mean/traj_pos_acc_001", 0.0)


class AlohaHeadRGBDTrainTester(AlohaTrainTester):
    """ALOHA trainer variant with only front/back RGBD context."""

    use_rgb2d_context = False


class AlohaJointTrainer(AlohaTrainTester):
    """Trainer for the ALOHA joint-action model
    (``flower_actor_3d_motion_latent_joint``).

    Reuses the ALOHA single-arm EEF pipeline (front/back 3D + wrist 2D rgb,
    anchor-relative rot6d EEF ``action`` for the model's EEF chain-of-thought,
    xyz workspace normalizer, binary openness config) and additionally threads the
    arm-joint targets/state through to the model and loads the per-joint
    normalizer. Validation is offline joint metrics (arm-joint L1 + gripper
    accuracy); there is no in-repo online ALOHA env rollout.

    Units: ``AlohaJointDataset`` delivers arm joints in RADIANS (the zarr stores
    degrees); the normalizer below is in radians (p99.9 of |value| x1.15 margin on
    the aloha_zarr train split, scripts/aloha/compute_joint_rel_action_stats.py).
    The model clamps normalized targets to [-joint_norm_clamp, joint_norm_clamp]
    (default 2.0). Gripper (dim 6) is binary command 0/1 and not normalized.

    Dataset name: ``AlohaJoint``.
    """

    # Symmetric per-joint [min, max] for the 6 anchor-relative arm-joint deltas,
    # in RADIANS. Hardcoded fallback derived from the bimanual aloha_zarr train
    # split (p99.9 x1.15); see compute_joint_rel_action_stats.py. When
    # USE_DATASET_JOINT_STATS is on and an aloha_joint_rel_action_stats.json sits
    # next to the loaded zarr (e.g. the combined bimanual+single-arm set), the
    # per-joint `hi` from that sidecar is used instead so each ALOHA set is
    # normalized to ITS OWN joint-delta envelope (mirrors USE_DATASET_XYZ_STATS).
    JOINT_NORM_HI = torch.tensor([0.270, 0.722, 0.544, 0.634, 0.482, 0.692])
    USE_DATASET_JOINT_STATS = True

    # MOTION_QUERY_CAMERAS / _configure_motion_cameras are inherited from
    # AlohaTrainTester (front+back motion queries for every ALOHA motion model).

    @staticmethod
    def _joint_stats_path(train_data_dir):
        """aloha_joint_rel_action_stats.json sits next to the *.zarr store."""
        if not train_data_dir:
            return ""
        return os.path.join(
            os.path.dirname(os.path.normpath(str(train_data_dir))),
            "aloha_joint_rel_action_stats.json",
        )

    def _resolve_joint_hi(self):
        """Per-joint symmetric `hi` (radians) for the 6 arm joints. Prefer the
        dataset's own sidecar; fall back to the hardcoded bimanual envelope."""
        hi = self.JOINT_NORM_HI.clone()
        source = "hardcoded fallback (bimanual)"
        if self.USE_DATASET_JOINT_STATS:
            path = self._joint_stats_path(getattr(self.args, "train_data_dir", ""))
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                vals = payload.get("suggested_normalizer", {}).get("hi")
                if vals is None or len(vals) != hi.numel():
                    raise ValueError(
                        f"{path}: suggested_normalizer.hi must have {hi.numel()} "
                        f"entries (6 arm joints), got {vals!r}."
                    )
                hi = torch.tensor([float(v) for v in vals], dtype=hi.dtype)
                source = f"dataset stats ({path})"
        if self._is_rank0():
            print(f"[ALOHA joint] arm-joint normalizer hi [{source}]: {hi.tolist()}")
        return hi

    @torch.no_grad()
    def get_joint_normalizer(self):
        hi = self._resolve_joint_hi()
        return torch.stack([-hi, hi])  # (2, 6)

    @torch.no_grad()
    def _copy_joint_normalizer(self, model):
        if model is None:
            return
        module = model.module if hasattr(model, "module") else model
        if not hasattr(module, "joint_normalizer"):
            return
        jn = self.get_joint_normalizer().to(
            device=module.joint_normalizer.device,
            dtype=module.joint_normalizer.dtype,
        )
        if tuple(jn.shape) != tuple(module.joint_normalizer.shape):
            raise ValueError(
                "joint_normalizer shape mismatch: model expects "
                f"{tuple(module.joint_normalizer.shape)} but trainer provides "
                f"{tuple(jn.shape)}. Set --joint_action_dim 6 for ALOHA."
            )
        module.joint_normalizer.copy_(jn)

    def get_model(self):
        model = super().get_model()
        self._copy_joint_normalizer(model)
        self._configure_motion_cameras(model)
        return model

    def load_checkpoint(self, model, ema_model, optimizer, lr_scheduler=None, scaler=None):
        result = super().load_checkpoint(model, ema_model, optimizer, lr_scheduler, scaler)
        # Re-assert the canonical normalizer (harmless on resume).
        self._copy_joint_normalizer(model)
        self._copy_joint_normalizer(ema_model)
        # The multi-camera flag is a runtime attribute (no params), so re-assert it
        # on resume in case an older checkpoint's module shipped the default.
        self._configure_motion_cameras(model)
        self._configure_motion_cameras(ema_model)
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

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split="val", epoch=None):
        """Offline joint metrics: arm-joint L1 (radians) and gripper accuracy on
        the joint chunk. Overrides the EEF-action eval (which would shape-mismatch
        the joint output)."""
        if val_iters == 0:
            if self._is_rank0():
                print(f"Step {step_id}: skipped {split} evaluation (val_iters={val_iters}).")
            return 0.0
        max_eval_batches = val_iters
        if val_iters is not None and val_iters < 0:
            max_eval_batches = None
        if max_eval_batches is not None and hasattr(loader, "__len__"):
            max_eval_batches = min(int(max_eval_batches), len(loader))

        model.eval()
        arm_l1_sum, grip_acc_sum, count = 0.0, 0.0, 0
        for i, sample in enumerate(tqdm(loader)):
            if max_eval_batches is not None and i >= max_eval_batches:
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
