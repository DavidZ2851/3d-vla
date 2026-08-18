import inspect
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.pytorch3d_transforms import quaternion_multiply
from modeling.utils.utils import euler_xyz_to_matrix, matrix_to_quaternion

from .depth2cloud import RoboTwinDepth2Cloud


_QWEN_MODEL_TYPES = {
    "qwen_flower_actor_3d",
    "qwen_flower_actor_3d_stagewise",
    "qwen_flower_actor_3d_grounding_cond",
    "qwen_flower_actor_3d_grounding_cond_llm_visual",
    "qwen_flower_actor_3d_grounding_cond_qwen3d",
    "qwen_flower_actor_3d_grounding_pred",
    "qwen_pi0_llm_actor_3d",
    "qwen_pi05_llm_actor_3d",
    "qwen_dualexpert_3dfa_actor",
}

# Joint-space (qpos) action models. These predict anchor-relative arm-joint
# deltas (+ a binary gripper command) and are executed via RoboTwin's
# action_type="qpos" rather than the EEF/TCP "ee"/"delta_ee" paths.
_JOINT_MODEL_TYPES = {
    "flower_actor_3d_motion_latent_joint",
    "flower_actor_2d_joint",
}


def _checkpoint_uses_multiscale(state_dict):
    return any(key.startswith("encoder.multiscale_fuser.") for key in state_dict)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("ema_weight") or ckpt.get("weight")
        if state_dict is None:
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if isinstance(state_dict, dict):
            return state_dict
    raise ValueError(
        "Unable to find a valid state_dict in checkpoint. "
        "Expected one of: ema_weight, weight, state_dict, model_state_dict."
    )


def _strip_module_prefix(state_dict):
    model_weights = {}
    for key, value in state_dict.items():
        model_weights[key[7:] if key.startswith("module.") else key] = value
    return model_weights


def _get_checkpoint_lora_summary(state_dict):
    lora_keys = [
        key for key in state_dict
        if ".lora_A." in key or ".lora_B." in key
    ]
    has_vision_lora = any(".visual." in key for key in lora_keys)
    has_llm_lora = any(".language_model." in key for key in lora_keys)
    return {
        "any": bool(lora_keys),
        "vision": has_vision_lora,
        "llm": has_llm_lora,
    }


def _get_checkpoint_embedding_weight(state_dict):
    embedding_weight = state_dict.get("encoder.vis_proj.weight")
    if embedding_weight is not None:
        return embedding_weight
    for key, value in state_dict.items():
        if (
            key.startswith("encoder.multiscale_fuser.projections.")
            and key.endswith(".weight")
        ):
            return value
    return None


def _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict):
    if args.model_type not in _QWEN_MODEL_TYPES:
        return

    embedding_weight = _get_checkpoint_embedding_weight(state_dict)
    if embedding_weight is not None:
        ckpt_embedding_dim = int(embedding_weight.shape[0])
        if args.embedding_dim != ckpt_embedding_dim:
            print(
                "Overriding embedding_dim from checkpoint: "
                f"{args.embedding_dim} -> {ckpt_embedding_dim}"
            )
            args.embedding_dim = ckpt_embedding_dim

    ckpt_uses_multiscale = _checkpoint_uses_multiscale(state_dict)
    requested_multiscale = bool(getattr(args, "use_multiscale", False))
    if ckpt_uses_multiscale and not requested_multiscale:
        print("Checkpoint uses multiscale vision fusion; enabling use_multiscale for evaluation.")
        args.use_multiscale = True
    elif requested_multiscale and not ckpt_uses_multiscale:
        print("Checkpoint does not use multiscale vision fusion; disabling use_multiscale for evaluation.")
        args.use_multiscale = False

    lora_summary = _get_checkpoint_lora_summary(state_dict)
    requested_vision_lora = (
        getattr(args, "vision_lora_rank", 0) > 0
        and getattr(args, "vision_lora_num_layers", 0) > 0
    )
    requested_llm_lora = (
        getattr(args, "llm_lora_rank", 0) > 0
        and getattr(args, "llm_lora_num_layers", 0) > 0
    )
    if (requested_vision_lora or requested_llm_lora) and not lora_summary["any"]:
        print("Checkpoint has no LoRA weights; disabling requested LoRA settings for evaluation.")
        args.vision_lora_rank = 0
        args.vision_lora_num_layers = 0
        args.llm_lora_rank = 0
        args.llm_lora_num_layers = 0
        return

    if lora_summary["vision"] and not requested_vision_lora:
        raise ValueError(
            "Checkpoint contains vision LoRA weights, but evaluation did not enable matching vision LoRA args."
        )
    if lora_summary["llm"] and not requested_llm_lora:
        raise ValueError(
            "Checkpoint contains LLM LoRA weights, but evaluation did not enable matching llm_lora_* args."
        )


def _normalise_quaternion(quat):
    denom = np.linalg.norm(quat, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    return quat / denom


def _normalise_quaternion_positive_w(quat):
    quat = _normalise_quaternion(quat)
    sign = np.where(quat[..., :1] < 0.0, -1.0, 1.0).astype(np.float32)
    return quat * sign


def _invert_quaternion_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float32)
    denom = np.sum(quat * quat, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    inv = quat.copy()
    inv[..., 1:] *= -1.0
    return inv / denom


def _multiply_quaternion_wxyz(quat_a, quat_b):
    quat_a = np.asarray(quat_a, dtype=np.float32)
    quat_b = np.asarray(quat_b, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(quat_a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(quat_b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


ROBOTWIN_TCP_TO_EE_OFFSET_M = 0.12


def _quaternion_wxyz_to_matrix_np(quat):
    quat = _normalise_quaternion(np.asarray(quat, dtype=np.float32))
    w, x, y, z = np.moveaxis(quat, -1, 0)
    out_shape = quat.shape[:-1] + (3, 3)
    mat = np.empty(out_shape, dtype=np.float32)
    mat[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mat[..., 0, 1] = 2.0 * (x * y - z * w)
    mat[..., 0, 2] = 2.0 * (x * z + y * w)
    mat[..., 1, 0] = 2.0 * (x * y + z * w)
    mat[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mat[..., 1, 2] = 2.0 * (y * z - x * w)
    mat[..., 2, 0] = 2.0 * (x * z - y * w)
    mat[..., 2, 1] = 2.0 * (y * z + x * w)
    mat[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mat


class RoboTwinEvalPreprocessor:
    def __init__(self, num_history=1, custom_imsize=224, depth2cloud=None):
        self.num_history = num_history
        self.custom_imsize = custom_imsize
        self.depth2cloud = depth2cloud

    def process_proprio(self, proprio):
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        return proprio[:, -self.num_history:]

    def process_obs(self, rgbs, depth, extrinsics, intrinsics):
        pcds = self.depth2cloud(
            depth.cuda(non_blocking=True).float(),
            extrinsics.cuda(non_blocking=True).float(),
            intrinsics.cuda(non_blocking=True).float(),
        )
        rgbs = rgbs.cuda(non_blocking=True).float() / 255.0
        if self.custom_imsize is not None and self.custom_imsize != rgbs.size(-1):
            b, nc, _, _, _ = rgbs.shape
            rgbs = F.interpolate(
                rgbs.flatten(0, 1),
                (self.custom_imsize, self.custom_imsize),
                mode="bilinear",
                antialias=True,
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)
            pcds = F.interpolate(
                pcds.flatten(0, 1),
                (self.custom_imsize, self.custom_imsize),
                mode="bilinear",
                antialias=True,
            ).reshape(b, nc, -1, self.custom_imsize, self.custom_imsize)
        return rgbs, pcds.float()


class Model:
    CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")

    def __init__(self, args):
        self.args = args
        self._is_joint_model = self.args.model_type in _JOINT_MODEL_TYPES
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        self.policy = self.get_policy()
        self.preprocessor = RoboTwinEvalPreprocessor(
            num_history=self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=RoboTwinDepth2Cloud(),
        )
        # GT-replay control: when RW2_GT_REPLAY=<train.zarr> is set, step() ignores
        # the policy and instead feeds the recorded GT rel_action chunks through the
        # exact same deploy conversion path (rel->abs, env-quat, TCP->EE, env exec).
        # Isolates the pipeline+env from the model: if GT replay succeeds, the task
        # is solvable by correct actions and the model is the only failure source.
        self._gt_replay = False
        import os
        gt_path = os.environ.get("RW2_GT_REPLAY", "").strip()
        if gt_path:
            import zarr
            z = zarr.open(gt_path, mode="r")
            self._gt_rel = np.asarray(z["rel_action"]).astype(np.float32)        # (T,16,2,8)
            self._gt_anchor = np.asarray(z["proprioception"])[:, -1].astype(np.float32)  # (T,2,8) current pose
            self._gt_T = self._gt_rel.shape[0]
            self._gt_replay = True
            print(f"\033[95m[GT-REPLAY] feeding recorded actions from {gt_path} "
                  f"(T={self._gt_T} frames); policy is bypassed.\033[0m", flush=True)
        # RW2_SWAP_RB: swap the R and B channels of every camera image before the
        # policy sees it, so a checkpoint trained on miscolored data is evaluated
        # under the colors it actually learned.
        #
        # Background: robotwin2_to_zarr's _decode_rgb_bytes runs cv2.imdecode (which
        # already yields the stored channel order) and then applies an extra
        # COLOR_BGR2RGB, so the zarr it wrote has R and B transposed relative to the
        # simulator. Models trained on that zarr (train_logs/robotwin2/rw2_bad_color)
        # grounded the word "red" in the blue channel. The live env hands us true RGB,
        # so evaluating them straight would score them on a color mapping they never
        # saw -- swapping here reproduces the training colors and makes the comparison
        # fair. Leave unset for models trained on the corrected zarr (e.g. rw2_all):
        # for those, the swap would introduce the very bug it compensates for.
        self._swap_rb = os.environ.get("RW2_SWAP_RB", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if self._swap_rb:
            print(
                "\033[95m[RW2_SWAP_RB] feeding R/B-swapped images to the policy "
                "(matches the rw2_bad_color training colors).\033[0m",
                flush=True,
            )
        # Opt-in head-camera RGBD recorder (active only when ROBOTWIN_ROLLOUT_DUMP_DIR
        # is set); dumps per-episode rgb/depth/summary for motion-expert spaghetti viz.
        from .rollout_recorder import RoboTwinRolloutRecorder
        self._rollout_recorder = RoboTwinRolloutRecorder.maybe_from_env(
            getattr(args, "task_name", None)
        )
        self.reset()

    def get_policy(self):
        model_class = fetch_model_class(self.args.model_type)
        model_kwargs = dict(
            backbone=self.args.backbone,
            finetune_backbone=self.args.finetune_backbone,
            num_finetune_vision_layers=self.args.num_finetune_vision_layers,
            finetune_text_encoder=self.args.finetune_text_encoder,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
        )
        model_init_params = inspect.signature(model_class.__init__).parameters
        qwen_kwargs = {
            "vision_lora_rank": self.args.vision_lora_rank,
            "vision_lora_alpha": self.args.vision_lora_alpha,
            "vision_lora_dropout": self.args.vision_lora_dropout,
            "vision_lora_num_layers": self.args.vision_lora_num_layers,
            "vision_lora_target_modules": self.args.vision_lora_target_modules,
            "llm_lora_rank": self.args.llm_lora_rank,
            "llm_lora_alpha": self.args.llm_lora_alpha,
            "llm_lora_dropout": self.args.llm_lora_dropout,
            "llm_lora_num_layers": self.args.llm_lora_num_layers,
            "llm_lora_target_modules": self.args.llm_lora_target_modules,
            "use_multiscale": getattr(self.args, "use_multiscale", False),
            "gradient_checkpointing": getattr(
                self.args, "gradient_checkpointing", False
            ),
        }
        for key, value in qwen_kwargs.items():
            if key in model_init_params:
                model_kwargs[key] = value
        motion_latent_kwargs = {
            "motion_expert_config": getattr(self.args, "motion_expert_config", None),
            "motion_expert_checkpoint": getattr(self.args, "motion_expert_checkpoint", None),
            "motion_solver_steps": getattr(self.args, "motion_solver_steps", 10),
            "motion_steps_to_run": getattr(self.args, "motion_steps_to_run", 1),
            "motion_finetune": getattr(self.args, "motion_finetune", False),
            "motion_scene_fusion": getattr(self.args, "motion_scene_fusion", True),
            "motion_action_fusion": getattr(self.args, "motion_action_fusion", False),
        }
        for key, value in motion_latent_kwargs.items():
            if key in model_init_params:
                model_kwargs[key] = value
        joint_kwargs = {
            "joint_action_dim": getattr(self.args, "joint_action_dim", 6),
            "joint_proprio_dim": getattr(self.args, "joint_proprio_dim", 7),
            "gripper_dim": getattr(self.args, "gripper_dim", 1),
            "joint_num_layers": getattr(self.args, "joint_num_layers", 8),
            "joint_embedding_dim": getattr(self.args, "joint_embedding_dim", 0) or None,
            "joint_num_attn_heads": getattr(self.args, "joint_num_attn_heads", 0) or None,
            # 2D ablation tags RoboTwin2's 3 scene cameras (head/left/right);
            # must match the size baked into the trained checkpoint.
            "num_camera_ids": getattr(self.args, "num_camera_ids", 3),
        }
        for key, value in joint_kwargs.items():
            if key in model_init_params:
                model_kwargs[key] = value
        policy = model_class(**model_kwargs)
        if hasattr(policy, "_continuous_openness"):
            policy._continuous_openness = bool(self.args.continuous_openness)
        if hasattr(policy, "_binary_openness_target_mode"):
            policy._binary_openness_target_mode = "closed_from_open"
            policy._binary_openness_target_threshold = 0.5
            policy._binary_openness_inference_threshold = float(
                self.args.binary_closed_prob_threshold
            )
        if hasattr(policy, "_relative_position_mode"):
            policy._relative_position_mode = "anchor"
        if hasattr(policy, "_relative_rotation"):
            policy._relative_rotation = bool(getattr(self.args, "relative_rotation", False))
        return policy

    def reset(self):
        self.policy.eval()
        self.policy = self.policy.to(self.args.device)
        self.reset_obs()

    def reset_obs(self):
        # Flush the just-finished episode before clearing buffers (per-episode dump).
        recorder = getattr(self, "_rollout_recorder", None)
        if recorder is not None:
            recorder.flush_episode()
        self.obs_cache = []
        self._latest_obs = None
        self._proprio_history = []
        self._cached_intrinsics = None
        self._cached_head_extrinsics = None
        # Per-arm gripper latch state (sticky grasp). 0=open. Reset every episode.
        self._grip_closed = [False, False]
        self._grip_open_run = [0, 0]
        # GT-replay chunk cursor (advances by prediction_len per replan).
        self._gt_idx = 0
        # Per-sim-step pose ring for PROPRIO_HISTORY_STRIDE (decoupled history).
        self._pose_ring = []
        # Joint-proprio history (joint models). Most recent state is last.
        self._joint_proprio_history = []

    def load_pretrained_weights(self, weight_dict):
        print(f"Loading weights from {self.args.checkpoint}")
        incompatible = self.policy.load_state_dict(weight_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f"Missing checkpoint keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"Unexpected checkpoint keys ({len(unexpected)}): {unexpected[:10]}")

    def update_obs(self, observation):
        self._latest_obs = observation
        self.obs_cache.append(observation)
        if len(self.obs_cache) > self.args.num_history:
            self.obs_cache.pop(0)
        # Per-sim-step pose ring (used by PROPRIO_HISTORY_STRIDE to keep the
        # history offset fixed regardless of replan cadence). Cheap; only read
        # when the env var is set.
        try:
            self._pose_ring.append(self._extract_current_pose(observation))
        except KeyError:
            pass
        recorder = getattr(self, "_rollout_recorder", None)
        if recorder is not None:
            recorder.on_obs(observation)

    def _require_latest_obs(self):
        if self._latest_obs is None:
            raise ValueError("No RoboTwin observation cached. Call update_obs() first.")
        return self._latest_obs

    def _get_camera_dict(self, observation):
        obs = observation.get("observation", {})
        missing = [name for name in self.CAMERA_NAMES if name not in obs]
        if missing:
            raise KeyError(
                "Missing RoboTwin cameras for online eval: "
                f"{missing}. Expected {list(self.CAMERA_NAMES)}."
            )
        return obs

    def _get_intrinsics(self, camera_obs):
        if self._cached_intrinsics is None:
            intrinsics = []
            for name in self.CAMERA_NAMES:
                intrinsic = np.asarray(camera_obs[name]["intrinsic_cv"], dtype=np.float32)
                intrinsics.append(intrinsic)
            self._cached_intrinsics = np.stack(intrinsics, axis=0)
        return self._cached_intrinsics

    @staticmethod
    def _extrinsic_to_4x4(extrinsic, camera_name):
        extrinsic = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic.shape == (4, 4):
            return extrinsic
        if extrinsic.shape == (3, 4):
            out = np.eye(4, dtype=np.float32)
            out[:3, :4] = extrinsic
            return out
        raise ValueError(
            f"RoboTwin camera `{camera_name}` extrinsic_cv must have shape "
            f"(3, 4) or (4, 4), got {extrinsic.shape}."
        )

    def _get_extrinsics(self, camera_obs):
        if self._cached_head_extrinsics is None:
            self._cached_head_extrinsics = self._extrinsic_to_4x4(
                camera_obs["head_camera"]["extrinsic_cv"], "head_camera"
            )
        return np.stack(
            [
                self._cached_head_extrinsics,
                self._extrinsic_to_4x4(
                    camera_obs["left_camera"]["extrinsic_cv"], "left_camera"
                ),
                self._extrinsic_to_4x4(
                    camera_obs["right_camera"]["extrinsic_cv"], "right_camera"
                ),
            ],
            axis=0,
        )

    def _env_wxyz_to_dataset_wxyz(self, quat_wxyz):
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
        if self.args.robotwin_tcp_quat_convention == "wxyz":
            return _normalise_quaternion_positive_w(quat_wxyz)
        if self.args.robotwin_tcp_quat_convention == "legacy_xyzw_to_wxyz":
            return _normalise_quaternion_positive_w(quat_wxyz[..., [3, 0, 1, 2]])
        raise ValueError(
            "Unsupported robotwin_tcp_quat_convention: "
            f"{self.args.robotwin_tcp_quat_convention!r}."
        )

    def _dataset_wxyz_to_env_wxyz(self, quat_wxyz):
        quat_wxyz = _normalise_quaternion_positive_w(
            np.asarray(quat_wxyz, dtype=np.float32)
        )
        if self.args.robotwin_tcp_quat_convention == "wxyz":
            return quat_wxyz
        if self.args.robotwin_tcp_quat_convention == "legacy_xyzw_to_wxyz":
            return _normalise_quaternion_positive_w(quat_wxyz[..., [1, 2, 3, 0]])
        raise ValueError(
            "Unsupported robotwin_tcp_quat_convention: "
            f"{self.args.robotwin_tcp_quat_convention!r}."
        )

    def _to_model_quat(self, quat_wxyz):
        quat = self._env_wxyz_to_dataset_wxyz(quat_wxyz)
        if self.args.rotation_format == "quat_xyzw":
            return quat[..., [1, 2, 3, 0]]
        return quat

    def _to_env_quat(self, quat):
        quat = np.asarray(quat, dtype=np.float32)
        if self.args.rotation_format == "quat_xyzw":
            quat = quat[..., [3, 0, 1, 2]]
        return self._dataset_wxyz_to_env_wxyz(quat)

    def _quat_multiply_in_model_format(self, quat_a, quat_b):
        a = quat_a
        b = quat_b
        if self.args.rotation_format == "quat_xyzw":
            a = quat_a[:, [3, 0, 1, 2]]
            b = quat_b[:, [3, 0, 1, 2]]
        out = quaternion_multiply(
            torch.from_numpy(a).float(),
            torch.from_numpy(b).float(),
        ).cpu().numpy()
        if self.args.rotation_format == "quat_xyzw":
            out = out[:, [1, 2, 3, 0]]
        return _normalise_quaternion(out)

    def _extract_current_pose(self, observation):
        # (2, 8) current TCP pose [xyz, quat_wxyz(model conv), grip] for both arms.
        endpose = observation.get("endpose", {})
        required = ("left_tcp_pose", "right_tcp_pose", "left_gripper", "right_gripper")
        missing = [key for key in required if key not in endpose]
        if missing:
            raise KeyError(
                "Missing RoboTwin TCP/endpose entries for online eval: "
                f"{missing}. Enable data_type.endpose in the task config."
            )

        left_pose = np.asarray(endpose["left_tcp_pose"], dtype=np.float32)
        right_pose = np.asarray(endpose["right_tcp_pose"], dtype=np.float32)
        left_pose = np.concatenate([left_pose[:3], self._to_model_quat(left_pose[3:7])], axis=0)
        right_pose = np.concatenate([right_pose[:3], self._to_model_quat(right_pose[3:7])], axis=0)
        left_grip = float(endpose["left_gripper"])
        right_grip = float(endpose["right_gripper"])
        if not self.args.continuous_openness:
            # robotwin2_to_zarr.py binarizes the proprio gripper at 0.5 (>=0.5 ->
            # 1 open, else 0 closed). Match it here so the gripper scalar the model
            # receives online has the same {0,1} distribution as in training;
            # otherwise mid-transition continuous values are off-distribution input
            # the model never saw.
            left_grip = 1.0 if left_grip >= 0.5 else 0.0
            right_grip = 1.0 if right_grip >= 0.5 else 0.0
        return np.stack(
            [
                np.concatenate([left_pose, [left_grip]], axis=0),
                np.concatenate([right_pose, [right_grip]], axis=0),
            ],
            axis=0,
        ).astype(np.float32)  # (2, 8)

    def _extract_proprio(self, observation):
        current = self._extract_current_pose(observation)

        # PROPRIO_HISTORY_STRIDE decouples the history spacing from the replan
        # cadence. Training built proprio with offsets [-stride, 0] (stride=16);
        # by default the wrapper advances _proprio_history once per replan, so a
        # full-16 chunk happens to reproduce that spacing -- but any other
        # EXEC_HORIZON would corrupt it. When the env var is set we instead read
        # the -stride frame from a per-sim-step pose ring, keeping the trained
        # spacing regardless of how often we replan.
        import os
        stride_env = os.environ.get("PROPRIO_HISTORY_STRIDE", "").strip()
        if stride_env:
            stride = int(stride_env)
            ring = self._pose_ring
            # ring[-1] is the current pose (pushed by update_obs); pick frames at
            # offsets 0, -stride, -2*stride, ... then order oldest->newest.
            n = self.args.num_history
            idxs = [max(0, len(ring) - 1 - i * stride) for i in range(n)][::-1]
            history = [ring[j] for j in idxs]
            proprio_np = np.stack(history, axis=0)[None]  # (1, nhist, 2, 8)
            return torch.from_numpy(proprio_np)

        self._proprio_history.append(current)
        if len(self._proprio_history) > self.args.num_history:
            self._proprio_history.pop(0)
        if len(self._proprio_history) < self.args.num_history:
            pad_count = self.args.num_history - len(self._proprio_history)
            history = [self._proprio_history[0]] * pad_count + self._proprio_history
        else:
            history = self._proprio_history
        proprio_np = np.stack(history, axis=0)[None]  # (1, nhist, 2, 8)
        return torch.from_numpy(proprio_np)

    def _current_joint_state(self, observation):
        """Return (left_arm(6), right_arm(6), left_grip, right_grip) from the
        RoboTwin qpos obs vector [left_arm(6), left_grip(1), right_arm(6),
        right_grip(1)]. Requires data_type.qpos=true in the task config."""
        joint = observation.get("joint_action", {})
        vector = joint.get("vector", None)
        if vector is None:
            raise KeyError(
                "Missing RoboTwin joint_action.vector for joint online eval. "
                "Enable data_type.qpos=true in the task config."
            )
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != 14:
            raise ValueError(
                "Expected a 14-dim RoboTwin qpos vector [left_arm(6), "
                f"left_grip(1), right_arm(6), right_grip(1)], got {vector.shape[0]}."
            )
        left_arm = vector[0:6]
        left_grip = float(vector[6])
        right_arm = vector[7:13]
        right_grip = float(vector[13])
        return left_arm, right_arm, left_grip, right_grip

    def _extract_joint_proprio(self, observation):
        """Build (1, nhist, nhand=2, 7) joint proprio: per arm [6 arm joints,
        1 binary gripper command]. Mirrors the training joint store, whose
        gripper column is the binary command (0=close, 1=open)."""
        left_arm, right_arm, left_grip, right_grip = self._current_joint_state(
            observation
        )
        if not self.args.continuous_openness:
            left_grip = 1.0 if left_grip >= 0.5 else 0.0
            right_grip = 1.0 if right_grip >= 0.5 else 0.0
        current = np.stack(
            [
                np.concatenate([left_arm, [left_grip]], axis=0),
                np.concatenate([right_arm, [right_grip]], axis=0),
            ],
            axis=0,
        ).astype(np.float32)  # (2, 7)

        self._joint_proprio_history.append(current)
        if len(self._joint_proprio_history) > self.args.num_history:
            self._joint_proprio_history.pop(0)
        if len(self._joint_proprio_history) < self.args.num_history:
            pad = self.args.num_history - len(self._joint_proprio_history)
            history = [self._joint_proprio_history[0]] * pad + self._joint_proprio_history
        else:
            history = self._joint_proprio_history
        proprio_np = np.stack(history, axis=0)[None]  # (1, nhist, 2, 7)
        return torch.from_numpy(proprio_np)

    def _current_joint_anchor(self, observation):
        """Absolute current arm joints per arm, (nhand=2, 6). The anchor for the
        anchor-relative joint deltas the model predicts."""
        left_arm, right_arm, _, _ = self._current_joint_state(observation)
        return np.stack([left_arm, right_arm], axis=0).astype(np.float32)  # (2, 6)

    @torch.no_grad()
    def preprocess(self, observation):
        camera_obs = self._get_camera_dict(observation)
        intrinsics = self._get_intrinsics(camera_obs)
        extrinsics = self._get_extrinsics(camera_obs)
        proprio = self._extract_proprio(observation)

        rgb_arrays = []
        depth_arrays = []
        for name in self.CAMERA_NAMES:
            rgb = np.asarray(camera_obs[name]["rgb"], dtype=np.uint8).transpose(2, 0, 1)
            if self._swap_rb:
                # (3,H,W) -> reverse the channel axis: RGB -> BGR. See RW2_SWAP_RB
                # in __init__.
                rgb = np.ascontiguousarray(rgb[::-1])
            depth = np.asarray(camera_obs[name]["depth"], dtype=np.float32)
            if rgb.shape[-2:] != depth.shape[-2:]:
                raise ValueError(
                    f"Online RoboTwin camera `{name}` has mismatched RGB/depth "
                    f"shapes: rgb={rgb.shape[-2:]}, depth={depth.shape[-2:]}."
                )
            rgb_arrays.append(rgb)
            depth_arrays.append(depth)

        image_shapes = {rgb.shape[-2:] for rgb in rgb_arrays}
        if len(image_shapes) != 1:
            raise ValueError(
                "Online RoboTwin evaluation expects head/left/right cameras at "
                f"the same resolution, got {sorted(image_shapes)}."
            )

        rgbs_np = np.stack(rgb_arrays, axis=0)[None]
        depths_np = np.stack(depth_arrays, axis=0)[None]
        rgbs_t = torch.from_numpy(rgbs_np)
        depths_t = torch.from_numpy(depths_np)
        extrinsics_t = torch.from_numpy(extrinsics[None])
        intrinsics_t = torch.from_numpy(intrinsics[None])

        rgbs, pcds = self.preprocessor.process_obs(
            rgbs_t,
            depths_t,
            extrinsics_t,
            intrinsics_t,
        )
        proprio = self.preprocessor.process_proprio(proprio)
        obs_dict = {"rgb": rgbs, "pcd": pcds, "proprio": proprio}
        if self._is_joint_model:
            joint_proprio = self._extract_joint_proprio(observation)
            obs_dict["joint_proprio"] = joint_proprio.to(
                self.args.device, non_blocking=True
            ).float()
            obs_dict["joint_anchor"] = self._current_joint_anchor(observation)
        return obs_dict

    def _build_eval_grounding_kwargs(self, batch_size, device):
        if self.args.model_type != "flower_actor_3d_grounding_cond":
            return {}
        return {
            "grounding_3d": torch.zeros(
                batch_size, 3, device=device, dtype=torch.float32
            ),
            "grounding_keep_mask": torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
        }

    def _tokenize_instruction(self, instruction):
        instr_mask = None
        instructions = [instruction]
        if not self.args.pre_tokenize:
            instr = instructions
        else:
            tok_out = self.tokenizer(instructions)
            if isinstance(tok_out, torch.Tensor):
                instr = tok_out.to(self.args.device, non_blocking=True)
            else:
                instr = tok_out.input_ids.to(self.args.device, non_blocking=True)
                instr_mask = tok_out.attention_mask.to(self.args.device, non_blocking=True)
        return instr, instr_mask

    def _euler_to_quat_wxyz_np(self, euler: np.ndarray) -> np.ndarray:
        """Convert (N, 3) XYZ Euler angles (rad) to (N, 4) wxyz quaternions."""
        t = torch.from_numpy(euler.astype(np.float32))
        mat = euler_xyz_to_matrix(t)  # (N, 3, 3)
        quat_wxyz = matrix_to_quaternion(mat).cpu().numpy()  # (N, 4) wxyz
        return _normalise_quaternion_positive_w(quat_wxyz)

    def _quat_proprio_to_euler(self, proprio: np.ndarray) -> np.ndarray:
        """Convert (nhand, 8) quat proprio to (nhand, 7) euler proprio for relative euler mode."""
        from utils import pytorch3d_transforms as p3d
        quat_wxyz = torch.from_numpy(proprio[:, 3:7].astype(np.float32))
        mat = p3d.quaternion_to_matrix(quat_wxyz)  # (nhand, 3, 3)
        # matrix_to_euler_xyz is defined in datasets/utils.py convention: XYZ extrinsic
        m00, m10 = mat[:, 0, 0], mat[:, 1, 0]
        m11, m12 = mat[:, 1, 1], mat[:, 1, 2]
        m20, m21, m22 = mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2]
        sy = torch.sqrt(m00 ** 2 + m10 ** 2)
        singular = sy < 1e-6
        ex = torch.where(singular, torch.atan2(-m12, m11), torch.atan2(m21, m22))
        ey = torch.atan2(-m20, sy)
        ez = torch.where(singular, torch.zeros_like(m10), torch.atan2(m10, m00))
        euler = torch.stack([ex, ey, ez], dim=-1).cpu().numpy()  # (nhand, 3)
        return np.concatenate([proprio[:, :3], euler, proprio[:, 7:]], axis=-1)  # (nhand, 7)

    def _apply_gripper_latch(self, raw_openness):
        # Sticky-grasp latch over the executed command stream (state persists on
        # self across chunks, reset per episode in reset_obs). The model runs
        # open-loop over a 16-step chunk and its per-step p_close is noisy, so a
        # single low frame mid-grasp would otherwise reopen the gripper and drop
        # the object. Schmitt-trigger + debounce:
        #   - close immediately when p_close >= close_hi,
        #   - reopen only after `min_open` CONSECUTIVE frames with p_close <= open_lo.
        # raw_openness: (T, num_arms) of p_close. Returns env grip (0=close,1=open).
        close_hi = self.args.gripper_close_thresh
        if close_hi is None:
            close_hi = float(self.args.binary_closed_prob_threshold)
        close_hi = float(close_hi)
        open_lo = float(self.args.gripper_open_thresh)
        min_open = int(self.args.gripper_min_open_steps)
        T, n = raw_openness.shape
        out = np.ones((T, n), dtype=np.float32)
        for a in range(n):
            for t in range(T):
                p = float(raw_openness[t, a])
                if not self._grip_closed[a]:
                    if p >= close_hi:
                        self._grip_closed[a] = True
                        self._grip_open_run[a] = 0
                else:
                    if p <= open_lo:
                        self._grip_open_run[a] += 1
                        if self._grip_open_run[a] >= min_open:
                            self._grip_closed[a] = False
                            self._grip_open_run[a] = 0
                    else:
                        self._grip_open_run[a] = 0
                out[t, a] = 0.0 if self._grip_closed[a] else 1.0
        return out

    def _convert_model_actions_to_env(self, trajectory, debug=False, apply_gripper_filter=False):
        # trajectory is either (T, 2, 8) with model-convention quaternions or
        # (T, 2, 7) with model-convention Euler rotations.
        # Returns (T, 2, 8) in RoboTwin's live wxyz quaternion convention.
        if self.args.rotation_format == "euler" and trajectory.shape[-1] == 7:
            T, num_arms, _ = trajectory.shape
            model_traj = np.zeros((T, num_arms, 8), dtype=np.float32)
            model_traj[:, :, :3] = trajectory[:, :, :3]
            model_traj[:, :, -1] = np.clip(trajectory[:, :, -1], 0.0, 1.0)
            for arm_i in range(num_arms):
                model_traj[:, arm_i, 3:7] = self._euler_to_quat_wxyz_np(
                    trajectory[:, arm_i, 3:6]
                )
        elif trajectory.shape[-1] == 8:
            model_traj = trajectory.copy()
        else:
            raise ValueError(
                "Unexpected action trajectory shape for RobotWin2 eval: "
                f"{trajectory.shape}."
            )

        env_traj = model_traj.copy()
        for hand_idx in range(env_traj.shape[1]):
            env_traj[:, hand_idx, 3:7] = _normalise_quaternion(
                np.stack(
                    [
                        self._to_env_quat(quat)
                        for quat in env_traj[:, hand_idx, 3:7]
                    ],
                    axis=0,
                )
            )
        raw_openness = np.clip(env_traj[..., -1], 0.0, 1.0)  # p_close in binary mode
        latched = False
        if self.args.continuous_openness:
            env_traj[..., -1] = raw_openness
        else:
            thr = float(self.args.binary_closed_prob_threshold)
            naive_grip = np.where(raw_openness >= thr, 0.0, 1.0)
            if apply_gripper_filter and bool(getattr(self.args, "gripper_latch", False)):
                env_traj[..., -1] = self._apply_gripper_latch(raw_openness)
                latched = True
            else:
                env_traj[..., -1] = naive_grip
        # DEBUG: log model gripper openness prediction per rollout step.
        if debug:
            if self.args.continuous_openness:
                mode = "continuous"
            elif latched:
                close_hi = self.args.gripper_close_thresh
                if close_hi is None:
                    close_hi = float(self.args.binary_closed_prob_threshold)
                mode = (
                    f"binary+latch(close>={float(close_hi):.2f}, "
                    f"reopen<={float(self.args.gripper_open_thresh):.2f}"
                    f"x{int(self.args.gripper_min_open_steps)})"
                )
            else:
                mode = f"binary(close_thr={float(self.args.binary_closed_prob_threshold):.2f})"
            with np.printoptions(precision=3, suppress=True):
                print(
                    f"[gripper-debug] {mode} raw_p_close(L,R)=\n{raw_openness}\n"
                    f"  -> env_grip(0=close,1=open)=\n{env_traj[..., -1]}",
                    flush=True,
                )
        return env_traj

    def _relative_to_absolute(self, actions, anchor):
        # actions follows robotwin2_to_zarr.py anchor-relative schema:
        #   xyz      = every future_xyz - current anchor_xyz
        #   rotation = absolute future quaternion, or q_future * inverse(q_anchor)
        #              when args.relative_rotation is true
        #   grip     = raw openness if continuous, else predicted p_closed
        # anchor: (2, 8) absolute pose in the model convention.
        if self.args.rotation_format == "euler":
            raise RuntimeError("Use _relative_euler_to_absolute_model for euler deltas.")

        # quat format: (T, 2, 8)
        out = actions.copy()
        out[:, :, :3] = anchor[None, :, :3] + actions[:, :, :3]
        action_quat = _normalise_quaternion(actions[:, :, 3:7])
        if bool(getattr(self.args, "relative_rotation", False)):
            out[:, :, 3:7] = _normalise_quaternion_positive_w(
                _multiply_quaternion_wxyz(action_quat, anchor[None, :, 3:7])
            )
        else:
            out[:, :, 3:7] = action_quat
        out[:, :, -1] = np.clip(actions[:, :, -1], 0.0, 1.0)
        return out

    def _relative_euler_to_absolute_model(self, actions, anchor):
        # actions: (T, 2, 7) = relative xyz + relative XYZ Euler + grip
        # (raw openness if continuous, else p_closed).
        # anchor: (2, 8) absolute TCP pose in the model's dataset quaternion convention.
        # The training labels convert relative wxyz quaternions to Euler, so at
        # inference we convert each Euler delta back to a quaternion and compose.
        abs_actions = []
        prev = anchor.copy()
        for step in actions:
            nxt = np.zeros_like(prev)
            nxt[:, :3] = prev[:, :3] + step[:, :3]
            delta_quat = self._euler_to_quat_wxyz_np(step[:, 3:6])
            nxt[:, 3:7] = _normalise_quaternion_positive_w(
                _multiply_quaternion_wxyz(delta_quat, prev[:, 3:7])
            )
            nxt[:, -1] = np.clip(step[:, -1], 0.0, 1.0)
            abs_actions.append(nxt)
            prev = nxt
        return np.stack(abs_actions, axis=0)

    def _absolute_to_relative_env(self, actions, anchor):
        # actions: (T, 2, 8), anchor: (2, 8), both in RoboTwin wxyz convention
        rel_actions = []
        prev = anchor.copy()
        for step in actions:
            delta = step.copy()
            delta[:, :3] = step[:, :3] - prev[:, :3]
            delta[:, 3:7] = _normalise_quaternion_positive_w(
                _multiply_quaternion_wxyz(
                    step[:, 3:7],
                    _invert_quaternion_wxyz(prev[:, 3:7]),
                )
            )
            # RoboTwin delta_ee applies TCP deltas but still reads gripper as an absolute target.
            delta[:, -1] = step[:, -1]
            rel_actions.append(delta)
            prev = step
        return np.stack(rel_actions, axis=0)

    def _tcp_targets_to_ee_commands(self, actions):
        # Current RoboTwin2 zarr labels are TCP/gripper-center poses, while
        # TASK_ENV.take_action(action_type="ee") tracks the robot EE/link frame.
        # Shift each desired TCP target back along local +x before sending.
        ee_actions = actions.copy()
        rot = _quaternion_wxyz_to_matrix_np(ee_actions[:, :, 3:7])
        local_offset = np.array(
            [ROBOTWIN_TCP_TO_EE_OFFSET_M, 0.0, 0.0],
            dtype=np.float32,
        )
        ee_actions[:, :, :3] -= np.einsum("...ij,j->...i", rot, local_offset)
        return ee_actions

    def _current_anchor_in_env_format(self, obs_dict):
        anchor = obs_dict["proprio"][0, -1].detach().cpu().numpy()[None]
        return self._convert_model_actions_to_env(anchor)[0]

    @torch.no_grad()
    def _step_joint(self, obs_dict, instruction):
        """Joint (qpos) rollout. The model predicts (B, T, nhand=2, 7): per arm
        6 anchor-relative arm-joint deltas + a gripper open-probability. We add
        the current absolute arm joints (anchor) to the deltas, binarize the
        gripper, and assemble a flat 14-dim RoboTwin qpos command per step:
        [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]."""
        batch_size = obs_dict["rgb"].shape[0]
        traj_mask = torch.full(
            (batch_size, self.args.prediction_len, 2),
            False,
            device=self.args.device,
            dtype=torch.bool,
        )
        instr, instr_mask = self._tokenize_instruction(instruction)
        joints = self.policy(
            None,
            traj_mask,
            obs_dict["rgb"],
            None,
            obs_dict["pcd"],
            instr,
            obs_dict["proprio"],
            run_inference=True,
            instr_mask=instr_mask,
            joint_proprio=obs_dict["joint_proprio"],
        ).view(batch_size, self.args.prediction_len, 2, 7)
        joints = joints[0].detach().cpu().numpy()  # (T, 2, 7)

        arm_delta = joints[..., :6]                 # anchor-relative arm-joint deltas
        grip_prob = joints[..., 6]                  # P(open)
        anchor = obs_dict["joint_anchor"]           # (2, 6) absolute current arm joints
        abs_arm = anchor[None] + arm_delta          # (T, 2, 6)

        if self.args.continuous_openness:
            env_grip = np.clip(grip_prob, 0.0, 1.0)
        else:
            # env gripper convention: 1=open, 0=close (matches set_gripper [0,1]).
            env_grip = np.where(grip_prob >= 0.5, 1.0, 0.0).astype(np.float32)

        # Flat qpos layout: [left_arm(6), left_grip, right_arm(6), right_grip].
        trajectory = np.concatenate(
            [
                abs_arm[:, 0, :],
                env_grip[:, 0:1],
                abs_arm[:, 1, :],
                env_grip[:, 1:2],
            ],
            axis=1,
        ).astype(np.float64)  # (T, 14)
        return trajectory

    @torch.no_grad()
    def step(self, obs_dict, instruction):
        if self._is_joint_model:
            return self._step_joint(obs_dict, instruction)
        batch_size = obs_dict["rgb"].shape[0]
        # euler format outputs (xyz + euler + grip) = 7D; all other formats: 8D
        action_dim = 7 if self.args.rotation_format == "euler" else 8
        traj_mask = torch.full(
            (batch_size, self.args.prediction_len, 2),
            False,
            device=self.args.device,
            dtype=torch.bool,
        )
        if self._gt_replay:
            # Bypass the policy: use the recorded GT rel_action chunk for the
            # current frame, anchored to the recorded (zarr) pose so we replay the
            # exact demonstrated absolute trajectory. GT openness is 1=open; the
            # downstream conversion expects p_close, so flip it.
            f = min(self._gt_idx, self._gt_T - 1)
            gt = self._gt_rel[f].copy()              # (16, 2, 8) anchor-rel xyz + rel quat + openness
            gt[..., -1] = 1.0 - gt[..., -1]          # openness(1=open) -> p_close(1=closed)
            anchor = self._gt_anchor[f]              # (2, 8) recorded abs TCP pose (model quat conv)
            trajectory = self._relative_to_absolute(gt, anchor)
            self._gt_idx += self.args.prediction_len
            print(f"\033[95m[GT-REPLAY] chunk@frame={f} (cursor->{self._gt_idx})\033[0m", flush=True)
        else:
            instr, instr_mask = self._tokenize_instruction(instruction)
            grounding_kwargs = self._build_eval_grounding_kwargs(
                batch_size=batch_size,
                device=self.args.device,
            )
            trajectory = self.policy(
                None,
                traj_mask,
                obs_dict["rgb"],
                None,
                obs_dict["pcd"],
                instr,
                obs_dict["proprio"],
                run_inference=True,
                instr_mask=instr_mask,
                **grounding_kwargs,
            ).view(batch_size, self.args.prediction_len, 2, action_dim)

            trajectory = trajectory[0].detach().cpu().numpy()  # (T, 2, action_dim)
            if self.args.policy_action_mode == "relative":
                anchor = obs_dict["proprio"][0, -1].detach().cpu().numpy()  # (2, 8) quat
                if self.args.rotation_format == "euler":
                    trajectory = self._relative_euler_to_absolute_model(trajectory, anchor)
                else:
                    trajectory = self._relative_to_absolute(trajectory, anchor)
        trajectory = self._convert_model_actions_to_env(
            trajectory, debug=False, apply_gripper_filter=True
        )
        if self.args.robotwin_action_type == "delta_ee":
            anchor = self._current_anchor_in_env_format(obs_dict)
            trajectory = self._absolute_to_relative_env(trajectory, anchor)
        elif self.args.robotwin_action_type == "ee" and self.args.robotwin_action_frame == "tcp":
            trajectory = self._tcp_targets_to_ee_commands(trajectory)
        return trajectory.reshape(self.args.prediction_len, -1)

    def get_action(self, instruction):
        observation = self._require_latest_obs()
        recorder = getattr(self, "_rollout_recorder", None)
        if recorder is not None:
            recorder.note_instruction(instruction)
        obs_dict = self.preprocess(observation)
        return self.step(obs_dict, instruction)


def create_model(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict)
    del ckpt
    model = Model(args)
    model.load_pretrained_weights(state_dict)
    return model


def build_eval_args(user_args):
    user_args = dict(user_args)
    if "pred_len" in user_args and "prediction_len" not in user_args:
        user_args["prediction_len"] = user_args["pred_len"]

    def _coerce_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y"}:
                return True
            if lowered in {"0", "false", "no", "n", "none", "null"}:
                return False
        return bool(value)

    defaults = {
        "checkpoint": None,
        "model_type": "flower_actor_3d",
        "backbone": "florence2",
        "num_history": 1,
        "prediction_len": 15,
        "pre_tokenize": True,
        "custom_img_size": 224,
        "num_finetune_vision_layers": 0,
        "finetune_backbone": True,
        "finetune_text_encoder": False,
        "fps_subsampling_factor": 1,
        "embedding_dim": 1024,
        "num_attn_heads": 16,
        "num_vis_instr_attn_layers": 3,
        "num_shared_attn_layers": 16,
        "relative_action": True,
        "relative_rotation": False,
        "policy_action_mode": "relative",
        "robotwin_action_type": "ee",
        "robotwin_action_frame": "tcp",
        "robotwin_tcp_quat_convention": "wxyz",
        "rotation_format": "quat_wxyz",
        "denoise_timesteps": 5,
        "denoise_model": "rectified_flow",
        "continuous_openness": False,
        "binary_closed_prob_threshold": 0.5,
        # Sticky-grasp gripper latch (deploy-time stabilization; off by default).
        "gripper_latch": False,
        "gripper_close_thresh": None,   # None -> use binary_closed_prob_threshold
        "gripper_open_thresh": 0.2,     # only reopen when p_close <= this ...
        "gripper_min_open_steps": 3,    # ... for this many consecutive frames
        "motion_expert_config": None,
        "motion_expert_checkpoint": None,
        "motion_solver_steps": 10,
        "motion_steps_to_run": 1,
        "motion_finetune": False,
        "motion_scene_fusion": True,
        "motion_action_fusion": False,
        # Joint (qpos) action models. RoboTwin2 is bimanual with 6 arm joints +
        # 1 binary gripper command per arm (these DIFFER from CALVIN's 7/8).
        "joint_action_dim": 6,
        "joint_proprio_dim": 7,
        "gripper_dim": 1,
        "joint_num_layers": 8,
        "joint_embedding_dim": 0,   # 0 -> share embedding_dim
        "joint_num_attn_heads": 0,  # 0 -> share num_attn_heads
        # 2D ablation (flower_actor_2d*): RoboTwin2 has 3 scene cameras.
        "num_camera_ids": 3,
        "vision_lora_rank": 0,
        "vision_lora_alpha": 32,
        "vision_lora_dropout": 0.05,
        "vision_lora_num_layers": 0,
        "vision_lora_target_modules": "attn.qkv,attn.proj",
        "llm_lora_rank": 0,
        "llm_lora_alpha": 32,
        "llm_lora_dropout": 0.05,
        "llm_lora_num_layers": 0,
        "llm_lora_target_modules": "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        "use_multiscale": False,
        "gradient_checkpointing": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    cfg = defaults.copy()
    cfg.update(user_args)
    for key in (
        "pre_tokenize",
        "finetune_backbone",
        "finetune_text_encoder",
        "relative_action",
        "relative_rotation",
        "continuous_openness",
        "gripper_latch",
        "motion_finetune",
        "motion_scene_fusion",
        "motion_action_fusion",
        "use_multiscale",
        "gradient_checkpointing",
    ):
        cfg[key] = _coerce_bool(cfg[key])
    for key in (
        "joint_action_dim",
        "joint_proprio_dim",
        "gripper_dim",
        "joint_num_layers",
        "joint_embedding_dim",
        "joint_num_attn_heads",
        "num_camera_ids",
    ):
        cfg[key] = int(cfg[key])
    cfg["binary_closed_prob_threshold"] = float(cfg["binary_closed_prob_threshold"])
    cfg["gripper_open_thresh"] = float(cfg["gripper_open_thresh"])
    cfg["gripper_min_open_steps"] = int(cfg["gripper_min_open_steps"])
    if cfg["gripper_close_thresh"] in (None, "", "None", "none", "null"):
        cfg["gripper_close_thresh"] = None
    else:
        cfg["gripper_close_thresh"] = float(cfg["gripper_close_thresh"])
    if "policy_action_mode" not in user_args and bool(cfg["relative_action"]):
        cfg["policy_action_mode"] = "relative"
    if not cfg["checkpoint"]:
        raise ValueError("deploy_policy.yml must provide `checkpoint` for analogical_manipulation.")
    if cfg["rotation_format"] != "quat_wxyz":
        raise ValueError(
            "Current RoboTwin2 checkpoints from robotwin2_to_zarr.py expect "
            "rotation_format `quat_wxyz`, "
            f"got {cfg['rotation_format']!r}."
        )
    if cfg["policy_action_mode"] not in {"absolute", "relative"}:
        raise ValueError(
            "policy_action_mode must be either `absolute` or `relative`, "
            f"got {cfg['policy_action_mode']!r}."
        )
    is_joint_model = cfg["model_type"] in _JOINT_MODEL_TYPES
    if is_joint_model:
        # Joint models are executed via RoboTwin's qpos (joint position) control.
        if "robotwin_action_type" not in user_args:
            cfg["robotwin_action_type"] = "qpos"
        if cfg["robotwin_action_type"] != "qpos":
            raise ValueError(
                f"Joint model {cfg['model_type']!r} must use "
                "robotwin_action_type=`qpos`, got "
                f"{cfg['robotwin_action_type']!r}."
            )
    elif cfg["robotwin_action_type"] not in {"ee", "delta_ee"}:
        raise ValueError(
            "robotwin_action_type must be either `ee` or `delta_ee` "
            "(or `qpos` for joint models), "
            f"got {cfg['robotwin_action_type']!r}."
        )
    if cfg["robotwin_action_frame"] not in {"tcp", "ee"}:
        raise ValueError(
            "robotwin_action_frame must be either `tcp` for current RoboTwin2 "
            "TCP labels or `ee` for legacy pre-shifted EE commands, "
            f"got {cfg['robotwin_action_frame']!r}."
        )
    if cfg["robotwin_tcp_quat_convention"] not in {"wxyz", "legacy_xyzw_to_wxyz"}:
        raise ValueError(
            "robotwin_tcp_quat_convention must be either `wxyz` for the current "
            "RoboTwin2 zarr or `legacy_xyzw_to_wxyz` for older converted "
            f"zarr/checkpoints, got {cfg['robotwin_tcp_quat_convention']!r}."
        )
    return SimpleNamespace(**cfg)
