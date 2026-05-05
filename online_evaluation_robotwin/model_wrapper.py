import inspect
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.pytorch3d_transforms import quaternion_multiply

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
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        self.policy = self.get_policy()
        self.preprocessor = RoboTwinEvalPreprocessor(
            num_history=self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=RoboTwinDepth2Cloud(),
        )
        self.reset()

    def get_policy(self):
        model_class = fetch_model_class(self.args.model_type)
        model_kwargs = dict(
            backbone=self.args.backbone,
            num_finetune_vision_layers=self.args.num_finetune_vision_layers,
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
        return model_class(**model_kwargs)

    def reset(self):
        self.policy.eval()
        self.policy = self.policy.to(self.args.device)
        self.reset_obs()

    def reset_obs(self):
        self.obs_cache = []
        self._latest_obs = None
        self._proprio_history = []
        self._cached_intrinsics = None
        self._cached_head_extrinsics = None

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

    def _get_extrinsics(self, camera_obs):
        if self._cached_head_extrinsics is None:
            self._cached_head_extrinsics = np.asarray(
                camera_obs["head_camera"]["extrinsic_cv"], dtype=np.float32
            )
        return np.stack(
            [
                self._cached_head_extrinsics,
                np.asarray(camera_obs["left_camera"]["extrinsic_cv"], dtype=np.float32),
                np.asarray(camera_obs["right_camera"]["extrinsic_cv"], dtype=np.float32),
            ],
            axis=0,
        )

    def _to_model_quat(self, quat_wxyz):
        if self.args.rotation_format == "quat_xyzw":
            return quat_wxyz[[1, 2, 3, 0]]
        return quat_wxyz

    def _to_env_quat(self, quat):
        if self.args.rotation_format == "quat_xyzw":
            return quat[[3, 0, 1, 2]]
        return quat

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

    def _extract_proprio(self, observation):
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
        current = np.stack(
            [
                np.concatenate([left_pose, [float(endpose["left_gripper"])]], axis=0),
                np.concatenate([right_pose, [float(endpose["right_gripper"])]], axis=0),
            ],
            axis=0,
        ).astype(np.float32)  # (2, 8)

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

    @torch.no_grad()
    def preprocess(self, observation):
        camera_obs = self._get_camera_dict(observation)
        rgb = np.stack(
            [
                np.asarray(camera_obs[name]["rgb"], dtype=np.uint8).transpose(2, 0, 1)
                for name in self.CAMERA_NAMES
            ],
            axis=0,
        )
        depth = np.stack(
            [
                np.asarray(camera_obs[name]["depth"], dtype=np.float32)
                for name in self.CAMERA_NAMES
            ],
            axis=0,
        )
        intrinsics = self._get_intrinsics(camera_obs)
        extrinsics = self._get_extrinsics(camera_obs)
        proprio = self._extract_proprio(observation)

        rgb_tensor = torch.from_numpy(rgb[None])
        depth_tensor = torch.from_numpy(depth[None])
        extrinsics_tensor = torch.from_numpy(extrinsics[None])
        intrinsics_tensor = torch.from_numpy(intrinsics[None])

        rgbs, pcds = self.preprocessor.process_obs(
            rgb_tensor,
            depth_tensor,
            extrinsics_tensor,
            intrinsics_tensor,
        )
        proprio = self.preprocessor.process_proprio(proprio)
        return {"rgb": rgbs, "pcd": pcds, "proprio": proprio}

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

    def _convert_model_actions_to_env(self, trajectory):
        # trajectory: (T, 2, 8) in model quaternion convention
        env_traj = trajectory.copy()
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
        env_traj[..., -1] = np.clip(env_traj[..., -1], 0.0, 1.0)
        return env_traj

    def _relative_to_absolute(self, actions, anchor):
        # actions: (T, 2, 8), anchor: (2, 8), both in model quaternion convention
        abs_actions = []
        prev = anchor.copy()
        for step in actions:
            nxt = step.copy()
            nxt[:, :3] = prev[:, :3] + step[:, :3]
            nxt[:, 3:7] = self._quat_multiply_in_model_format(
                step[:, 3:7],
                prev[:, 3:7],
            )
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
            delta[:, 3:7] = _normalise_quaternion(
                _multiply_quaternion_wxyz(
                    step[:, 3:7],
                    _invert_quaternion_wxyz(prev[:, 3:7]),
                )
            )
            rel_actions.append(delta)
            prev = step
        return np.stack(rel_actions, axis=0)

    def _current_anchor_in_env_format(self, obs_dict):
        anchor = obs_dict["proprio"][0, -1].detach().cpu().numpy()[None]
        return self._convert_model_actions_to_env(anchor)[0]

    @torch.no_grad()
    def step(self, obs_dict, instruction):
        batch_size = obs_dict["rgb"].shape[0]
        action_dim = 8
        traj_mask = torch.full(
            (batch_size, self.args.prediction_len, 2),
            False,
            device=self.args.device,
            dtype=torch.bool,
        )
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

        trajectory = trajectory[0].detach().cpu().numpy()  # (T, 2, 8)
        if self.args.policy_action_mode == "relative":
            anchor = obs_dict["proprio"][0, -1].detach().cpu().numpy()
            trajectory = self._relative_to_absolute(trajectory, anchor)
        trajectory = self._convert_model_actions_to_env(trajectory)
        if self.args.robotwin_action_type == "delta_ee":
            anchor = self._current_anchor_in_env_format(obs_dict)
            trajectory = self._absolute_to_relative_env(trajectory, anchor)
        return trajectory.reshape(self.args.prediction_len, -1)

    def get_action(self, instruction):
        observation = self._require_latest_obs()
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
    defaults = {
        "checkpoint": None,
        "model_type": "flower_actor_3d",
        "backbone": "clip",
        "num_history": 1,
        "prediction_len": 10,
        "pre_tokenize": True,
        "custom_img_size": 224,
        "num_finetune_vision_layers": 0,
        "fps_subsampling_factor": 5,
        "embedding_dim": 144,
        "num_attn_heads": 9,
        "num_vis_instr_attn_layers": 2,
        "num_shared_attn_layers": 16,
        "relative_action": False,
        "policy_action_mode": "absolute",
        "robotwin_action_type": "ee",
        "rotation_format": "quat_wxyz",
        "denoise_timesteps": 10,
        "denoise_model": "rectified_flow",
        "motion_expert_config": None,
        "motion_expert_checkpoint": None,
        "motion_solver_steps": 10,
        "motion_steps_to_run": 1,
        "motion_finetune": False,
        "motion_scene_fusion": True,
        "motion_action_fusion": False,
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
    if "policy_action_mode" not in user_args and bool(cfg["relative_action"]):
        cfg["policy_action_mode"] = "relative"
    if not cfg["checkpoint"]:
        raise ValueError("deploy_policy.yml must provide `checkpoint` for analogical_manipulation.")
    if cfg["rotation_format"] not in {"quat_wxyz", "quat_xyzw"}:
        raise ValueError(
            "RoboTwin online evaluation currently supports only quaternion action spaces "
            f"(`quat_wxyz` or `quat_xyzw`), got {cfg['rotation_format']!r}."
        )
    if cfg["policy_action_mode"] not in {"absolute", "relative"}:
        raise ValueError(
            "policy_action_mode must be either `absolute` or `relative`, "
            f"got {cfg['policy_action_mode']!r}."
        )
    if cfg["robotwin_action_type"] not in {"ee", "delta_ee"}:
        raise ValueError(
            "robotwin_action_type must be either `ee` or `delta_ee`, "
            f"got {cfg['robotwin_action_type']!r}."
        )
    return SimpleNamespace(**cfg)
