import inspect
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.data_preprocessors import fetch_data_preprocessor
from utils.depth2cloud import fetch_depth2cloud
import robosuite.utils.transform_utils as T
from online_evaluation_libero.utils_with_libero import (
    convert_depth_batch_to_meters,
    compose_extrinsics,
    extract_rgb_depth,
)

class Model:
    gripper_open_threshold = 0.02
    osc_action_scale = np.asarray(
        [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32
    )

    def __init__(self, args):
        self.args = args
        self.policy = self.get_policy()
        self._configure_binary_gripper()
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        self.preprocessor = fetch_data_preprocessor("libero")(
            keypose_only=False,
            num_history=self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud("libero"),
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
            nhand=1,
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
        # Rolling proprio buffer for temporal history.
        # The policy expects proprio shaped like (B, nhist, 1, proprio_dim) where the nhist dimension corresponds to real timesteps [t-nhist+1, ..., t].
        self._proprio_history = []
        self._last_vec_env_id = None
        self._current_tcp = None

    def load_pretrained_weights(self, weight_dict=None):
        if weight_dict is None:
            state = torch.load(self.args.checkpoint, map_location="cpu", weights_only=True)
            key = "ema_weight" if state.get("ema_weight") is not None else "weight"
            weight_dict = state[key]
        stripped = {}
        for k, v in weight_dict.items():
            if k.startswith("module."):
                stripped[k[7:]] = v
            else:
                stripped[k] = v
        # Keep strict=False for compatibility across model variants.
        self.policy.load_state_dict(stripped, strict=False)
        self._configure_binary_gripper()

    def _configure_binary_gripper(self):
        module = self.policy.module if hasattr(self.policy, "module") else self.policy
        module._continuous_openness = False
        module._signed_binary_openness_output = True

    @staticmethod
    def _signed_gripper_command(gripper):
        """Convert model outputs to robosuite binary gripper commands.

        Original PandaGripper convention is `-1` for open and `+1` for close.
        """
        gripper = np.asarray(gripper, dtype=np.float32)
        if gripper.size and np.nanmin(gripper) >= 0.0 and np.nanmax(gripper) <= 1.0:
            return np.where(gripper >= 0.5, 1.0, -1.0).astype(np.float32)
        return np.where(gripper >= 0.0, 1.0, -1.0).astype(np.float32)

    @classmethod
    def _tcp_target_gripper_command(cls, gripper):
        return cls._signed_gripper_command(gripper)

    @classmethod
    def _robosuite_gripper_command(cls, gripper):
        return cls._signed_gripper_command(gripper)

    def _build_eval_grounding_kwargs(self, batch_size, device):
        """Mirror training-time dropped-grounding behavior at online eval."""
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

    def _extract_proprio(self, obs_list):
        proprio_list = []  # (B, proprio_dim), one per parallel env
        tcp_states = []
        for obs in obs_list:
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            openness = np.abs(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[0])
            binary_openness = float(openness > self.gripper_open_threshold)
            tcp_states.append(np.concatenate([eef_pos, quat_xyzw, [openness]], axis=0))

            if self.args.rotation_format == "euler":
                rot = R.from_quat(quat_xyzw).as_euler("xyz").astype(np.float32)
            elif self.args.rotation_format == "axisangle":
                rot = T.quat2axisangle(quat_xyzw).astype(np.float32)
            else:
                rot = quat_xyzw.astype(np.float32)
            proprio = np.concatenate([eef_pos, rot, [binary_openness]], axis=0)
            proprio_list.append(proprio)

        self._current_tcp = np.asarray(tcp_states, dtype=np.float32)
        proprio_current = np.asarray(proprio_list, dtype=np.float32)  # (B, proprio_dim)
        # Maintain a rolling history across environment steps so the policy sees real
        # proprio at [t-2, t-1, t] (for num_history=3), instead of repeating the current proprio across history slots.
        self._proprio_history.append(proprio_current)
        if len(self._proprio_history) > self.args.num_history:
            self._proprio_history.pop(0)
        # If we don't have enough past frames yet, pad with the oldest available proprio so the tensor shape stays consistent.
        if len(self._proprio_history) < self.args.num_history:
            pad_count = self.args.num_history - len(self._proprio_history)
            pad_item = self._proprio_history[0]
            history = [pad_item] * pad_count + self._proprio_history
        else:
            history = self._proprio_history
        # (B, nhist, proprio_dim) -> (B, nhist, 1, proprio_dim)
        proprio_np = np.stack(history, axis=1)[:, :, None, :]
        return torch.from_numpy(proprio_np)

    @torch.no_grad()
    def preprocess(self, obs_list, vec_env, intrinsics_cache, agent_extrinsics_cache):
        # `evaluate_one_task` recreates `vec_env` per task; clear history so temporal conditioning doesn't leak across tasks.
        if getattr(self, "_last_vec_env_id", None) != id(vec_env):
            self._proprio_history = []
            self._last_vec_env_id = id(vec_env)

        rgb_batch, depth_batch = extract_rgb_depth(obs_list)
        depth_batch = convert_depth_batch_to_meters(vec_env, depth_batch)
        # Only wrist camera pose changes over time; intrinsics and agentview pose are cached.
        from online_evaluation_libero.utils_with_libero import get_wrist_extrinsics

        wrist_extrinsics = get_wrist_extrinsics(vec_env)
        extrinsics = compose_extrinsics(agent_extrinsics_cache, wrist_extrinsics)
        intrinsics = intrinsics_cache
        proprio = self._extract_proprio(obs_list)

        if rgb_batch.shape[-2:] != (224, 224) or depth_batch.shape[-2:] != (224, 224):
            raise ValueError(
                "LIBERO online evaluation expects 224x224 RGBD from agentview and wrist, "
                f"got rgb={rgb_batch.shape[-2:]}, depth={depth_batch.shape[-2:]}."
            )

        rgbs = torch.from_numpy(rgb_batch)
        depths = torch.from_numpy(depth_batch)
        extrinsics = torch.from_numpy(extrinsics)
        intrinsics = torch.from_numpy(intrinsics)

        rgbs, pcds = self.preprocessor.process_obs(
            rgbs, None, depths, extrinsics, intrinsics, augment=False
        )
        proprio = self.preprocessor.process_proprio(proprio)
        return {"rgb": rgbs, "pcd": pcds, "proprio": proprio}

    @torch.no_grad()
    def step(self, obs_dict, instruction):
        batch_size = obs_dict["rgb"].shape[0]
        action_dim = 7 if self.args.rotation_format in ("axisangle", "euler") else 8
        traj_mask = torch.full(
            (batch_size, self.args.prediction_len, 1),
            False,
            device=self.args.device,
            dtype=torch.bool,
        )

        instr_mask = None
        instructions = [instruction] * batch_size
        if not self.args.pre_tokenize:
            instr = instructions
        else:
            tok_out = self.tokenizer(instructions)
            if isinstance(tok_out, torch.Tensor):
                instr = tok_out.cuda(non_blocking=True)
            else:
                instr = tok_out.input_ids.cuda(non_blocking=True)
                instr_mask = tok_out.attention_mask.cuda(non_blocking=True)

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
        ).view(batch_size, self.args.prediction_len, 1, action_dim)

        # Return the whole predicted chunk so the eval loop can execute it sequentially.
        actions = trajectory[:, :, 0, :].detach().cpu().numpy()
        if self.args.relative_action:
            if self.args.libero_controller == "OSC_POSE":
                return self._relative_action_chunk_to_osc(actions)
            if self.args.libero_controller == "TCP_TARGET_POSE":
                return self._relative_action_chunk_to_tcp_target(actions)
            raise ValueError(
                f"Unsupported LIBERO controller for relative_action=True: "
                f"{self.args.libero_controller!r}"
            )
        return self._postprocess_absolute_gripper_chunk(actions)

    def _postprocess_absolute_gripper_chunk(self, actions: np.ndarray) -> np.ndarray:
        out = actions.copy()
        if self.args.libero_controller == "TCP_TARGET_POSE":
            out[..., -1:] = self._tcp_target_gripper_command(out[..., -1:])
        else:
            out[..., -1:] = self._robosuite_gripper_command(out[..., -1:])
        return out.astype(np.float32)

    def _relative_action_chunk_to_tcp_target(self, rel_actions: np.ndarray) -> np.ndarray:
        if rel_actions.ndim != 3:
            raise ValueError(f"Expected rel_actions shaped (B, T, D), got {rel_actions.shape}")

        batch_size, pred_len, _ = rel_actions.shape
        flat_rel_actions = rel_actions.reshape(batch_size * pred_len, -1)

        delta_pos = flat_rel_actions[:, :3]
        gripper_command = self._tcp_target_gripper_command(flat_rel_actions[:, -1:])

        if self.args.rotation_format == "euler":
            rel_quat = R.from_euler("xyz", flat_rel_actions[:, 3:6]).as_quat().astype(np.float32)
        elif self.args.rotation_format == "axisangle":
            rel_quat = np.stack([T.axisangle2quat(x) for x in flat_rel_actions[:, 3:6]], axis=0).astype(np.float32)
        else:
            rel_quat = flat_rel_actions[:, 3:7].astype(np.float32)

        rel_quat_norm = np.linalg.norm(rel_quat, axis=1, keepdims=True)
        rel_quat = rel_quat / np.maximum(rel_quat_norm, 1e-6)
        rel_quat[rel_quat[:, 3] < 0] *= -1.0

        target_mode = getattr(self.args, "libero_tcp_target_mode", "absolute")
        if target_mode == "relative":
            out = np.concatenate([delta_pos, rel_quat, gripper_command], axis=1).astype(np.float32)
            return out.reshape(batch_size, pred_len, -1)
        if target_mode != "absolute":
            raise ValueError(f"Unknown TCP_TARGET_POSE target_mode={target_mode!r}")
        if self._current_tcp is None:
            raise RuntimeError("Missing current TCP state; call preprocess before step.")

        current_pos = self._current_tcp[:, :3]
        current_quat = self._current_tcp[:, 3:7]
        delta_pos = rel_actions[..., :3]

        # Execute the chunk from one reference state, matching CALVIN's chunked rollout logic.
        target_pos = current_pos[:, None, :] + np.cumsum(delta_pos, axis=1)

        target_quat = np.zeros((batch_size, pred_len, 4), dtype=np.float32)
        current_rot = R.from_quat(current_quat)
        for step_idx in range(pred_len):
            step_rel_quat = rel_quat[step_idx::pred_len]
            step_target = (R.from_quat(step_rel_quat) * current_rot).as_quat().astype(np.float32)
            step_target = step_target / np.maximum(
                np.linalg.norm(step_target, axis=1, keepdims=True),
                1e-6,
            )
            step_target[step_target[:, 3] < 0] *= -1.0
            target_quat[:, step_idx, :] = step_target
            current_rot = R.from_quat(step_target)

        out = np.concatenate(
            [target_pos, target_quat, gripper_command.reshape(batch_size, pred_len, 1)],
            axis=-1,
        ).astype(np.float32)
        return out

    def _relative_action_chunk_to_osc(self, rel_actions: np.ndarray) -> np.ndarray:
        if self.args.rotation_format != "axisangle":
            raise ValueError(
                "LIBERO relative OSC_POSE evaluation expects axisangle rotation_format "
                f"for unnormalized OSC deltas, got {self.args.rotation_format!r}."
            )
        osc = np.zeros_like(rel_actions, dtype=np.float32)
        osc[..., :6] = np.clip(
            rel_actions[..., :6] / self.osc_action_scale[None, None, :],
            -1.0,
            1.0,
        )
        osc[..., 6] = self._robosuite_gripper_command(rel_actions[..., 6:7]).reshape(
            rel_actions.shape[0], rel_actions.shape[1]
        )
        return osc


def create_model(args):
    state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(state)
    _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict)
    model = Model(args)
    model.load_pretrained_weights(weight_dict=state_dict)
    return model


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


def _get_checkpoint_lora_summary(state_dict):
    lora_keys = [key for key in state_dict if ".lora_A." in key or ".lora_B." in key]
    has_vision_lora = any(".visual." in key for key in lora_keys)
    has_llm_lora = any(".language_model." in key for key in lora_keys)
    return {"any": bool(lora_keys), "vision": has_vision_lora, "llm": has_llm_lora}


def _get_checkpoint_embedding_weight(state_dict):
    embedding_weight = state_dict.get("encoder.vis_proj.weight")
    if embedding_weight is not None:
        return embedding_weight
    for key, value in state_dict.items():
        if key.startswith("encoder.multiscale_fuser.projections.") and key.endswith(".weight"):
            return value
    return None


def _checkpoint_uses_multiscale(state_dict):
    return any(key.startswith("encoder.multiscale_fuser.") for key in state_dict)


def _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict):
    if args.model_type not in {
        "qwen_flower_actor_3d",
        "qwen_flower_actor_3d_stagewise",
        "qwen_flower_actor_3d_grounding_cond",
        "qwen_flower_actor_3d_grounding_cond_llm_visual",
        "qwen_flower_actor_3d_grounding_cond_qwen3d",
        "qwen_flower_actor_3d_grounding_pred",
        "qwen_pi0_llm_actor_3d",
        "qwen_pi05_llm_actor_3d",
        "qwen_dualexpert_3dfa_actor",
    }:
        return

    embedding_weight = _get_checkpoint_embedding_weight(state_dict)
    if embedding_weight is not None:
        ckpt_embedding_dim = int(embedding_weight.shape[0])
        if args.embedding_dim != ckpt_embedding_dim:
            print(
                f"Overriding embedding_dim from checkpoint: "
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
            "Checkpoint contains vision LoRA weights, but evaluation did not enable matching vision LoRA args. "
            "Please pass the matching vision_lora_* arguments used during training."
        )
    if lora_summary["llm"] and not requested_llm_lora:
        raise ValueError(
            "Checkpoint contains LLM LoRA weights, but evaluation did not enable matching llm_lora_* args."
        )
    if requested_vision_lora and not lora_summary["vision"] and lora_summary["any"]:
        raise ValueError(
            "Evaluation requested vision LoRA, but the checkpoint does not contain vision LoRA weights."
        )
    if requested_llm_lora and not lora_summary["llm"] and lora_summary["any"]:
        raise ValueError(
            "Evaluation requested LLM LoRA, but the checkpoint does not contain LLM LoRA weights."
        )
