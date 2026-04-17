import inspect
import numpy as np, torch
from modeling.encoder.text import fetch_tokenizers
from modeling.policy import fetch_model_class
from utils.data_preprocessors import fetch_data_preprocessor
from utils.depth2cloud import fetch_depth2cloud
from utils.pytorch3d_transforms import quaternion_to_axis_angle
import robosuite.utils.transform_utils as T
from online_evaluation_libero.utils_with_libero import (
    convert_depth_batch_to_meters,
    compose_extrinsics,
    extract_rgb_depth,
)

class Model:
    def __init__(self, args):
        self.args = args
        self.policy = self.get_policy()
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
        return model_class(**model_kwargs)

    def reset(self):
        self.policy.eval()
        self.policy = self.policy.to(self.args.device)
        # Rolling proprio buffer for temporal history.
        # The policy expects proprio shaped like (B, nhist, 1, proprio_dim) where the nhist dimension corresponds to real timesteps [t-nhist+1, ..., t].
        self._proprio_history = []
        self._last_vec_env_id = None

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

    def _extract_proprio(self, obs_list):
        proprio_list = []  # (B, proprio_dim), one per parallel env
        for obs in obs_list:
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            
            # NOTE: if use quat proprio
            # quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            # quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            axis_angle = T.quat2axisangle(obs["robot0_eef_quat"])

            openness = np.abs(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[0])
            proprio = np.concatenate([eef_pos, axis_angle, [openness]], axis=0)
            proprio_list.append(proprio)

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
        action_dim = 7 if self.args.rotation_format in ("axisangle", "euler") else 13
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
        ).view(batch_size, self.args.prediction_len, 1, action_dim)

        # Use first step action for online control.
        actions = trajectory[:, 0, 0, :].detach().cpu().numpy()
        return actions


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
