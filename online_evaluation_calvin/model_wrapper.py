import logging
import inspect

import numpy as np
import torch

from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.depth2cloud import fetch_depth2cloud
from utils.data_preprocessors import fetch_data_preprocessor


logger = logging.getLogger(__name__)


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
    # Qwen checkpoints in this repo can store LLM LoRA keys as either:
    # - "...language_model.layers.<id>..."
    # - "...language_model.base_model.model.layers.<id>..."
    # We treat any LoRA key under `language_model` as LLM LoRA.
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


def _checkpoint_uses_multiscale(state_dict):
    return any(
        key.startswith("encoder.multiscale_fuser.")
        for key in state_dict
    )


def _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict):
    if args.model_type not in {
        "qwen_flower_actor_3d",
        "qwen_flower_actor_3d_stagewise",
        "qwen_flower_actor_3d_grounding_cond",
        "qwen_flower_actor_3d_grounding_pred",
        "qwen_pi0_llm_actor_3d",
        "qwen_pi05_llm_actor_3d",
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


class Model:

    def __init__(self, args):
        self.args = args
        # Default for evaluate_policy.rollout (absolute TCP after any _to_abs in step())
        self.calvin_action_exec = "cartesian_abs"
        self.policy = self.get_policy()
        self.preprocessor = fetch_data_preprocessor('calvin')(
            keypose_only=False,
            num_history=1,
            custom_imsize=args.custom_img_size,
            depth2cloud=fetch_depth2cloud('calvin')
        )
        self.reset()

    def get_policy(self):
        """Initialize the model."""
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        model_class = fetch_model_class(self.args.model_type)
        model_kwargs = dict(
            backbone=self.args.backbone,
            num_finetune_vision_layers=self.args.num_finetune_vision_layers,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=1,
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
        """Set model to evaluation mode."""
        device = self.args.device
        self.policy.eval()
        self.policy = self.policy.to(device)

    def load_pretrained_weights(self):
        ckpt = torch.load(self.args.checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        model_weights = _strip_module_prefix(state_dict)
        print(f'Loading weights from {self.args.checkpoint}')
        self.policy.load_state_dict(model_weights, strict=True)

    @torch.no_grad()
    def preprocess(self, obs, env):
        real_proprio = self.preprocessor.process_actions(
            torch.from_numpy(np.concatenate([
                obs['robot_obs'][:6],
                (obs['robot_obs'][[-1]] + 1) / 2
            ], -1))[None, None, None].float()
        )  # (B=1, nhist=1, nhand=1, 7)
        proprio = real_proprio
        
        # NOTE: if pad to bimanual
        # fake_proprio = torch.zeros_like(real_proprio)
        # fake_proprio[..., -1] = 0.5
        # proprio = torch.cat([real_proprio, fake_proprio], 2)  # (B=1, nhist=1, nhand=2, 7)
        
        # rgbs, pcds are (B=1, ncam=2, 3, H, W)
        rgbs, pcds = self.preprocessor.process_obs(
            torch.from_numpy(obs["rgb_obs"]["rgb_static"].transpose(2, 0, 1)[None, None]),
            torch.from_numpy(obs["rgb_obs"]["rgb_gripper"].transpose(2, 0, 1)[None, None]),
            torch.from_numpy(obs["depth_obs"]["depth_static"][None, None]),
            torch.from_numpy(obs["depth_obs"]["depth_gripper"][None, None]),
            torch.from_numpy(np.linalg.inv(np.array(env.cameras[1].view_matrix).reshape((4, 4)).T)[None]),
            augment=False
        )
        return {'rgb': rgbs, 'pcd': pcds, 'proprio': proprio}

    @torch.no_grad()
    def step(self, obs, instruction):
        """
        Args:
            obs: {
                rgb: (B=1, ncam=2, 3, H, W),
                pcd: (B=1, ncam=2, 3, H, W),
                proprio: (B=1, nhist=1, nhand=1(could pad to 2), 7)
            }
            instruction: str, the instruction of the task

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Process inputs
        trajectory_mask = torch.full((1, self.args.pred_len, 1), False, device=device)  
        # NOTE: if pad to bimanual: 
        # trajectory_mask = torch.full((1, self.args.pred_len, 2), False, device=device)

        instr_mask = None
        if not self.args.pre_tokenize:
            instr = [instruction,]
        else:
            tok_out = self.tokenizer([instruction,])
            if isinstance(tok_out, torch.Tensor):
                instr = tok_out.cuda(non_blocking=True)
            else:
                instr = tok_out.input_ids.cuda(non_blocking=True)
                instr_mask = tok_out.attention_mask.cuda(non_blocking=True)

        # Forward pass
        trajectory = self.policy(
            None,
            trajectory_mask,  # (1, T, 1 or pad to 2)
            obs['rgb'],  # (1, 2, 3, h, w)
            None,  # (1, 1, 3, 84, 84)
            obs['pcd'],  # (1, 2, 3, h, w)
            instr,
            obs['proprio'],  # (1, nhist, 1 or pad to 2, 7)
            run_inference=True,
            instr_mask=instr_mask,
        ).view(1, self.args.pred_len, 1, 7)  # (1, T, nhand=1 or pad to 2, 7)  # NOTE: if pad to bimanual: ).view(1, self.args.pred_len, 2, 7)
        trajectory = trajectory.data.cpu().numpy()[:,:,0,:]  # (1, T, 7)
        trajectory[..., -1] = 2 * (trajectory[..., -1] >= 0.5).astype(int) - 1  # openness
        # Back to absolute actions
        if bool(self.args.relative_action):
            trajectory = self._to_abs(
                trajectory,
                obs['proprio'][:, [-1], 0].cpu().numpy()
            )

        return trajectory

    @staticmethod
    def _to_abs(action, proprio):
        assert action.shape[-1] == 7 and len(action.shape) == 3
        assert proprio.shape[-1] == 7 and len(proprio.shape) == 3
        abs_pos_orn = proprio[..., :6] + np.cumsum(action[..., :6], -2)
        return np.concatenate([abs_pos_orn, action[..., 6:]], -1)


def create_model(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    _maybe_override_qwen_eval_args_from_checkpoint(args, state_dict)
    del ckpt
    model = Model(args)
    model.load_pretrained_weights()
    return model
