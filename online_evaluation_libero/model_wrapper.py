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
        return model_class(
            backbone=self.args.backbone,
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

    def reset(self):
        self.policy.eval()
        self.policy = self.policy.to(self.args.device)
        # Rolling proprio buffer for temporal history.
        # The policy expects proprio shaped like (B, nhist, 1, proprio_dim) where the nhist dimension corresponds to real timesteps [t-nhist+1, ..., t].
        self._proprio_history = []
        self._last_vec_env_id = None

    def load_pretrained_weights(self):
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
    model = Model(args)
    model.load_pretrained_weights()
    return model
