import logging

import numpy as np
import torch

from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.depth2cloud import fetch_depth2cloud
from utils.data_preprocessors import fetch_data_preprocessor


logger = logging.getLogger(__name__)


class Model:

    def __init__(self, args):
        self.args = args
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
        return model_class(
            backbone=self.args.backbone,
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
            denoise_model=self.args.denoise_model
        )

    def reset(self):
        """Set model to evaluation mode."""
        device = self.args.device
        self.policy.eval()
        self.policy = self.policy.to(device)

    def load_pretrained_weights(self):
        state_dict = torch.load(self.args.checkpoint, map_location="cpu")["ema_weight"]
        model_weights = {}
        for key in state_dict:
            _key = key[7:]
            model_weights[_key] = state_dict[key]
        print(f'Loading weights from {self.args.checkpoint}')
        self.policy.load_state_dict(model_weights, strict=True)

    @torch.no_grad()
    def preprocess(self, obs, env):
        # proprio should be (B=1, nhist=1, nhand=1, 7)
        proprio = self.preprocessor.process_actions(
            torch.from_numpy(np.concatenate([
                obs['robot_obs'][:6],
                (obs['robot_obs'][[-1]] + 1) / 2
            ], -1))[None, None, None].float()
        )
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
                proprio: (B=1, nhist=1, nhand=1, 7)
            }
            instruction: str, the instruction of the task

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Process inputs
        trajectory_mask = torch.full([self.args.pred_len], False).to(device)
        trajectory_mask = trajectory_mask[None, :, None]

        instr = (
            [instruction,] if not self.args.pre_tokenize
            else self.tokenizer([instruction,]).cuda(non_blocking=True)
        )

        # Forward pass
        trajectory = self.policy(
            None,
            trajectory_mask,  # (1, T, 1)
            obs['rgb'],  # (1, 2, 3, h, w)
            None,  # (1, 1, 3, 84, 84)
            obs['pcd'],  # (1, 2, 3, h, w)
            instr,
            obs['proprio'],  # (1, nhist, 1, 7)
            run_inference=True
        )  # (1, T, 1, 7)
        trajectory = trajectory.view(1, self.args.pred_len, 7)  # (1, T, 7)
        trajectory = trajectory.data.cpu().numpy()
        trajectory[..., -1] = 2 * (trajectory[..., -1] >= 0.5).astype(int) - 1

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
    model = Model(args)
    model.load_pretrained_weights()
    return model
