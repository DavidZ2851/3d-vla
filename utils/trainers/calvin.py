import gc

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

try:
    from online_evaluation_calvin.evaluate_policy import evaluate_sequence
    from online_evaluation_calvin.utils_with_calvin import get_env
    from online_evaluation_calvin.multistep_sequences import get_sequences
except ImportError:
    pass
from .base import BaseTrainTester


class CALVINTrainTester(BaseTrainTester):

    @torch.no_grad()
    def get_workspace_normalizer(self):
        # return nn.Parameter(torch.tensor([
        #     [-0.02, -0.02, -0.02, -0.05, -0.05, -0.05],
        #     [ 0.02,  0.02,  0.02,  0.05,  0.05,  0.05]
        # ]), requires_grad=False)
        # NOTE: use 6d for euler, no need to normalize rot
        return nn.Parameter(torch.tensor([
            [-0.02, -0.02, -0.02],
            [ 0.02,  0.02,  0.02]
        ]), requires_grad=False)

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["wrist_depth"], sample["extrinsics_wrist"],
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
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val'):
        """Run a given number of evaluation steps."""
        if split == 'train':
            return super().evaluate_nsteps(
                model, loader, step_id, val_iters, split
            )
        elif split == 'val':
            return super().evaluate_nsteps(
                model, loader, step_id, val_iters, split
            )
        # Run online eval for val set
        model.eval()
        env = get_env(
            "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml",
            show_gui=False
        )
        _model = EvalWrapper(
            model,
            self.preprocessor,
            self.tokenizer,
            self.args.pre_tokenize,
            self.args.relative_action
        )
        results = evaluate_policy(
            _model,
            env,
            task_config_file='online_evaluation_calvin/configs/new_playtable_tasks.yaml',
            ann_config_file='online_evaluation_calvin/configs/new_playtable_validation.yaml'
        )
        if step_id > -1:
            self.writer.add_scalar('online_1st_task', results, step_id)
        # Also log to terminal
        print(f"Step {step_id}:")
        print(f"SR: {results:.03f}")
        env.close()
        del env
        gc.collect()
        return -results


class EvalWrapper:

    def __init__(self, policy, preprocessor, tokenizer,
                 pre_tokenize=True, relative_action=True, pred_len=10):
        self.policy = policy
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.pre_tokenize = pre_tokenize
        self.relative_action = relative_action
        self.pred_len = pred_len

    def reset(self):
        """Set model to evaluation mode."""
        pass

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
        # Process inputs
        trajectory_mask = torch.full([self.pred_len], False).cuda(non_blocking=True)
        trajectory_mask = trajectory_mask[None, :, None]

        instr = (
            [instruction,] if not self.pre_tokenize
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
        trajectory = trajectory.view(1, self.pred_len, 7)  # (1, T, 7)
        trajectory = trajectory.data.cpu().numpy()
        trajectory[..., -1] = 2 * (trajectory[..., -1] >= 0.5).astype(int) - 1

        # Back to absolute actions
        if bool(self.relative_action):
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


def evaluate_policy(model, env, task_config_file, ann_config_file):
    """Run this function to evaluate a model on the CALVIN challenge."""
    # Load configs for tasks and instructions
    task_cfg = OmegaConf.load(task_config_file)
    task_oracle = hydra.utils.instantiate(task_cfg)
    instr_dict = OmegaConf.load(ann_config_file)

    # Evaluate on 100 sequences, for the first task only
    eval_sequences = get_sequences(100)

    # Loop over all sampled test sequences
    results = []
    for (initial_state, eval_sequence) in tqdm(eval_sequences):
        result, _ = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, instr_dict, save_video=False, first_only=True,
            verbose=False
        )
        results.append(result)

    return sum(results) / len(results)
