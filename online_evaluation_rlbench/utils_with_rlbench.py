import os
import glob
import random

import open3d  # DON'T DELETE THIS!
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers, fetch_text_encoders
from online_evaluation_rlbench.get_stored_demos import get_stored_demos
import utils.pytorch3d_transforms as pytorch3d_transforms


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (8,)
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()
        for _ in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            #print('target action:', target)
            obs, reward, terminate = self._task.step(action_collision)
            #print('actual action:', obs.gripper_pose)

            # Check if we reached the desired pose (planner may be inaccurate)
            pos = obs.gripper_pose[:3]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            criteria = (dist_pos < 5e-3,)
            if dist_pos > 5e-3:
                print('planner error:', dist_pos)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy.cuda()
        self._policy.eval()
        self._relative = bool(getattr(self._policy, "_relative", True))
        self._rotation_format = getattr(self._policy, "_rotation_format", "quat_xyzw")
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)
        self.encoder, _ = fetch_text_encoders('clip')
        self.encoder.cuda()
        self.encoder.eval()

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)
        #self._instr = self.encoder(self._instr)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, nhist, ncam, 3, H, W)
            pcds: (1, nhist, ncam, 3, H, W)
            gripper: (B, nhist, output_dim)
            prediction_len: int

        Returns:
            (1, prediction_len, 8)
        """
        real = gripper[:, :, None, :]              # (B, nhist, 1, 8)
        fake = torch.zeros_like(real)               # (B, nhist, 1, 8)
        fake[..., 6] = 1.0                          # identity quat w=1
        proprio_2hand = torch.cat([real, fake], 2)  # (B, nhist, 2, 8)
        pred = self._policy(
            None,
            torch.full([1, prediction_len, 2], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            proprio_2hand,
            run_inference=True
        ).view(1, prediction_len, 2, 8)
        return pred[:,:,:1,:]  # (1, prediction_len, nhand=1, 8)
    
    @property
    def uses_relative_actions(self):
        return self._relative

    def relative_to_absolute_actions(self, rel_actions, anchor_action):
        """
        Convert relative actions to absolute world-frame actions.
        Args:
            rel_actions: (T, 1, 8) tensor in xyz + quat + gripper format
            anchor_action: (1, 8) current absolute state for left/right hands
        """
        if not self._relative:
            return rel_actions
        abs_actions = []
        prev = anchor_action.clone()
        for t in range(rel_actions.shape[0]):
            nxt = rel_actions[t].clone()
            nxt[:, :3] = prev[:, :3] + rel_actions[t, :, :3]
            if self._rotation_format == "quat_xyzw":
                q_rel_wxyz = rel_actions[t, :, 3:7][:, [3, 0, 1, 2]]
                q_prev_wxyz = prev[:, 3:7][:, [3, 0, 1, 2]]
                q_next_wxyz = pytorch3d_transforms.quaternion_multiply(
                    q_rel_wxyz, q_prev_wxyz
                )
                q_next_wxyz = q_next_wxyz / q_next_wxyz.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                nxt[:, 3:7] = q_next_wxyz[:, [1, 2, 3, 0]]
            elif self._rotation_format == "quat_wxyz":
                q_next = pytorch3d_transforms.quaternion_multiply(
                    rel_actions[t, :, 3:7], prev[:, 3:7]
                )
                q_next = q_next / q_next.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                nxt[:, 3:7] = q_next
            else:
                raise ValueError(
                    f"Unsupported rotation format for relative eval: {self._rotation_format}"
                )
            abs_actions.append(nxt)
            prev = nxt
        return torch.stack(abs_actions, dim=0)


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size

    def get_obs_action(self, obs):
        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        obs_rgb = [
            torch.tensor(state_dict["rgb"][i]).float().permute(2, 0, 1) / 255.0
            for i in range(len(state_dict["rgb"]))
        ]
        obs_pc = [
            torch.tensor(state_dict["pc"][i]).float().permute(2, 0, 1)
            if len(state_dict["pc"]) > 0 else None
            for i in range(len(state_dict["rgb"]))
        ]
        state = torch.cat(obs_rgb + obs_pc, dim=0)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1,
        record_video=True
    ):
        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        for demo_id, demo in enumerate(var_demos):
            if record_video:
                frames = []

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                if record_video:
                    frames.append((
                        np.concatenate([
                            rgb[0].cpu().numpy()[2],
                            rgb[0].cpu().numpy()[3]
                        ], -1).transpose(1, 2, 0) * 255
                    ).astype(np.uint8))
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )
                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu()  # (T=1, nhand=1, dim=8)
                    if actioner.uses_relative_actions:
                        anchor = gripper[0].detach().cpu().view(1, 8)
                        actions = actioner.relative_to_absolute_actions(actions, anchor)
                    actions = actions.numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        obs, reward, _ = move(action[0], collision_checking=False)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward
            if record_video:
                rgb, _, _ = self.get_rgb_pcd_gripper_from_obs(obs)
                frames.append((
                    np.concatenate([
                        rgb[0].cpu().numpy()[2],
                        rgb[0].cpu().numpy()[3]
                    ], -1).transpose(1, 2, 0) * 255
                ).astype(np.uint8))

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id + 1}",
                f"SR: {total_reward:.2f}/{demo_id + 1}",
                "# valid demos", demo_id + 1,
            )
            if record_video:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
                clip = ImageSequenceClip(frames, fps=1)
                clip.write_videofile(f"eval_logs/peract_reproduce/{task_str}_{demo_id}.mp4", codec="libx264")

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras,
        **kwargs
    ):
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )

        return obs_config
